import os
import random

import torch
import torch.nn as nn
from torch import optim

from src.utils import USE_CUDA, SOS_token, normalize_string

from .data_processor import DataProcessor
from .encoder import EncoderRNN
from .decoder import LuongAttnDecoderRNN
from .search_decoder import GreedySearchDecoder


class Model:
    def __init__(
            self,
            model_name='cb_model',
            attn_model='dot',
            # attn_model = 'general'
            # attn_model = 'concat'
            hidden_size=500,
            encoder_n_layers=2,
            decoder_n_layers=2,
            dropout=0.1,
            batch_size=64,
            clip=50.0,
            teacher_forcing_ratio=1.0,
            learning_rate=0.0001,
            decoder_learning_ratio=5.0,
            n_iteration=4000,
            print_every=1,
            save_every=500,
            checkpoint_iter=4000
    ):
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')
        self.processor = DataProcessor()
        self.encoder = None
        self.decoder = None
        self.embedding = None

        # Configure models
        self.model_name = model_name
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.encoder_n_layers = encoder_n_layers
        self.decoder_n_layers = decoder_n_layers
        self.dropout = dropout
        self.batch_size = batch_size

        # Configure training/optimization
        self.clip = clip
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.learning_rate = learning_rate
        self.decoder_learning_ratio = decoder_learning_ratio
        self.n_iteration = n_iteration
        self.print_every = print_every
        self.save_every = save_every

        self.checkpoint_iter = checkpoint_iter
        self.checkpoint = None
        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.encoder_optimizer_sd = None
        self.decoder_optimizer_sd = None

        self.searchDecoder = None

        self.training_data = None
        self.processor.process_data('data/pairs.csv')

    def collect_data(self, start_date, end_date, filename):
        self.processor.process_data(filename, start_date, end_date)

    def mask_loss(self, inp, target, mask):
        n_total = mask.sum()
        cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = cross_entropy.masked_select(mask).mean()
        loss = loss.to(self.device)
        return loss, n_total.item()

    def _train(
            self,
            input_variable,
            lengths,
            target_variable,
            mask,
            max_target_len,

    ):

        # Zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Set device options
        input_variable = input_variable.to(self.device)
        target_variable = target_variable.to(self.device)
        mask = mask.to(self.device)
        # Lengths for rnn packing should always be on the cpu
        lengths = lengths.to("cpu")

        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(self.batch_size)]])
        decoder_input = decoder_input.to(self.device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, n_total = self.mask_loss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_total)
                n_totals += n_total
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                decoder_input = decoder_input.to(self.device)
                # Calculate and accumulate loss
                mask_loss, n_total = self.mask_loss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_total)
                n_totals += n_total

        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

        # Adjust ml_model weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return sum(print_losses) / n_totals

    def train_iterators(
            self,
            pairs,
            save_dir
    ):

        # Load batches for each iteration
        training_batches = [self.processor.get_batch_to_train([random.choice(pairs) for _ in range(self.batch_size)])
                            for _ in range(self.n_iteration)]

        # Initializations
        print('Initializing ...')
        start_iteration = 1
        print_loss = 0
        if self.checkpoint:
            start_iteration = self.checkpoint['iteration'] + 1

        # Training loop
        print("Training...")
        for iteration in range(start_iteration, self.n_iteration + 1):
            training_batch = training_batches[iteration - 1]
            # Extract fields from batch
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            # Run a training iteration with batch
            loss = self._train(
                input_variable,
                lengths,
                target_variable,
                mask,
                max_target_len,
            )
            print_loss += loss

            # Print progress
            if iteration % self.print_every == 0:
                print_loss_avg = print_loss / self.print_every
                print('''Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}
                '''.format(iteration, iteration / self.n_iteration * 100, print_loss_avg))

                print_loss = 0

            # Save checkpoint
            if iteration % self.save_every == 0:
                directory = os.path.join(save_dir, self.model_name,
                                         f'{self.encoder_n_layers}-{self.decoder_n_layers}_{self.hidden_size}')
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save({
                    'iteration': iteration,
                    'en': self.encoder.state_dict(),
                    'de': self.decoder.state_dict(),
                    'en_opt': self.encoder_optimizer.state_dict(),
                    'de_opt': self.decoder_optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': self.processor.vocabulary.__dict__,
                    'embedding': self.embedding.state_dict()
                }, os.path.join(directory, f'{iteration}_checkpoint.tar'))

    def evaluate(self, searcher, sentence):
        # Format input sentence as a batch
        # words -> indexes
        indexes_batch = [self.processor.vocabulary.indexes_from_sentence(sentence)]
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(self.device)
        lengths = lengths.to("cpu")
        # Decode sentence with searcher
        tokens, scores = searcher(input_batch, lengths, self.processor.max_sentence_length)
        # indexes -> words
        decoded_words = [self.processor.vocabulary.index2word[token.item()] for token in tokens]
        return decoded_words

    def evaluate_input(self):
        while True:
            try:
                # Get input sentence
                input_sentence = input('> ')
                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit':
                    break
                # Normalize sentence
                input_sentence = normalize_string(input_sentence)
                # Evaluate sentence
                output_words = self.evaluate(self.searchDecoder, input_sentence)
                # Format and print response sentence
                output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                print('Bot:', ' '.join(output_words))

            except KeyError:
                print("Error: Encountered unknown word.")

    def run(self, input_filename, checkpoint_filename=None):
        self.training_data = self.processor.read_data(input_filename)

        self.embedding = nn.Embedding(self.processor.vocabulary.num_words, self.hidden_size)

        if checkpoint_filename:

            self.checkpoint = torch.load(checkpoint_filename, map_location=torch.device('cpu'))
            self.encoder_optimizer_sd = self.checkpoint['en_opt']
            self.decoder_optimizer_sd = self.checkpoint['de_opt']
            embedding_sd = self.checkpoint['embedding']
            self.processor.vocabulary.__dict__ = self.checkpoint['voc_dict']
            self.embedding.load_state_dict(embedding_sd)

        print('Building encoder and decoder ...')
        # Initialize encoder & decoder models
        self.encoder = EncoderRNN(
            self.hidden_size,
            self.embedding,
            self.encoder_n_layers,
            self.dropout
        )
        self.decoder = LuongAttnDecoderRNN(
            self.attn_model,
            self.embedding,
            self.hidden_size,
            self.processor.vocabulary.num_words,
            self.decoder_n_layers,
            self.dropout
        )

        self.searchDecoder = GreedySearchDecoder(
            self.encoder,
            self.decoder,
            self.device
        )

        if checkpoint_filename:
            encoder_sd = self.checkpoint['en']
            decoder_sd = self.checkpoint['de']
            self.encoder.load_state_dict(encoder_sd)
            self.decoder.load_state_dict(decoder_sd)
        # Use appropriate device
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        # Initialize optimizers
        print('Building optimizers ...')
        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(),
            lr=self.learning_rate
        )
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(),
            lr=self.learning_rate * self.decoder_learning_ratio
        )

        if checkpoint_filename:
            self.encoder_optimizer.load_state_dict(self.encoder_optimizer_sd)
            self.decoder_optimizer.load_state_dict(self.decoder_optimizer_sd)

        print('Models built and ready to go!')

    def train(self, save_dir='data/output'):
        # Ensure dropout layers are in train mode
        self.encoder.train()
        self.decoder.train()

        # If you have cuda, configure cuda to call
        for state in self.encoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in self.decoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        # Run training iterations
        print("Starting Training!")
        self.train_iterators(self.training_data, save_dir)