import torch
import torch.nn as nn

from src.utils import SOS_token


class GreedySearchDecoder(nn.Module):
    """
    classe que define o modelo de busca gulosa para determinar a melhor resposta da rede
    """
    def __init__(self, encoder, decoder, device):
        """
        método de inicialização do GreedyDecoder
        Args:
            encoder: modelo de encoder usado na rede
            decoder: modelo de decoder usado na rede
            device: tipo de hardware usado (CPU ou GPU)
        """
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_seq, input_length, max_length):
        """
        método para calcular a melhor resposta para cada palavra da sequência de entrada
        Args:
            input_seq: sequência de entrada
            input_length: tamanho da sequência de entrada
            max_length: tamanho máximo da sequẽncia de entrada

        Returns: uma lista com as palavras que compõem a melhor resposta da rede, e uma lista com suas potuações
        """
        # Forward input through encoder ml_model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self.device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores
