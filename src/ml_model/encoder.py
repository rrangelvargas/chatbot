import torch.nn as nn


class EncoderRNN(nn.Module):
    """
    classe que define o modelo de encoder para a rede neural
    """
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        """
        método de inicialização do encoder
        Args:
            hidden_size: tamanho da camada oculta
            embedding: embedding do modelo para redução de dimensão
            n_layers: número de camadas
            dropout: fator de dropout para reduzir overfitting
        """
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        """
        método para enondar a sequência de entrada e preparar para a rede
        Args:
            input_seq: sequência de indíces das palavras de entrada
            input_lengths: tamanho das palavras de entrada
            hidden: última camada oculta

        Returns: resultado da entrada codificada e o novo estado da camada oculta
        """
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden
