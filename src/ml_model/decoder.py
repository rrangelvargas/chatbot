import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    """
    classe definindo o modelo de atenção de Luong
    """
    def __init__(self, method, hidden_size):
        """
        método de inicialização do modelo de atenção
        Args:
            method: qual tipo de método será utilizado (dot, general ou concat)
            hidden_size: tamanho da camada escondida
        """
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def general_score(self, hidden, encoder_output):
        """
        método para calcular os pesos de atenção de acordo com o metodo 'general'
        Args:
            hidden: tamanho da camada oculta
            encoder_output: sequência de entrada

        Returns: resultado dos pesos de atenção
        """
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        """
        método para calcular os pesos de atenção de acordo com o metodo 'concat'
        Args:
            hidden: última camada oculta
            encoder_output: sequência de entrada

        Returns: resultado dos pesos de atenção
        """
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    # Luong attention layer
    @staticmethod
    def dot_score(hidden, encoder_output):
        """
        método para calcular os pesos de atenção de acordo com o metodo 'dot'
        Args:
            hidden: última camada oculta
            encoder_output: sequência de entrada

        Returns: resultado dos pesos de atenção
        """
        return torch.sum(hidden * encoder_output, dim=2)

    def forward(self, hidden, encoder_outputs):
        """
        método para cálculo das probabilidades das entidades
        Args:
            hidden: última camada oculta
            encoder_outputs: sequência de entrada

        Returns: matriz com as probabilidades das entidades normalizadas
        """
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        else:
            raise Exception('Method not implemented!')

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    """
    classe que define o decoder de Luong baseado no método de modelos de atenção
    """
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        """
        Args:
            attn_model: tipo do modelo de atenção
            embedding: embeding do modelo para redução de dimensão
            hidden_size: tamanho da caamda oculta
            output_size: tamanho da saída
            n_layers: número de camadas
            dropout: fator de dropout para reduzir overfitting
        """
        super(LuongAttnDecoderRNN, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        """
        método para calcular o resultado do decoder para uma palavra
        Args:
            input_step: palavra atual da entrada
            last_hidden: última camada oculta
            encoder_outputs: sequência de entrada

        Returns: resultado com as pontuações das possíveis respostas e o último estado da camada oculta
        """
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
