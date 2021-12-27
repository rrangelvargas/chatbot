import csv
import itertools
import os
import random

import torch

from src import PostgresClient
import codecs
from src.utils import Vocabulary, normalize_string, PAD_token


def zero_padding(sentences, fill_value=PAD_token):
    return list(itertools.zip_longest(*sentences, fillvalue=fill_value))


def binary_matrix(sentences, value=PAD_token):
    m = []
    for i, seq in enumerate(sentences):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


class DataProcessor:
    """
    classe que define o módulo utilizado para fazer o processamento e a formatação dos dados
    """
    def __init__(
            self,
            encoding='iso-8859-1',
            batch_size=5,
    ):
        """
        método de inicialização do DataProcessor
        Args:
            encoding: tipo de enconding dos dados
            batch_size: tamanho dos lotes
        """
        self.encoding = encoding
        self.batch_size = batch_size
        self.delimiter = str(codecs.decode(b'\t', "unicode_escape"))
        self.max_sentence_length = 30
        self.min_sentence_length = 1
        self.vocabulary = Vocabulary()
        self.db_client = PostgresClient

    # Extracts pairs of sentences from conversations
    def extract_sentence_pairs(self, start_date=None, end_date=None):
        """
        método para extrair os pares do banco de dados
        Args:
            start_date: data inicial para extração
            end_date: data final para extração

        Returns: lista com os pares extraídos
        """
        query = '''
            SELECT cm.conversation_id, cm.message_id, cm.message_index, m.text
            FROM conversation_message AS cm
            LEFT JOIN message AS m ON cm.message_id=m.id
            LEFT JOIN conversation c on cm.conversation_id = c.id
        '''

        if start_date:
            query += f" WHERE c.started_at>={start_date}"
        if end_date:
            query += f" AND c.ended_at <={end_date}"

        query += '''
            GROUP BY cm.conversation_id, cm.message_id, cm.message_index, m.text
            ORDER BY cm.conversation_id, cm.message_index
        '''

        qa_pairs = []
        result = self.db_client.execute_query(query)
        print("\nExtracting pairs...")

        for i in range(len(result)-1):
            if result[i][0] == result[i+1][0]:
                input_line = result[i][3].strip()
                target_line = result[i+1][3].strip()
                qa_pairs.append([input_line, target_line])

        return qa_pairs

    def format_text(self, output_file, start_date=None, end_date=None):
        """
        método para escrever os dados coletados do banco de dados em um arquivo a ser usado pela rede
        Args:
            output_file: caminho do arquivo de saída
            start_date: data inicial para extração
            end_date: data final para extração
        """

        print("\nWriting newly formatted file...")
        if start_date:
            with open(output_file, 'a', encoding='utf-8') as output:
                writer = csv.writer(output, delimiter=self.delimiter, lineterminator='\n')
                for pair in self.extract_sentence_pairs(start_date, end_date):
                    writer.writerow(pair)
        else:
            with open(output_file, 'w', encoding='utf-8') as output:
                writer = csv.writer(output, delimiter=self.delimiter, lineterminator='\n')
                for pair in self.extract_sentence_pairs():
                    writer.writerow(pair)

        print("\nDone!")

    def get_pairs_from_file(self, datafile):
        """
        método para ler os pares de sentenças de um arquivo e retornar
        Args:
            datafile:

        Returns:

        """
        print("Reading lines...")
        # Read the file and split into lines
        lines = open(datafile, encoding='utf-8'). \
            read().strip().split('\n')
        # Split every line into pairs and normalize
        pairs = [[normalize_string(s) for s in line.split('\t')] for line in lines]
        return self.filter_pairs(pairs)

    def filter_pair(self, p):
        """
        método para filtrar um dado part de sentenças de acordo com o tamanho máximo
        Args:
            p: par de sentenças a ser filtrado

        Returns: true se ambas as sentenças são menores que o tamanho máximo, false caso contrário
        """
        return len(p[0].split(' ')) < self.max_sentence_length and len(p[1].split(' ')) < self.max_sentence_length

    def filter_pairs(self, pairs):
        """
        método para filtrar uma lista de pares de acordo com o tamanho máximo da sentença
        Args:
            pairs: lista de pares

        Returns: lista de pares filtradas
        """
        return [pair for pair in pairs if self.filter_pair(pair)]

    def load_prepare_data(self, datafile):
        """
        método para carregar os dados do arquivo no vocabulário da rede
        Args:
            datafile: arquivo contendo os pares de sentenças

        Returns: lista com os pares de sentenças
        """
        print("Start preparing training data ...")
        pairs = self.get_pairs_from_file(datafile)
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))
        print("Counting words...")
        for pair in pairs:
            self.vocabulary.add_sentence(pair[0])
            self.vocabulary.add_sentence(pair[1])
        print("Counted words:", self.vocabulary.num_words)
        return pairs

    def process_data(
            self,
            output_file,
            start_date=None,
            end_date=None,
            retrain=False
    ):
        """
        método usado para coletar os dados caso ainda não existam
        Args:
            output_file: caminho do arquivo de saída
            start_date: data inicial para extração
            end_date: data final para extração
            retrain: flag para determinar se será realizado um retreinamento ou não
        """

        if not os.path.exists(output_file) or retrain:
            self.format_text(output_file, start_date, end_date)

    def read_data(self, output_file):
        """
        método para ler e filtrar os pares de sentenças
        Args:
            output_file: caminho para o arquivo contendo os pares

        Returns: lista com os pares filtrados
        """

        pairs = self.load_prepare_data(output_file)
        trimmed_pairs = self.vocabulary.trim_rare_words(pairs, self.min_sentence_length)

        return trimmed_pairs

    def input_var(self, sentences):
        """
        método para obter os tensores de entrada e seus tamanhos para cada sentença de entrada
        Args:
            sentences: lista com as sentenças de entrada

        Returns: uma lista com os tensores de entrada e outra com seus tamanhos

        """
        indexes_batch = [self.vocabulary.indexes_from_sentence(sentence) for sentence in sentences]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        pad_list = zero_padding(indexes_batch)
        pad_var = torch.LongTensor(pad_list)
        return pad_var, lengths

    def output_var(self, sentences):
        """
        método para obter os tensores de saída, o tensor de perda correspondente e
        seus tamanhos para cada sentença de entrada
        Args:
            sentences: lista com as sentenças de entrada

        Returns: uma lista com os tensores de saída, outra com os tensores de perda
        correspondentes e outra com seus tamanhos

        """
        indexes_batch = [self.vocabulary.indexes_from_sentence(sentence) for sentence in sentences]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        pad_list = zero_padding(indexes_batch)
        mask = binary_matrix(pad_list)
        mask = torch.BoolTensor(mask)
        pad_var = torch.LongTensor(pad_list)
        return pad_var, mask, max_target_len

    def get_batch_to_train(self, pair_batch):
        """
        método para obter us dados de um lote de treinamento
        Args:
            pair_batch: pares do lote a ser usado no treinamento

        Returns: dados de entrada e saída e seus tamanhos, e os tensores de perda

        """
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = self.input_var(input_batch)
        output, mask, max_target_len = self.output_var(output_batch)
        return inp, lengths, output, mask, max_target_len

    def get_batches(self, pairs):
        """
        método para coletar o lote de acordo com o tamanho definido
        Args:
            pairs: lista dos pares de sentenças

        Returns: os dados de um lote do tamanho definido
        """
        return self.get_batch_to_train([random.choice(pairs) for _ in range(self.batch_size)])
