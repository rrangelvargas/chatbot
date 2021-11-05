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
    def __init__(
            self,
            encoding='iso-8859-1',
            batch_size=5,
    ):
        self.encoding = encoding
        self.batch_size = batch_size
        self.delimiter = str(codecs.decode(b'\t', "unicode_escape"))
        self.max_sentence_length = 10
        self.min_sentence_length = 3
        self.vocabulary = Vocabulary()
        self.db_client = PostgresClient

    # Extracts pairs of sentences from conversations
    def extract_sentence_pairs(self, start_date=None, end_date=None):
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
        i = 0
        while i < 5:
            j = 0
            while i + j < len(result) - 1 and result[i + j + 1][2] != 0:
                input_line = result[i + j][3].strip()
                target_line = result[i + j + 1][3].strip()
                # Filter wrong samples (if one of the lists is empty)
                if input_line and target_line:
                    qa_pairs.append([input_line, target_line])
                j += 1
            i += j + 1

        return qa_pairs

    def format_text(self, output_file, start_date=None, end_date=None):
        # Write new csv file
        print("\nWriting newly formatted file...")
        if start_date:
            with open(output_file, 'a', encoding='utf-8') as output:
                writer = csv.writer(output, delimiter=self.delimiter, lineterminator='\n')
                for pair in self.extract_sentence_pairs(start_date, end_date):
                    writer.writerow(pair)
        else:
            print(os.path)
            with open(output_file, 'w', encoding='utf-8') as output:
                writer = csv.writer(output, delimiter=self.delimiter, lineterminator='\n')
                for pair in self.extract_sentence_pairs():
                    writer.writerow(pair)

        print("\nDone!")

    # Read query/response pairs and return a voc object
    def get_pairs_from_file(self, datafile):
        print("Reading lines...")
        # Read the file and split into lines
        lines = open(datafile, encoding='utf-8'). \
            read().strip().split('\n')
        # Split every line into pairs and normalize
        pairs = [[normalize_string(s) for s in line.split('\t')] for line in lines]
        return self.filter_pairs(pairs)

    # Returns True if both sentences in a pair 'p' are under the MAX_LENGTH threshold
    def filter_pair(self, p):
        # Input sequences need to preserve the last word for EOS token
        return len(p[0].split(' ')) < self.max_sentence_length and len(p[1].split(' ')) < self.max_sentence_length

    # Filter pairs using filterPair condition
    def filter_pairs(self, pairs):
        return [pair for pair in pairs if self.filter_pair(pair)]

    # Using the functions defined above, return a populated voc object and pairs list
    def load_prepare_data(self, datafile):
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
            end_date=None
    ):
        if not os.path.exists(output_file) or start_date:
            self.format_text(output_file, start_date, end_date)

    def read_data(self, output_file):
        pairs = self.load_prepare_data(output_file)
        trimmed_pairs = self.vocabulary.trim_rare_words(pairs, self.min_sentence_length)

        return trimmed_pairs

    # Returns padded input sequence tensor and lengths
    def input_var(self, sentences):
        indexes_batch = [self.vocabulary.indexes_from_sentence(sentence) for sentence in sentences]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        pad_list = zero_padding(indexes_batch)
        pad_var = torch.LongTensor(pad_list)
        return pad_var, lengths

    # Returns padded target sequence tensor, padding mask, and max target length
    def output_var(self, sentences):
        indexes_batch = [self.vocabulary.indexes_from_sentence(sentence) for sentence in sentences]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        pad_list = zero_padding(indexes_batch)
        mask = binary_matrix(pad_list)
        mask = torch.BoolTensor(mask)
        pad_var = torch.LongTensor(pad_list)
        return pad_var, mask, max_target_len

    # Returns all items for a given batch of pairs
    def get_batch_to_train(self, pair_batch):
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = self.input_var(input_batch)
        output, mask, max_target_len = self.output_var(output_batch)
        return inp, lengths, output, mask, max_target_len

    def get_batches(self, pairs):
        return self.get_batch_to_train([random.choice(pairs) for _ in range(self.batch_size)])