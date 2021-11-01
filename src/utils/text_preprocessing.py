import csv
import os
import re
import codecs
from text_utils import normalize_string


class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.add_word(word)


class TextProcessor:
    def __init__(self):
        self.encoding = 'iso-8859-1'
        self.delimiter = str(codecs.decode(b'\t', "unicode_escape"))
        self.lines_fields = ["lineID", "characterID", "movieID", "character", "text"]
        self.conversations_fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
        self.max_sentence_length = 10
        self.min_sentence_length = 3

    # Splits each line of the file into a dictionary of fields
    def load_lines(self, file_name, fields):
        lines = {}
        with open(file_name, 'r', encoding=self.encoding) as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                line_obj = {}
                for i, field in enumerate(fields):
                    line_obj[field] = values[i]
                lines[line_obj['lineID']] = line_obj
        return lines

    # Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
    def load_conversations(self, file_name, lines, fields):
        conversations = []
        with open(file_name, 'r', encoding=self.encoding) as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                conv_obj = {}
                for i, field in enumerate(fields):
                    conv_obj[field] = values[i]
                # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
                utterance_id_pattern = re.compile('L[0-9]+')
                line_ids = utterance_id_pattern.findall(conv_obj["utteranceIDs"])
                # Reassemble lines
                conv_obj["lines"] = []
                for line_id in line_ids:
                    conv_obj["lines"].append(lines[line_id])
                conversations.append(conv_obj)
        return conversations

    # Extracts pairs of sentences from conversations
    @staticmethod
    def extract_sentence_pairs(conversations):
        qa_pairs = []
        for conversation in conversations:
            # Iterate over all the lines of the conversation
            for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
                input_line = conversation["lines"][i]["text"].strip()
                target_line = conversation["lines"][i + 1]["text"].strip()
                # Filter wrong samples (if one of the lists is empty)
                if input_line and target_line:
                    qa_pairs.append([input_line, target_line])
        return qa_pairs

    def format_text(self, lines_input_file, conversations_input_file, output_file):
        # Load lines and process conversations
        print("\nProcessing corpus...")
        lines = self.load_lines(lines_input_file, self.lines_fields)
        print("\nLoading conversations...")
        conversations = self.load_conversations(conversations_input_file,
                                                lines, self.conversations_fields)

        # Write new csv file
        print("\nWriting newly formatted file...")
        with open(output_file, 'w', encoding='utf-8') as output:
            writer = csv.writer(output, delimiter=self.delimiter, lineterminator='\n')
            for pair in self.extract_sentence_pairs(conversations):
                writer.writerow(pair)

        print("\nDone!")

    # Read query/response pairs and return a voc object
    @staticmethod
    def read_vocabulary(datafile, corpus_name):
        print("Reading lines...")
        # Read the file and split into lines
        lines = open(datafile, encoding='utf-8'). \
            read().strip().split('\n')
        # Split every line into pairs and normalize
        pairs = [[normalize_string(s) for s in line.split('\t')] for line in lines]
        voc = Vocabulary(corpus_name)
        return voc, pairs

    # Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
    def filter_pair(self, p):
        # Input sequences need to preserve the last word for EOS token
        return len(p[0].split(' ')) < self.max_sentence_length and len(p[1].split(' ')) < self.max_sentence_length

    # Filter pairs using filterPair condition
    def filter_pairs(self, pairs):
        return [pair for pair in pairs if self.filter_pair(pair)]

    # Using the functions defined above, return a populated voc object and pairs list
    def load_prepare_data(self, corpus_name, datafile):
        print("Start preparing training data ...")
        voc, pairs = self.read_vocabulary(datafile, corpus_name)
        print("Read {!s} sentence pairs".format(len(pairs)))
        pairs = self.filter_pairs(pairs)
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))
        print("Counting words...")
        for pair in pairs:
            voc.add_sentence(pair[0])
            voc.add_sentence(pair[1])
        print("Counted words:", voc.num_words)
        return voc, pairs

    def trim_rare_words(self, voc, pairs):
        # Trim words used under the MIN_COUNT from the voc
        voc.trim(self.min_sentence_length)
        # Filter out pairs with trimmed words
        keep_pairs = []
        for pair in pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True
            # Check input sentence
            for word in input_sentence.split(' '):
                if word not in voc.word2index:
                    keep_input = False
                    break
            # Check output sentence
            for word in output_sentence.split(' '):
                if word not in voc.word2index:
                    keep_output = False
                    break

            # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
            if keep_input and keep_output:
                keep_pairs.append(pair)

        print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                    len(keep_pairs) / len(pairs)))
        return keep_pairs

    def process_data(
            self,
            input_path,
            output_path,
            lines_input_file_name,
            conversations_input_file_name,
            output_file_name,
            corpus_name
    ):
        output_file = os.path.join(output_path, output_file_name)
        if not os.path.exists(output_file):
            lines_input_file = os.path.join(input_path, lines_input_file_name)
            conversations_input_file = os.path.join(input_path, conversations_input_file_name)
            self.format_text(lines_input_file, conversations_input_file, output_file)

        voc, pairs = self.load_prepare_data(output_file, corpus_name)
        trimmed_pairs = self.trim_rare_words(voc, pairs)

        return trimmed_pairs

