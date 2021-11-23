import re
import unicodedata

# Default word tokens
from datetime import datetime

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Vocabulary:
    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
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

    def indexes_from_sentence(self, sentence):
        return [self.word2index[word] for word in separate_punctuation(sentence).split(' ')] + [EOS_token]

    def trim_rare_words(self, pairs, min_sentence_length):
        # Trim words used under the MIN_COUNT from the voc
        self.trim(min_sentence_length)
        # Filter out pairs with trimmed words
        keep_pairs = []
        for pair in pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True
            # Check input sentence
            for word in input_sentence.split(' '):
                if word not in self.word2index:
                    keep_input = False
                    break
            # Check output sentence
            for word in output_sentence.split(' '):
                if word not in self.word2index:
                    keep_output = False
                    break

            # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
            if keep_input and keep_output:
                keep_pairs.append(pair)

        print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                    len(keep_pairs) / len(pairs)))
        return keep_pairs


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def separate_punctuation(s):
    s = s.replace("?", " ?")
    s = s.replace(".", " .")
    s = s.replace("!", " !")
    return s


def format_answer(answer):
    s = ''
    for i in range(len(answer)):
        if answer[i] == "EOS":
            break
        s = f'{s} {answer[i]}'
    s = s.replace(" t ", "'t ")
    s = s.replace(" m ", "'m ")
    s = s.replace(" re ", "'re ")
    s = s.replace(" s ", "'s ")
    s = s.replace(" ?", "?")
    s = s.replace(" .", ".")
    s = s.replace(" !", "!")

    return s


def datetime_to_timestamp(datetime_obj: datetime):
    return datetime_obj.strftime("%Y-%m-%d %H:%M:%S.%f")
