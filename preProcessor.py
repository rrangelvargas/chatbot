import tensorflow as tf
import re
import tensorflow_datasets as tfds

path_to_movie_lines = 'movie_lines.txt'
path_to_movie_conversations = 'movie_conversations.txt'

class PreProcessor:
    def __init__(self):
        # Maximum number of samples to preprocess
        self.MAX_SAMPLES = 50000

        # Maximum sentence length
        self.MAX_LENGTH = 40

        self.VOCAB_SIZE = None

        self.START_TOKEN = None

        self.END_TOKEN = None

        self.tokenizer = None



    def preprocess_sentence(self, sentence):
        sentence = sentence.lower().strip()
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
        sentence = sentence.strip()
        # adding a start and an end token to the sentence
        return sentence

    def load_conversations(self):
        # dictionary of line id to text
        id2line = {}
        with open(path_to_movie_lines, errors='ignore') as file:
            lines = file.readlines()
        for line in lines:
            parts = line.replace('\n', '').split(' +++$+++ ')
            id2line[parts[0]] = parts[4]

        inputs, outputs = [], []
        with open(path_to_movie_conversations, 'r') as file:
            lines = file.readlines()
        for line in lines:
            parts = line.replace('\n', '').split(' +++$+++ ')
            # get conversation in a list of line ID
            conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
            for i in range(len(conversation) - 1):
                inputs.append(self.preprocess_sentence(id2line[conversation[i]]))
                outputs.append(self.preprocess_sentence(id2line[conversation[i + 1]]))
                if len(inputs) >= self.MAX_SAMPLES:
                    return inputs, outputs
        return inputs, outputs

    # Tokenize, filter and pad sentences
    def tokenize_and_filter(self, inputs, outputs, tokenizer):

        # Define start and end token to indicate the start and end of a sentence
        self.START_TOKEN, self.END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

        tokenized_inputs, tokenized_outputs = [], []
        
        for (sentence1, sentence2) in zip(inputs, outputs):
            # tokenize sentence
            sentence1 = self.START_TOKEN + tokenizer.encode(sentence1) + self.END_TOKEN
            sentence2 = self.START_TOKEN + tokenizer.encode(sentence2) + self.END_TOKEN
            # check tokenized sentence max length
            if len(sentence1) <= self.MAX_LENGTH and len(sentence2) <= self.MAX_LENGTH:
                tokenized_inputs.append(sentence1)
                tokenized_outputs.append(sentence2)
        
        # pad tokenized sentences
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=self.MAX_LENGTH, padding='post')
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=self.MAX_LENGTH, padding='post')
        
        return tokenized_inputs, tokenized_outputs

    def get_processed_data(self):
        questions, answers = self.load_conversations()

        print('Sample question: {}'.format(questions[20]))
        print('Sample answer: {}'.format(answers[20]))

        # Build tokenizer using tfds for both questions and answers
        self.tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            questions + answers, target_vocab_size=2**13)

        # Vocabulary size plus start and end token
        self.VOCAB_SIZE = self.tokenizer.vocab_size + 2

        print('Tokenized sample question: {}'.format(self.tokenizer.encode(questions[20])))

        questions, answers = self.tokenize_and_filter(questions, answers, self.tokenizer)

        print('Vocab size: {}'.format(self.VOCAB_SIZE))
        print('Number of samples: {}'.format(len(questions)))

        return questions, answers