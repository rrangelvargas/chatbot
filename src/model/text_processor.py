import csv
import os
from src.api import DBClient
import codecs
from src.utils import Vocabulary, normalize_string, PAD_token, SOS_token, EOS_token
from src.config import POSTGRES_DB, POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD

class TextProcessor:
    def __init__(self, encoding='iso-8859-1'):
        self.encoding = encoding
        self.delimiter = str(codecs.decode(b'\t', "unicode_escape"))
        self.max_sentence_length = 10
        self.min_sentence_length = 3
        self.vocabulary = Vocabulary()

    # Extracts pairs of sentences from conversations
    @staticmethod
    def extract_sentence_pairs(start_date=None, end_date=None):
        c = DBClient(POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST)
        query = '''
            SELECT cl.conversation_id, cl.line_id, cl.line_index, l.text
            FROM conversation_line AS cl
            LEFT JOIN line AS l ON cl.line_id=l.id
        '''

        if start_date:
            query += f" WHERE l.start_date>={start_date}"
        if end_date:
            query += f" AND l.end_date <={end_date}"

        query += " GROUP BY cl.conversation_id, cl.line_id, cl.line_index, l.text ORDER BY cl.conversation_id, cl.line_index"

        qa_pairs = []
        result = c.execute_query(query)
        print("\nExtracting pairs...")
        i = 0
        while i < len(result):  # We ignore the last line (no answer for it)
            j = 0
            while i+j < len(result)-1 and result[i+j+1][2] != 0:
                input_line = result[i+j][3].strip()
                target_line = result[i+j + 1][3].strip()
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
            with open(output_file, 'w', encoding='utf-8') as output:
                writer = csv.writer(output, delimiter=self.delimiter, lineterminator='\n')
                for pair in self.extract_sentence_pairs():
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
        return pairs

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

    def trim_rare_words(self, pairs):
        # Trim words used under the MIN_COUNT from the voc
        self.vocabulary.trim(self.min_sentence_length)
        # Filter out pairs with trimmed words
        keep_pairs = []
        for pair in pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True
            # Check input sentence
            for word in input_sentence.split(' '):
                if word not in self.vocabulary.word2index:
                    keep_input = False
                    break
            # Check output sentence
            for word in output_sentence.split(' '):
                if word not in self.vocabulary.word2index:
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
            output_path,
            output_file_name,
            corpus_name,
            start_date=None,
            end_date=None
    ):
        output_file = os.path.join(output_path, output_file_name)
        if not os.path.exists(output_file) or start_date:
            self.format_text(output_file, start_date, end_date)

        pairs = self.load_prepare_data(corpus_name, output_file)
        trimmed_pairs = self.trim_rare_words(pairs)

        return trimmed_pairs



# def create_database():
#     C = DBClient(POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST)
# #     # C.execute_query('ALTER TABLE line ALTER COLUMN id TYPE TEXT;')
# #     C.execute_query('DROP TABLE conversation CASCADE;')
# #     C.execute_query('DROP TABLE conversation_line;')
# #     C.execute_query('DROP TABLE line;')
#     C.execute_query('CREATE TABLE conversation(ID SERIAL PRIMARY KEY NOT NULL);')
#     C.execute_query('''
#     CREATE TABLE conversation_line(
#     ID SERIAL PRIMARY KEY,
#     CONVERSATION_ID INT NOT NULL,
#     LINE_ID TEXT NOT NULL,
#     LINE_INDEX INT NOT NULL,
#     CONSTRAINT fk_conversation FOREIGN KEY(CONVERSATION_ID) REFERENCES conversation(ID),
#     CONSTRAINT fk_line FOREIGN KEY(LINE_ID) REFERENCES line(ID)
#     );
#     ''')
# #     # t = "(10, 'teste'), (11, 'teste2')"
# #     C.execute_query('DELETE FROM conversation WHERE id=2')
# #     # C.execute_query('INSERT INTO line(text) VALUES(%s);', ("teste3",))


# # Splits each line of the file into a dictionary of fields
# def load_lines(file_name):
#     c = DBClient(POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST)
#     query_str = ""
#     with open(file_name, 'r', encoding='iso-8859-1') as f:
#         for line in f:
#             values = line.split(" +++$+++ ")
#             query_str += f"{values[0], normalize_string(values[4])}, "
#     query_str = f"INSERT INTO line(id, text) VALUES {query_str[:-2]};"
#     c.execute_query(query_str)


# # Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
# def load_conversations(file_name):
#     c = DBClient(POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST)
#     with open(file_name, 'r', encoding='iso-8859-1') as f:
#         i = 1
#         conv_query = ""
#         conv_lin_query = ""
#         for line in f:
#             conv_query += f"({i}), "
#             values = line.split(" +++$+++ ")
#
#             utterance_id_pattern = re.compile('L[0-9]+')
#             line_ids = utterance_id_pattern.findall(values[3])
#             for j in range(len(line_ids)):
#                 conv_lin_query += f"{i, line_ids[j], j}, "
#             i += 1
#     c.execute_query(f"INSERT INTO conversation(id) VALUES {conv_query[:-2]};")
#     c.execute_query(f"INSERT INTO conversation_line(conversation_id, line_id, line_index) VALUES {conv_lin_query[:-2]};")

# collect_conversation()

# load_lines('data/movie_lines.txt')
# load_conversations('data/movie_conversations.txt')
# create_database()
P = TextProcessor()
P.process_data('data', 'output.csv', 'name')