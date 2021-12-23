import re
import unicodedata
from datetime import datetime

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Vocabulary:
    """
    classe que define o vocabulário da rede neural
    """
    def __init__(self):
        """
        inicialização do vocabulário
        """
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def add_sentence(self, sentence):
        """
        método para adiconar uma frase ao vocabulário
        Args:
            sentence: frase a ser adicionada
        """

        # para cada palavra da frase, é chamado o método de adiconar palavra ao vocabulário
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """
        método para adiconar palavra ao vocabulário
        Args:
            word: palavra a ser adicionada
        """

        # adiciona a palavra caso não exista
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # caso já exista, apenas aumenta a contagem da palavra
            self.word2count[word] += 1

    def trim(self, min_count):
        """
        método para remover do vocabulário as palavras que aparecem menos vezes que a contagem mínima
        Args:
            min_count: quantidade mínima de vezes que a palavra deve aparecer no dataset de treinamento
        """
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

        # Reinitializando os dicionários
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.add_word(word)

    def indexes_from_sentence(self, sentence):
        """
        método para retornar o índice de cada palavra em uma frase
        Args:
            sentence: frase de entrada

        Returns: lista com o índice de cada palavra da frase

        """
        return [self.word2index[word] for word in separate_punctuation(sentence).split(' ')] + [EOS_token]

    def trim_rare_words(self, pairs, min_sentence_length):
        """
        método para remover palavras raras e os pares de mensagens nos quais essas palavras aparecem
        Args:
            pairs: lista de pares de mensagens
            min_sentence_length: quantidade mínima de vezes que a palavra deve aparecer no dataset de treinamento

        Returns: lista dos pares restantes
        """

        # filtrando o vocabulário
        self.trim(min_sentence_length)

        keep_pairs = []
        for pair in pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True
            # checando a mensagem inicial
            for word in input_sentence.split(' '):
                if word not in self.word2index:
                    keep_input = False
                    break
            # checando a resposta
            for word in output_sentence.split(' '):
                if word not in self.word2index:
                    keep_output = False
                    break

            # apenas manter o par se toas as palavras de ambas as mensagens estão no vocabulário
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


def normalize_string(s):
    """
    método para normalização de string
    Args:
        s: string original

    Returns: string com todos os caracteres minúsculos e apenas letras e números
    """
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def separate_punctuation(s):
    """
    método para separar a pontuação das palavras em uma string
    Args:
        s: string original

    Returns: string com as potuações separadas

    """
    s = s.replace("?", " ?")
    s = s.replace(".", " .")
    s = s.replace("!", " !")
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def format_answer(answer):
    '''
    método para formatar a resposta antes de enviar ao usuário
    Args:
        answer: lista com as palavras da resposta em ordem

    Returns: resposta com correções de formatação
    '''
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
    '''
    método para transformar um objeto datetime em um formato compatível com o banco de dados
    Args:
        datetime_obj: objeto datetime

    Returns: data e hora no formato compatível com o banco de dados
    '''
    return datetime_obj.strftime("%Y-%m-%d %H:%M:%S.%f")
