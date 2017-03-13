import cPickle as pickle
import itertools
import json

import numpy as np
from nltk import tokenize


class Tokenizer(object):
    """Basic Tokenizer class to convert text to a sequence of numbers or one-hot encodings of the sequences.
    The API of this class is based on the Tokenizer Keras implementation: https://keras.io/preprocessing/text/
    The class uses the NLTK package to tokenize the text.
    """

    UNKNOWN_TAG = '<UNK>'
    PADDING_START = 'start'
    PADDING_END = 'end'

    def __init__(self, max_words=None, lower=True, maxlen=None):
        """Tokenizer constructor.
        Args:
            max_words (int): Maximum number of words to create the dictionary mapping words to numbers. If not provided
                all the words seen in training will be used, otherwise the top (num_words - 1) most frequent words will
                be used. The -1 is due to the fact that an unknown tag must be added for words that are not in the dict
            lower (bool): If True, the word dictionary created and the text processed will be transformed to lowercase
            maxlen (int): Maximum sequence length for texts. If None the maximum length of the training texts will be
                used
        """
        self.max_words = max_words
        self.lower = lower
        self.maxlen = maxlen
        # Initialization of attributes. All of them will be empty until fit_on_texts is called
        self.num_words = 0  # Number of words in our dictionary
        self._word_count = {}  # Map token_word => num_occurrences
        self._word_idx = {}  # Map token_word => index. Index is 0-based

    def fit_on_texts(self, texts):
        """Train the tokenizer from a list of texts (strings) in order to create the index dictionary.
        Args:
            texts (list): List of strings from which the tokenizer will build the dictionary map
        """
        # Get a list of token words from all the given texts
        if self.lower:
            token_array = [tokenize.word_tokenize(text.lower()) for text in texts]
        else:
            token_array = [tokenize.word_tokenize(text) for text in texts]
        # Get the maximum length of all the text sequences
        if self.maxlen is None:
            self.maxlen = max(map(len, token_array))
        token_list = list(itertools.chain.from_iterable(token_array))
        del token_array  # Help GC for large texts
        # Create the word_count dictionary mapping word token to number of ocurrences
        self.__create_word_count(token_list)
        # Create the word_idx dictionary mapping word token to its integer value (index)
        self.__create_word_idx()

    def word_to_idx(self, word):
        """Get the number representation (index) of the given word.
        If the word is not known, the index of the unknown tag (<UNK>) will be returned.
        Args:
            word (str): The word to get the number representation of
        """
        # If the word appears in our dictionary, return the respective index. Otherwise return the index of <UNK>
        if word in self._word_idx:
            return self._word_idx[word]
        else:
            return self._word_idx[self.UNKNOWN_TAG]

    def text_to_sequence(self, text, padding=None):
        """Transform a text into a sequence (list) of numbers, each number representing a word.
        It uses the Tokenizer.word_to_idx method. It can pad the sequence with zeros or truncat it if needed if you
        indicate that you want to use the maxlen attribute.
        Args:
            :param int text: The text in a string
            :param padding: If the sequence should be padded (or truncated) to have exact maxlen items. If None, the
                sequence will have the text word length. Other options are 'start' or 'end', which indicate were the
                transformation should be applied. For 'start', the padding/truncating will be at the beginning and for
                'end' it will be at the end of the sequence.
        """
        tokens = tokenize.word_tokenize(text)
        tokens = [self.word_to_idx(word) for word in tokens]
        if padding is not None:
            tokens = self.pad_sequence(tokens, self.maxlen, padding)
        return tokens

    def texts_to_sequences(self, texts, padding=None):
        """Shortcut for the text_to_sequence method when applied to a list of texts"""
        return [self.text_to_sequence(text, padding) for text in texts]

    def one_hot(self, word_idx):
        """Encodes the given word index (which in turns represents a word) to a one-hot vector.
        The returned vector is a numpy array of shape (num_words,) and boolean types.
        Args:
            word_idx (int): The numerical representation of a word. This index must be one given by one of the tokenizer
                methods.
        """
        # Check if the given index is from the word_idx dictionary
        if word_idx > self.num_words - 1:
            raise ValueError(
                'Tokenizer cannot encode the given index. The index {} is not recognized as being one of the '
                'tokenizer words'.format(word_idx))
        # Initialize tuple with all zeros and set 1 to the given word index position
        one_hot = [0] * self.num_words
        one_hot[word_idx] = 1
        return one_hot

    def word_to_one_hot(self, word):
        """Shortcut for encoding a word into a one-hot vector.
        Calls word_to_idx followed by one_hot
        """
        word_idx = self.word_to_idx(word)
        return self.one_hot(word_idx)

    def text_to_one_hot_seq(self, text, padding=None):
        """Encodes each word in the text as one-hot vector"""
        idx_seq = self.text_to_sequence(text, padding)
        return [self.one_hot(word_idx) for word_idx in idx_seq]

    def save(self, path):
        """Saves the tokenizer in the given location using pickle"""
        with open(path, 'w') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Loads a saved tokenizer pickled"""
        with open(path, 'r') as f:
            return pickle.load(f)

    @staticmethod
    def tokenize(text):
        """Tokenizes the given text.
        It uses the nltk.tokenize.word_tokenize method, it is an alias of this method.
        """
        return tokenize.word_tokenize(text)

    @staticmethod
    def pad_sequence(sequence, maxlen, padding='start'):
        # If the sequence already has exactly maxlen items, return it
        if len(sequence) == maxlen:
            return sequence

        if padding != Tokenizer.PADDING_START and padding != Tokenizer.PADDING_END:
            raise ValueError(
                'The padding method {} is not correct. It can be either \'start\' or \'end\''.format(padding))

        len_dif = abs(len(sequence) - maxlen)
        if maxlen > len(sequence):
            return ([0] * len_dif + sequence) if (padding == Tokenizer.PADDING_START) else (sequence + [0] * len_dif)
        else:
            return sequence[len_dif:] if (padding == Tokenizer.PADDING_START) else sequence[:-len_dif]

    def __create_word_count(self, token_list):
        """Creates a dictionary mapping a word token to the number of ocurrences of that token in the given list.
        Args:
            token_list (list): List of word tokens
        """
        for token in token_list:
            if token in self._word_count:
                self._word_count[token] += 1
            else:
                self._word_count[token] = 1

    def __create_word_idx(self):
        """Creates a dictionary mapping word tokens to their numerical representation.
        It uses the word_count attribute as its source and creates the dictionary with at most num_words if specified,
        otherwise all the word tokens are present. Adds the <UNK> tag for words that are not represented in the
        dictionary.
        """
        # Always substract 1, as we need to add the <UNK> tag, identifying words that are not in the dictionary
        if self.max_words is None or len(self._word_count) <= (self.max_words - 1):
            # The whole dictionary of words seen in "training" can be used
            self._word_idx = self.__create_word_idx_from_tuples(self._word_count.iteritems())
        else:
            # Create the dictionary with the top num_words most used
            # key=lambda x: x[1] --> use the second element in the item tuple for comparison, which is the number of
            # ocurrences
            sorted_tokens = sorted(self._word_count.iteritems(), key=lambda item: item[1], reverse=True)
            sorted_tokens = sorted_tokens[:(self.max_words - 1)]
            self._word_idx = self.__create_word_idx_from_tuples(sorted_tokens)

        # Add the UNKNOWN_TAG to the dictionary
        self._word_idx[self.UNKNOWN_TAG] = len(self._word_idx) + 1  # Assign the last index to <UNK>
        # Set the actual length of our dictionary, which can be different from max_words
        self.num_words = len(self._word_idx)

    @staticmethod
    def __create_word_idx_from_tuples(tuples):
        # Add 1 to the index of the enumerate as we want the dictionary to be 1-based (starting from 1) to be able to
        # pad the sequences with 0 and consider this value as a mask
        return {word_tuple[0]: index + 1 for index, word_tuple in enumerate(tuples)}


class NumpyTokenizer(Tokenizer):
    """Tokenizer that return its sequences, arrays or lists as numpy arrays"""

    def text_to_sequence(self, text, padding=None):
        return np.asarray(super(NumpyTokenizer, self).text_to_sequence(text, padding), dtype=np.uint16)

    def texts_to_sequences(self, texts, padding=None):
        return np.asarray(super(NumpyTokenizer, self).texts_to_sequences(texts, padding), dtype=np.uint16)

    def one_hot(self, word_idx):
        # Check if the given index is from the word_idx dictionary
        if word_idx > self.num_words - 1:
            raise ValueError(
                'Tokenizer cannot encode the given index. The index {} is not recognized as being one of the '
                'tokenizer words'.format(word_idx))
        return (np.arange(self.num_words) == word_idx).astype(np.bool)

    def text_to_one_hot_seq(self, text, padding=None):
        return np.asarray(super(NumpyTokenizer, self).text_to_one_hot_seq(text, padding), dtype=np.bool)


class VQATokenizer(NumpyTokenizer):
    """Specific implementation of the Tokenizer class for the VQA text data.
    It knows how to parse the data from the JSONs containing the questions and answers to feed itself and create the
    dictionary of words.
    """

    def __init__(self, questions_path, annotations_path, max_words=None, lower=True):
        """A successive call to feed() needs to be made in order to actually automatically create the word dictionary"""
        super(VQATokenizer, self).__init__(max_words, lower)
        self.questions_path = questions_path
        self.annotations_path = annotations_path

    def feed(self):
        # Load JSON files
        print 'Loading questions and answers from JSON files...'
        with open(self.questions_path, 'r') as f:
            questions_json = json.load(f)
        with open(self.annotations_path, 'r') as f:
            annotations_json = json.load(f)

        # Parse question and answer string from the files
        print 'Parsing questions...'
        questions = [question['question'] for question in questions_json['questions']]
        print 'Parsing answers...'
        answers = [answer['answer'] for annotation in annotations_json['annotations']
                   for answer in annotation['answers']]

        # Create the word dictionary fitting on the questions and answers
        print 'Fitting texts...'
        self.fit_on_texts(questions + answers)
        print 'Done!'
        return self