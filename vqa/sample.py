from abc import ABCMeta
import numpy as np

from dframe.dataset.sample import Value

from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences

class VQASample:
    """Base class to hold the information of an example/sample in a dataset"""

    def __init__(self, image, question, answer=None):
        if image is None or question is None:
            raise ValueError('A sample must have at least an image and a question')

        self._image = image
        self._question = question
        self._answer = answer

    def get_image(self, pretrained_features=None):
        return self._image.get_data(pretrained_features)

    def get_question(self):
        return self._question.get_data()

    def get_answer(self):
        return self._answer.get_data()

    @staticmethod
    def _get_elems_length(elems):
        length = 0
        if elems is not None:
            try:
                length = len(elems)
            except TypeError:
                length = 1
        return length


class Image:
    def __init__(self, image_id, image_path, features_idx=None):
        self.id = image_id
        self.path = image_path
        self.features_idx = features_idx

    def get_data(self, pretrained_features=None):
        if pretrained_features is None:
            img = image.load_img(self.path, target_size=(224, 224))
            return image.img_to_array(img)
        return pretrained_features[self.features_idx]


class Text:

    __metaclass__ = ABCMeta

    def __init__(self, text, tokenizer):
        self.text = text
        self.tokenizer = tokenizer

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def pad_sequence(self, sequence, maxlen, padding='start'):
        # If the sequence already has exactly maxlen items, return it
        if len(sequence) == maxlen:
            return sequence

        if padding != 'start' and padding != 'end':
            raise ValueError(
                'The padding method {} is not correct. It can be either \'start\' or \'end\''.format(padding))

        len_dif = abs(len(sequence) - maxlen)
        if maxlen > len(sequence):
            return (np.concatenate([[0] * len_dif, sequence])) if (padding == 'start') else (np.concatenate([sequence, [0] * len_dif]))
        else:
            return sequence[len_dif:] if (padding == 'start') else sequence[:-len_dif]


class Question(Text):

    def __init__(self, question_id, image_id, question_string, tokenizer):
        self.id = question_id
        self.image_id = image_id
        super(Question, self).__init__(question_string, tokenizer)

    def get_question_id(self):
        return self.id

    def get_data(self):
        try:
            question = self.tokenizer.texts_to_sequences([self.text.encode('utf8')])[0]
            return pad_sequences([question], 22)[0]
        except AttributeError:
            raise AttributeError(
                'No tokenizer has been set in order to process the text. Use set_tokenizer or the constructor param')


class Answer(Text):

    def __init__(self, answer_id, question_id, image_id, answer_string, tokenizer):
        self.id = answer_id
        self.question_id = question_id
        self.image_id = image_id
        self.tokens_idx = None
        super(Answer, self).__init__(answer_string, tokenizer)

    def get_data(self):
        try:
            self.tokens_idx = self.tokenizer.texts_to_sequences([self.text.encode('utf8')])[0]
            one_hot_ans = np.zeros(20000)
            if self.tokens_idx:
                idx = self.tokens_idx[0]
                one_hot_ans[idx] = 1
            return one_hot_ans.astype(np.bool_)
        except AttributeError:
            raise AttributeError(
                'No tokenizer has been set in order to process the text. Use set_tokenizer or the constructor param')


