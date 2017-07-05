import cPickle as pickle
import json
import os
import scipy.io
import random
import numpy as np
import h5py
from itertools import izip_longest, ifilter
import threading

from keras.applications.vgg16 import VGG16, preprocess_input as vgg_prep_input
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet_prep_input
from dframe.dataset.dataset import Dataset
from vqa.sample import VQASample, Image, Question, Answer

from vqa import config


class VQADataset(Dataset):
    """Class that holds the VQA 1.0 and 2.0 dataset of the VQA challenge.
    You can find the data here: http://www.visualqa.org/download.html"""

    PARSED_IMAGES_KEY = 'images'
    PARSED_QUESTIONS_KEY = 'questions'
    PARSED_ANSWERS_KEY = 'answers'

    def __init__(self, name, images_path, questions_path, annotations_path=None, tokenizer=None,
                 img_pretrained_features=None):
        # Assign attributes
        self.name = name
        self.images_path = images_path
        self.questions_path = questions_path
        self.annotations_path = annotations_path
        self.tokenizer = tokenizer
        self.img_pretrained_features = img_pretrained_features
        self.PARSED_DATASET_PATH = os.path.join(config.DATA_PATH, self.name + '_parsed.p')

        super(VQADataset, self).__init__()

    def build(self, force=False, denoise=0):
        """Creates the dataset with all of its samples.
        This method saves a parsed version of the VQA 1.0 or 2.0 dataset information in ROOT_PATH/data, which can lately
        be used to reconstruct the dataset, saving a lot of processing. You can however force to reconstruct the dataset
        from scratch with the 'force' param.
        Note that it does not save the dataset itself but a middle step with the data formatted
        Args:
            force (bool): True to force the building of the dataset from the raw data. If False is given, this method
                will look for the parsed version of the data to speed up the process
            denoise (int): Threshold to denoise dataset. By default set to 0 (no denoising).
        """
        print 'Building dataset...'
        if force:
            parsed_dataset = self.__parse_dataset()
        else:
            parsed_dataset = self.__retrieve_parsed_dataset()
            if parsed_dataset is None:
                parsed_dataset = self.__parse_dataset()

        images = parsed_dataset[self.PARSED_IMAGES_KEY]
        questions = parsed_dataset[self.PARSED_QUESTIONS_KEY]
        if self.annotations_path is not None:
            answers = parsed_dataset[self.PARSED_ANSWERS_KEY]
            if denoise:
                answers = self.__denoise_dataset(answers, denoise)
            self.__create_samples(images, questions, answers)
            del answers
        else:
            self.__create_samples(images, questions)

        del parsed_dataset
        del images
        del questions
        return self

    @staticmethod
    def get_image_id(image_name):
        # Image name has the form: COCO_[train|val|test][2014|2015]_{image_id 12 digits}.jpg
        image_id = image_name.split('_')[-1]  # Retrieve image_id.jpg
        image_id = image_id.split('.')[0]  # Remove file format
        return int(image_id)  # Convert to int and return

    def __retrieve_parsed_dataset(self):
        if not os.path.isfile(self.PARSED_DATASET_PATH):
            return None

        with open(self.PARSED_DATASET_PATH, 'rb') as f:
            print 'Loading parsed dataset from: {}'.format(self.PARSED_DATASET_PATH)
            parsed_dataset = pickle.load(f)

        return parsed_dataset

    def __parse_dataset(self):
        # Parse partial sets
        print 'Parsing dataset...'
        images = self.__parse_images()
        questions = self.__parse_questions()
        if self.annotations_path is not None:
            answers = self.__parse_annotations()
            parsed_dataset = {self.PARSED_IMAGES_KEY: images,
                              self.PARSED_QUESTIONS_KEY: questions,
                              self.PARSED_ANSWERS_KEY: answers}
        else:
            parsed_dataset = {self.PARSED_IMAGES_KEY: images,
                              self.PARSED_QUESTIONS_KEY: questions}

         # Save parsed dataset for later reuse
        if not os.path.isdir(config.DATA_PATH):
            os.mkdir(config.DATA_PATH)
        with open(self.PARSED_DATASET_PATH, 'wb') as f:
            print 'Saving parsed dataset to: {}'.format(self.PARSED_DATASET_PATH)
            pickle.dump(parsed_dataset, f)

        return parsed_dataset

    def __parse_images(self):
        print 'Parsing images from image directory: {}'.format(self.images_path)
        if self.img_pretrained_features.split("_")[1] is 'kcnn':
            if self.img_pretrained_features is 'train_kcnn':
                prefix = 'COCO_train2014_'
                id_start = len(prefix)
                image_ids_path = config.PRETRAINED_FEATURES_PATH + '/train_list.txt'
            elif self.img_pretrained_features is 'val_kcnn':
                prefix = 'COCO_val2014_'
                id_start = len(prefix)
                image_ids_path = config.PRETRAINED_FEATURES_PATH + '/val_list.txt'
            elif self.img_pretrained_features is 'test_kcnn':
                prefix = 'COCO_test2015_'
                id_start = len(prefix)
                image_ids_path = config.PRETRAINED_FEATURES_PATH + '/test_list.txt'
            else:
                raise ValueError('Not a valid dataset partition selected')

            if self.img_pretrained_features is 'train_kcnn':
                features = scipy.io.loadmat(config.PRETRAINED_FEATURES_PATH + '/train_ImageNet_FisherVectors.mat')[
                    'features']
            elif self.img_pretrained_features is 'val_kcnn':
                features = scipy.io.loadmat(config.PRETRAINED_FEATURES_PATH + '/val_ImageNet_FisherVectors.mat')[
                    'features']
            elif self.img_pretrained_features is 'test_kcnn':
                features = scipy.io.loadmat(config.PRETRAINED_FEATURES_PATH + '/test_ImageNet_FisherVectors.mat')[
                    'features']
            else:
                features = None

            id_end = id_start + 12  # The string id in the image name has 12 characters
            with open(image_ids_path, 'r') as f:
                tmp = f.read()
                image_ids = tmp.split('\n')
                image_ids.remove('')  # Remove the empty item (the last one) as tmp ends with '\n'
                image_ids = map(lambda x: int(x[id_start:id_end]), image_ids)

            image_ids_dict = {}
            for idx, image_id in enumerate(image_ids):
                image_ids_dict[image_id] = idx

            images = {
                image_id: Image(image_id, os.path.join(self.images_path, prefix + str(image_id)), features_idx, features)
                for image_id, features_idx in image_ids_dict.iteritems()}
        else:
            image_filenames = os.listdir(self.images_path)
            print 'Length of image filenames: '+ str(len(image_filenames))
            images = {self.get_image_id(image_filename): Image(self.get_image_id(image_filename),
                                                               os.path.join(self.images_path, image_filename), idx)
                      for idx, image_filename in enumerate(image_filenames)}
        print 'Finish parsing images'
        return images

    def __parse_questions(self):
        print 'Parsing questions json file: {}'.format(self.questions_path)

        with open(self.questions_path, 'rb') as f:
            questions_json = json.load(f)
        questions = {
            question['question_id']: Question(question['question_id'], question['image_id'], question['question'],
                                              self.tokenizer) for question in questions_json['questions']}
        print 'Finish parsing questions'
        print len(questions)
        return questions

    def __parse_annotations(self):
        print 'Parsing annotations json file: {}'.format(self.annotations_path)

        with open(self.annotations_path, 'rb') as f:
            annotations_json = json.load(f)

        # (annotation['question_id'] * 10 + (answer['answer_id'] - 1): creates a unique answer id
        # The value answer['answer_id'] it is not unique across all the answers, only on the subset of answers
        # of that question.
        # As question_id is composed by appending the question number (0-2) to the image_id (which is unique)
        # we've composed the answer id the same way. The subtraction of 1 is due to the fact that the
        # answer['answer_id'] ranges from 1 to 10 instead of 0 to 9
        answers = {(int(annotation['question_id']) * 10 + (int(answer['answer_id']) - 1)):
                       Answer(answer['answer_id'], annotation['question_id'], annotation['image_id'],
                              answer['answer'], self.tokenizer)
                   for annotation in annotations_json['annotations'] for answer in annotation['answers']}
        print 'Finish parsing annotations'

        return answers

    def __create_samples(self, images=None, questions=None, answers=None):
        print 'Creating samples from parsed dataset...'
        if answers is not None:
            for answer_id, answer in answers.iteritems():
                self._samples.append(VQASample(images[answer.image_id], questions[answer.question_id], answer))
        else:
            for question_id, question in questions.iteritems():
                self._samples.append(VQASample(images[question.image_id], question))

            print 'Finish creating samples from dataset'

    def __denoise_dataset(self, answers, threshold):
        print 'Starting to clean dataset'
        cleaned_answers = {}
        chunks = [answers.iteritems()] * 10 #10 plausible answers for each question
        divided_answers = list(dict(ifilter(None, v)) for v in izip_longest(*chunks))
        for subset in divided_answers:
            answer_text = [answer.text for idx, answer in subset.iteritems()]
            cleaned = [answer for idx, answer in subset.iteritems() if answer_text.count(answer.text) > threshold]
            for ans in cleaned:
                cleaned_answers[int(ans.question_id) * 10 + int(ans.id) - 1] = ans
        print 'Length of the clean dataset: ' + str(len(cleaned_answers))
        with open(os.path.join(config.DATA_PATH, 'cleaned_answers' + str(threshold) + '.p'), 'wb') as f:
            print 'Saving denoised answers dataset with threshold {}...'.format(str(threshold))
            pickle.dump(cleaned_answers, f)
            print 'Finish saving denoised answers!'
        return cleaned_answers

    def generator(self, batch_size, shuffle=True, lang_only=False):
        lock = threading.Lock()
        with lock:
            if not lang_only:
                if self.img_pretrained_features.split("_")[1] is 'kcnn':
                    features = scipy.io.loadmat(config.PRETRAINED_FEATURES_PATH + '/' + self.img_pretrained_features.split('_')[0] +
                                                '_ImageNet_FisherVectors.mat')['features']
                elif self.img_pretrained_features.split("_")[1] is 'vgg16':
                    image_data = h5py.File(os.path.join(config.DATA_PATH, self.name+'_image_features_vgg16.h5'), 'r')
                    features = image_data['image_features']
                else:
                    image_data = h5py.File(os.path.join(config.DATA_PATH, self.name + '_image_features_resnet.h5'), 'r')
                    features = image_data['image_features']

            num_samples = self.len()
            batch_start = 0
            batch_end = batch_size

            while True:
                # Get and yield the batch
                if not lang_only:
                    if self.img_pretrained_features.split("_")[1] is 'kcnn':
                        I = np.zeros((batch_size, 1024), dtype=np.float32)
                    elif self.img_pretrained_features.split("_")[1] is 'vgg16':
                        I = np.zeros((batch_size, 25088), dtype=np.float32)
                    elif self.img_pretrained_features.split("_")[1] is 'resnet':
                        I = np.zeros((batch_size, 1, 1, 1, 2048), dtype=np.float32)
                    else:
                        I = np.zeros((batch_size, 224, 224, 3), dtype=np.uint8)
                Q = np.zeros((batch_size, 22), dtype=np.int32)
                A = np.zeros((batch_size, 10000), dtype=np.bool_)
                for idx, sample in enumerate(self._samples[batch_start:batch_end]):
                    if not lang_only:
                        if self.img_pretrained_features.split("_")[1] is 'vgg16':
                            I[idx] = np.reshape(features[sample._image.features_idx], 25088)
                        else:
                            I[idx] = features[sample._image.features_idx]
                    Q[idx] = sample.get_question()
                    A[idx] = sample.get_answer()
                if not lang_only:
                    yield ([I, Q], A)
                else:
                    yield (Q, A)

                batch_start += batch_size
                # An epoch has finished
                if batch_start >= num_samples:
                    batch_start = 0
                    # Change the order so the model won't see the samples in the same order in the next epoch
                    if shuffle:
                        random.shuffle(self._samples)
                batch_end = batch_start + batch_size
                if batch_end > num_samples:
                    batch_end = num_samples

    def extract_vgg16_image_features(self, images, verbose=1, layers2remove=0, include_top=False):
        vgg = VGG16(weights='imagenet', include_top=include_top)
        for layer in range(layers2remove):
            vgg.layers.pop()
        vgg.outputs = [vgg.layers[-1].output]
        vgg.layers[-1].outbound_nodes = []
        file = h5py.File(os.path.join(config.DATA_PATH, self.name+'_image_features_vgg'+str(layers2remove)+'.h5'), 'w')
        features = file.create_dataset('feats')
        print 'Predicting VGG-16 image features...'
        for idx, image in images.iteritems():
            features[str(idx)] = vgg.predict(vgg_prep_input(np.expand_dims(image.get_data(), axis=0)), batch_size=100, verbose=verbose)
        print 'VGG-16 image features predicted!'

        with h5py.File(os.path.join(config.DATA_PATH, self.name+'_image_features_vgg'+str(layers2remove)+'.h5'), 'w') as f:
            f.create_dataset('image_features', data=np.array(features))
            f.close()

    def extract_resnet50_image_features(self, images, verbose=0, layers2remove=0, include_top=False):
        resnet = ResNet50(weights='imagenet', include_top=include_top)
        for layer in range(layers2remove):
            resnet.layers.pop()
        resnet.outputs = [resnet.layers[-1].output]
        resnet.layers[-1].outbound_nodes = []
        features = []
        print 'Predicting ResNet-50 image features...'
        for idx, image in images.iteritems():
            features.append(resnet.predict(resnet_prep_input(np.expand_dims(image.get_data(), axis=0)), batch_size=100,
                                             verbose=verbose))
        print 'ResNet-50 image features predicted!'

        with h5py.File(os.path.join(config.DATA_PATH, self.name + '_image_features_resnet.h5'),
                       'w') as f:
            f.create_dataset('image_features', data=np.array(features))
            f.close()
