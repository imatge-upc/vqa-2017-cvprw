import cPickle as pickle
import json
import os
import scipy.io
import random
import numpy as np
import h5py

#from dframe.dataset.dataset import Dataset
from vqa.dframedataset import Dataset
from vqa.sample import VQASample, Image, Question, Answer

from vqa import config


class VQADataset(Dataset):
    """Class that holds the VQA 1.0 dataset of the VQA challenge.
    You can find the data here: http://www.visualqa.org/download.html"""

    PARSED_IMAGES_KEY = 'images'
    PARSED_QUESTIONS_KEY = 'questions'
    PARSED_ANSWERS_KEY = 'answers'

    def __init__(self, name, images_path, questions_path, annotations_path=None, tokenizer=None, img_pretrained_features=None):
        # Assign attributes
        self.name = name
        self.images_path = images_path
        self.questions_path = questions_path
        self.annotations_path = annotations_path
        self.tokenizer = tokenizer
        self.img_pretrained_features = img_pretrained_features
        self.PARSED_DATASET_PATH = os.path.join(config.DATA_PATH, self.name + '_parsed.p')

        super(VQADataset, self).__init__()

    def build(self, force=False):
        """Creates the dataset with all of its samples.
        This method saves a parsed version of the VQA 1.0 dataset information in ROOT_PATH/data, which can lately
        be used to reconstruct the dataset, saving a lot of processing. You can however force to reconstruct the dataset
        from scratch with the 'force' param.
        Note that it does not save the dataset itself but a middle step with the data formatted
        Args:
            force (bool): True to force the building of the dataset from the raw data. If False is given, this method
                will look for the parsed version of the data to speed up the process
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
            self.__create_samples(images, questions, answers)
            del answers
        else:
            self.__create_samples(images, questions)
        del parsed_dataset


        #path = config.DATA_PATH + "/" + self.name
        #self.save_dict_to_hdf5(images, path + "_images.h5", scipy.io.loadmat(config.PRETRAINED_FEATURES_PATH + '/train_ImageNet_FisherVectors.mat')['features'])
        #self.save_dict_to_hdf5(questions, path +"_questions.h5")
        #self.save_dict_to_hdf5(answers, path + "_answers.h5")

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
        print self.img_pretrained_features
        if self.img_pretrained_features is not None:
            if self.img_pretrained_features is 'train':
                prefix = 'COCO_train2014_'
                id_start = len(prefix)
                image_ids_path = config.PRETRAINED_FEATURES_PATH + '/train_list.txt'
            elif self.img_pretrained_features is 'val':
                prefix = 'COCO_val2014_'
                id_start = len(prefix)
                image_ids_path = config.PRETRAINED_FEATURES_PATH + '/val_list.txt'
            elif self.img_pretrained_features is 'test':
                prefix = 'COCO_test2015_'
                id_start = len(prefix)
                image_ids_path = config.PRETRAINED_FEATURES_PATH + '/test_list.txt'
            else:
                raise ValueError('Not a valid dataset partition selected')
            id_end = id_start + 12  # The string id in the image name has 12 characters
            with open(image_ids_path, 'r') as f:
                tmp = f.read()
                image_ids = tmp.split('\n')
                image_ids.remove('')  # Remove the empty item (the last one) as tmp ends with '\n'
                image_ids = map(lambda x: int(x[id_start:id_end]), image_ids)

            image_ids_dict = {}
            for idx, image_id in enumerate(image_ids):
                image_ids_dict[image_id] = idx

            images = {image_id: Image(image_id, os.path.join(self.images_path, prefix + str(image_id)), features_idx)
                      for image_id, features_idx in image_ids_dict.iteritems()}
        else:
            image_filenames = os.listdir(self.images_path)
            images = {self.get_image_id(image_filename): Image(self.get_image_id(image_filename),
                                                               os.path.join(self.images_path, image_filename))
                      for image_filename in image_filenames}
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
        answers = {(annotation['question_id'] * 10 + (answer['answer_id'] - 1)):
                       Answer(answer['answer_id'], annotation['question_id'], annotation['image_id'],
                              answer['answer'], self.tokenizer)
                   for annotation in annotations_json['annotations'] for answer in annotation['answers']}
        print 'Finish parsing annotations'

        return answers

    def __create_samples(self, images, questions, answers=None):
        if answers is not None:
            print 'Creating samples from parsed dataset...'
            for answer_id, answer in answers.iteritems():
                self.add(VQASample(images[answer.image_id], questions[answer.question_id], answer))
            print 'Finish creating samples from dataset'
        else:
            for question_id, question in questions.iteritems():
                self.add(VQASample(images[question.image_id], question))

    def generator(self, batch_size, shuffle=True):
        batch_start = 0
        num_samples = self.len()
        batch_start = 0
        batch_end = batch_size
        if self.img_pretrained_features is 'train':
            features = scipy.io.loadmat(config.PRETRAINED_FEATURES_PATH + '/train_ImageNet_FisherVectors.mat')['features']
        elif self.img_pretrained_features is 'val':
            features = scipy.io.loadmat(config.PRETRAINED_FEATURES_PATH + '/val_ImageNet_FisherVectors.mat')['features']
        elif self.img_pretrained_features is 'test':
            features = scipy.io.loadmat(config.PRETRAINED_FEATURES_PATH + '/test_ImageNet_FisherVectors.mat')['features']
        else:
            features = None
        print type(features)
        while True:
            # Get and yield the batch
            if features is not None:
                I = np.zeros((batch_size, 1024), dtype=np.float16)
            else:
                I = np.zeros((batch_size, 224, 224, 3), dtype=np.uint8)
            Q = np.zeros((batch_size, 22), dtype=np.int32)
            A = np.zeros((batch_size, 20000), dtype=np.bool_)
            for idx, sample in enumerate(self._samples[batch_start:batch_end]):
                I[idx] = sample.get_image(features)
                Q[idx] = sample.get_question()
                A[idx] = sample.get_answer()
            yield ([I, Q], A)

            batch_start += batch_size
            # An epoch has finished
            if batch_start >= num_samples:
                batch_start = 0
                # Change the order so the model won't see the samples in the same order in the next epoch
                random.shuffle(self._samples)
            batch_end = batch_start + batch_size
            if batch_end > num_samples:
                batch_end = num_samples

    def save_dict_to_hdf5(self, dic, filename, pretrained_features=None):
        with h5py.File(filename, 'w') as h5file:
            self.recursively_save_dict_contents_to_group(h5file, '/', dic, pretrained_features)

    def recursively_save_dict_contents_to_group(self, h5file, path, dic, pretrained_features):
        try:
            for key, item in dic.items():
                h5file[path + str(key)] = item.get_data(pretrained_features)
        except Exception:
            for key, item in dic.items():
                h5file[path + str(key)] = item.get_data()

    def load_dict_from_hdf5(self, filename):
        with h5py.File(filename, 'r') as h5file:
            return self.recursively_load_dict_contents_from_group(h5file, '/')

    def recursively_load_dict_contents_from_group(self, h5file, path):
        ans = {}
        for key, item in h5file[path].items():
            ans[key] = item.value

        return ans
