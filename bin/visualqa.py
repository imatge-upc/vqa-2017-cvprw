import os
import argparse
import cPickle as pickle
import json
import h5py
import numpy as np
import scipy.io
from keras import backend as K
#from dframe.dataset.persistence import H5pyPersistenceManager
from dframe.dataset.persistence import PicklePersistenceManager
from vqa import config
from vqa.dataset import VQADataset
#from vqa.tokenizer import VQATokenizer
from vqa.models import ModelZero
from keras.preprocessing.text import Tokenizer

EPOCHS = 40
BATCH_SIZE = 128
VOCABULARY_SIZE = 20000


def main(action, model_id, force):
    # Create persistence manager
    manager = PicklePersistenceManager()
    tokenizer = get_tokenizer(os.path.join(config.DATA_PATH, 'tokenizer.p'))
    model = get_model(int(model_id))
    if str(action) == 'train':
        train_dataset = get_train_dataset(manager, tokenizer, force)
        val_dataset = get_val_dataset(manager, tokenizer, force)
        #test_dataset = get_test_dataset(manager, tokenizer, force)

        model.train(train_dataset, val_dataset, BATCH_SIZE, EPOCHS)
        model.validate(val_dataset, config.MODELS_PATH + '/weights_m0.h5', BATCH_SIZE)
        #model.test(test_dataset,config.MODELS_PATH + '/weights_m0.h5', BATCH_SIZE )
    else:
        print 'Not allowed action'
    K.clear_session()


def get_model(model_id):
    switcher = {
        0: ModelZero()
    }
    return switcher.get(model_id)


def get_train_dataset(manager, tokenizer, force = False):
    train_dataset_path = os.path.join(config.DATA_PATH, 'train_dataset.h5')
    # Check if the data directory (where we will store our preprocessed datasets) exists. Create it if is doesn't
    if not os.path.isdir(config.DATA_PATH):
        os.mkdir(config.DATA_PATH)
    if os.path.isfile(train_dataset_path):
        with open(train_dataset_path, 'rb') as f:
            print 'Loading train dataset from: {}'.format(train_dataset_path)
            train_dataset = manager.load(f)
            print 'Finished loading train dataset'
    else:
        # Load configuration
        conf = config.check_config()
        # Create & persist train dataset
        print 'Creating train dataset'
        train_dataset = VQADataset("train_dataset", conf.get_train_images_path(), conf.get_train_questions_path(),
                                   conf.get_train_annotations_path(), tokenizer, 'train').build(force)
    return train_dataset


def get_val_dataset(manager, tokenizer, force = False):
    val_dataset_path = os.path.join(config.DATA_PATH, 'val_dataset.p')
    # Check if the data directory (where we will store our preprocessed datasets) exists. Create it if is doesn't
    if not os.path.isdir(config.DATA_PATH):
        os.mkdir(config.DATA_PATH)
    if os.path.isfile(val_dataset_path):
        with open(val_dataset_path, 'rb') as f:
            print 'Loading validation dataset from: {}'.format(val_dataset_path)
            val_dataset = pickle.load(f)
            print 'Finished loading validation dataset'
    else:
        # Load configuration
        conf = config.check_config()
        # Create & persist train dataset
        print 'Creating validation dataset'
        val_dataset = VQADataset("val_dataset", conf.get_val_images_path(), conf.get_val_questions_path(),
                                   conf.get_val_annotations_path(), tokenizer, 'val').build(force)
    return val_dataset

def get_test_dataset(manager, tokenizer, force = False):
    test_dataset_path = os.path.join(config.DATA_PATH, 'test_dataset.p')
    # Check if the data directory (where we will store our preprocessed datasets) exists. Create it if is doesn't
    if not os.path.isdir(config.DATA_PATH):
        os.mkdir(config.DATA_PATH)
    if os.path.isfile(test_dataset_path):
        with open(test_dataset_path, 'rb') as f:
            print 'Loading validation dataset from: {}'.format(test_dataset_path)
            test_dataset = pickle.load(f)
            print 'Finished loading validation dataset'
    else:
        # Load configuration
        conf = config.check_config()
        # Create & persist train dataset
        print 'Creating validation dataset'
        test_dataset = VQADataset("test_dataset", conf.get_test_images_path(), conf.get_test_questions_path(),
                                 tokenizer=tokenizer, img_pretrained_features='test').build(force)
    return test_dataset


def get_tokenizer(tokenizer_path):
    if os.path.isfile(tokenizer_path):
        print 'Loading tokenizer'
        tokenizer = pickle.load(open(tokenizer_path, 'r'))
        print 'Finish loading tokenizer'
    else:
        conf = config.check_config()
        tokenizer = Tokenizer(nb_words=VOCABULARY_SIZE)
        print 'Loading questions and answers from JSON files...'
        with open(conf.get_train_questions_path(), 'r') as f:
            questions_json = json.load(f)
        with open(conf.get_train_annotations_path(), 'r') as f:
            annotations_json = json.load(f)

        # Parse question and answer string from the files
        print 'Parsing questions...'
        questions = [question['question'] for question in questions_json['questions']]
        print 'Parsing answers...'
        answers = [answer['answer'] for annotation in annotations_json['annotations']
                   for answer in annotation['answers']]
        # Create the word dictionary fitting on the questions and answers
        print 'Fitting texts...'
        words = questions + answers
        words = [word.encode('utf8') for word in words]
        tokenizer.fit_on_texts(words)
        print 'Done!'
        # Save it for later use
        print 'Saving tokenizer'
        pickle.dump(tokenizer, open(tokenizer_path,'w'))
        print 'Finish saving tokenizer'
    return tokenizer

##############################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main entry point to interact with the VQA module')
    parser.add_argument(
        '-a',
        '--action',
        choices=['train'],
        help='Which action should be performed on the model.'
    )
    parser.add_argument(
        '-m',
        '--model',
        choices=['0'],
        default='0',
        help='Model to select. Model 0 refers to the baseline model.'
    )
    parser.add_argument(
        '-f',
        '--force',
        action='store_true',
        default=False,
        help='Add this flag if you want to force the dataset parsing from scratch'
    )

    args = parser.parse_args()

    main(args.action, args.model, args.force)
