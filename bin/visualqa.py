import os
import argparse
import cPickle as pickle
import json

from keras import backend as K
from vqa import config
from vqa.dataset import VQADataset
from vqa.models import ModelZero, ModelOne, ModelTwo, ModelThree, ModelFour, ModelFive
from keras.preprocessing.text import Tokenizer

EPOCHS = 40
BATCH_SIZE = 100
VOCABULARY_SIZE = 10000
QUESTION_MAX_LEN = 22


def main(action, model_id, force, tokenizer_path, weights):
    tokenizer = get_tokenizer(os.path.join(config.DATA_PATH, tokenizer_path))
    model = get_model(int(model_id))
    if str(action) == 'train':
        train_dataset = get_train_dataset(tokenizer, force)
        val_dataset = get_val_dataset(tokenizer, force)
        model.train(train_dataset, val_dataset, BATCH_SIZE, EPOCHS)
    elif str(action) == 'test':
        test_dataset = get_test_dataset(tokenizer, force)
        model.test(test_dataset, os.path.join(config.MODELS_PATH, weights), BATCH_SIZE )
    else:
        print 'Not allowed action'
    K.clear_session()


def get_model(model_id):
    switcher = {
        0: ModelZero(vocabulary_size=VOCABULARY_SIZE, question_max_len=QUESTION_MAX_LEN),
        1: ModelOne(vocabulary_size=VOCABULARY_SIZE, question_max_len=QUESTION_MAX_LEN),
        2: ModelTwo(vocabulary_size=VOCABULARY_SIZE, question_max_len=QUESTION_MAX_LEN),
        3: ModelThree(vocabulary_size=VOCABULARY_SIZE, question_max_len=QUESTION_MAX_LEN),
        4: ModelFour(vocabulary_size=VOCABULARY_SIZE, question_max_len=QUESTION_MAX_LEN),
        5: ModelFive(vocabulary_size=VOCABULARY_SIZE, question_max_len=QUESTION_MAX_LEN)
    }
    return switcher.get(model_id)


def get_train_dataset(tokenizer, force = False):
    # Load configuration
    conf = config.check_config()
    # Create & persist train dataset
    print 'Creating train dataset'
    train_dataset = VQADataset("train_dataset", conf.get_train_images_path(), conf.get_train_questions_path(),
                               conf.get_train_annotations_path(), tokenizer, 'n_a').build(force)
    return train_dataset


def get_val_dataset(tokenizer, force = False):
    # Load configuration
    conf = config.check_config()
    # Create & persist val dataset
    print 'Creating validation dataset'
    val_dataset = VQADataset("val_dataset", conf.get_val_images_path(), conf.get_val_questions_path(),
                               conf.get_val_annotations_path(), tokenizer, 'n_a').build(force)
    return val_dataset

def get_test_dataset(tokenizer, force = False):
    # Load configuration
    conf = config.check_config()
    # Create & persist test dataset
    print 'Creating test dataset'
    test_dataset = VQADataset("test-dev_dataset", conf.get_test_images_path(), conf.get_test_questions_path(),
                             tokenizer=tokenizer, img_pretrained_features='n_a').build(force)
    return test_dataset


def get_tokenizer(tokenizer_path):
    if os.path.isfile(tokenizer_path):
        print 'Loading tokenizer'
        tokenizer = pickle.load(open(tokenizer_path, 'r'))
        print 'Finish loading tokenizer'
    else:
        conf = config.check_config()
        tokenizer = Tokenizer(VOCABULARY_SIZE)
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
        choices=['train', 'test'],
        help='Which action should be performed on the model.'
    )
    parser.add_argument(
        '-m',
        '--model',
        choices=['0', '1', '2', '3', '4', '5'],
        default='3',
        help='Model to select. Model 0: baseline model with pretrained image features. Model 1: VGG based image features'
    )
    parser.add_argument(
        '-f',
        '--force',
        action='store_true',
        default=False,
        help='Add this flag if you want to force the dataset parsing from scratch'
    )
    parser.add_argument(
        '-tk',
        '--tokenizer',
        default='tokenizer.p'
    )
    parser.add_argument(
        '-w',
        '--weights',
        help='Add weights path to generate test submission file.'
    )

    args = parser.parse_args()

    main(args.action, args.model, args.force, args.tokenizer, args.weights)
