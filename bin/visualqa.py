import os
import argparse

from dframe.dataset.persistence import H5pyPersistenceManager
from vqa import config
from vqa.dataset import VQADataset
from vqa.tokenizer import VQATokenizer
from vqa.models import Model_0

EPOCHS = 1
BATCH_SIZE = 128
VOCABULARY_SIZE = 2000


def main(action, model_id, force):
    model = get_model(model_id)
    if str(action) == 'train':
        train_dataset = get_train_dataset(force)
        model.train(train_dataset, BATCH_SIZE, EPOCHS)
    else:
        print 'Not allowed action'


def get_model(model_id):
    switcher = {
        0: Model_0()
    }
    return switcher.get(model_id)

def get_train_dataset(force = False):
    # Check if the data directory (where we will store our preprocessed datasets) exists. Create it if is doesn't
    if not os.path.isdir(config.DATA_PATH):
        os.mkdir(config.DATA_PATH)

    # Load configuration
    conf = config.check_config()
    tokenizer_path = os.path.join(config.DATA_PATH, 'tokenizer.p')
    # Create persistence manager
    manager = H5pyPersistenceManager()
    # Create & persist train dataset
    train_dataset = VQADataset(conf.get_train_images_path(), conf.get_train_questions_path(),
                               conf.get_train_annotations_path(), get_tokenizer(conf, tokenizer_path), force).build()
    manager.save(train_dataset, os.path.join(config.DATA_PATH, 'train_dataset.h5'))
    return train_dataset


def get_tokenizer(conf, tokenizer_path):
    if os.path.isfile(tokenizer_path):
        print 'Loading tokenizer'
        tokenizer = VQATokenizer.load(tokenizer_path)
        print 'Finish loading tokenizer'
    else:
        # Create the tokenizer with the VQA dataset words. The constructor creates the instance but it is the feed
        # method that actually parses the dataset and creates the word dictionary. It returns the instance itself
        tokenizer = VQATokenizer(conf.get_train_questions_path(), conf.get_train_annotations_path(),
                                 max_words=VOCABULARY_SIZE).feed()
        # Save it for later use
        print 'Saving tokenizer'
        tokenizer.save(tokenizer_path)
        print 'Finish saving tokenizer'

    return tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main entry point to interact with the VQA module')
    parser.add_argument(
        '-a',
        '--action',
        choices=['train'],
        help='Which action should be perform on the model.'
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
