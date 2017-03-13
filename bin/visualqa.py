import os
import argparse

from dframe.dataset.persistence import H5pyPersistenceManager
from vqa import config
from vqa.dataset import VQADataset
from vqa.tokenizer import VQATokenizer

def main(action, force):
    print 'Action: ' + str(action)
    if str(action) == 'train':
        train_dataset = get_train_dataset(force)
    else:
        print 'Not allowed action'

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
                               conf.get_train_annotations_path(), get_tokenizer(conf), force)
    manager.save(train_dataset, os.path.join(config.DATA_PATH, 'train_dataset.h5'))
    return train_dataset


def get_tokenizer(conf):
    if os.path.isfile(conf.TOKENIZER_PATH):
        print 'Loading tokenizer'
        tokenizer = VQATokenizer.load(conf.TOKENIZER_PATH)
        print 'Finish loading tokenizer'
    else:
        # Create the tokenizer with the VQA dataset words. The constructor creates the instance but it is the feed
        # method that actually parses the dataset and creates the word dictionary. It returns the instance itself
        tokenizer = VQATokenizer(conf.get_train_questions_path(), conf.get_train_annotations_path(),
                                 max_words=conf.VOCABULARY_SIZE).feed()
        # Save it for later use
        print 'Saving tokenizer'
        tokenizer.save(conf.TOKENIZER_PATH)
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
        '-f',
        '--force',
        action='store_true',
        default=False,
        help='Add this flag if you want to force the dataset\'s parsing from scratch'
    )

    args = parser.parse_args()

    main(args.action, args.force)
