import os
import cPickle as pickle

from keras.preprocessing.text import Tokenizer
from dframe.dataset.persistence import H5pyPersistenceManager
from vqa import config
from vqa.dataset import VQADataset

def main():
    # Check if the data directory (where we will store our preprocessed datasets) exists. Create it if is doesn't
    if not os.path.isdir(config.DATA_PATH):
        os.mkdir(config.DATA_PATH)

    # Load configuration
    conf = config.check_config()
    TOKENIZER_PATH = os.path.join(config.DATA_PATH, 'tokenizer.p')
    # Create persistence manager
    manager = H5pyPersistenceManager()
    # Create & persist train dataset
    train_dataset = VQADataset(conf.get_train_images_path(), conf.get_train_questions_path(),
                               conf.get_train_annotations_path(), get_tokenizer(TOKENIZER_PATH), True)
    manager.save(train_dataset, os.path.join(config.DATA_PATH, 'train_dataset.h5'))

def get_tokenizer(tokenizer_path):
    tokenizer_dir = os.path.dirname(os.path.abspath(tokenizer_path))
    if not os.path.isdir(tokenizer_dir):
        os.mkdir(tokenizer_dir)

    # Tokenizer
    if os.path.isfile(tokenizer_path):
        print 'Loading tokenizer from: {}'.format(tokenizer_path)
        tokenizer = pickle.load(open(tokenizer_path, 'r'))
    else:
        tokenizer = Tokenizer(nb_words=20000)  # TODO: Hardcoded!!!
        print 'Saving tokenizer...'
        pickle.dump(tokenizer, open(tokenizer_path, 'w'))
    return tokenizer

if __name__ == '__main__':
    main()
