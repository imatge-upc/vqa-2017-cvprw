import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
from dframe.dataset.persistence import H5pyPersistenceManager
sys.path.insert(0,parentdir)
import config
from dataset import VQADataset


def main():
    # Check if the data directory (where we will store our preprocessed datasets) exists. Create it if is doesn't
    if not os.path.isdir(config.DATA_PATH):
        os.mkdir(config.DATA_PATH)

    # Load configuration
    conf = config.check_config()
    # Create persistence manager
    manager = H5pyPersistenceManager()
    # Create & persist train dataset
    train_dataset = VQADataset(conf.get_train_images_path(), conf.get_train_questions_path(),
                               conf.get_train_annotations_path())
    # manager.save(train_dataset, os.path.join(config.DATA_PATH, 'train_dataset.h5'))


if __name__ == '__main__':
    main()
