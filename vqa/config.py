import json
import os

import vqa

BIN_PATH = os.path.join(vqa.ROOT_PATH, 'bin')
CONFIG_PATH = os.path.join(vqa.ROOT_PATH, 'config')
CONFIG_FILE_PATH = os.path.join(CONFIG_PATH, 'config.json')
DATA_PATH = os.path.join(vqa.ROOT_PATH, 'data')
MODELS_PATH = os.path.join(DATA_PATH, 'models')
PRETRAINED_FEATURES_PATH = os.path.join(DATA_PATH, 'pretrained_features')
VGG_DATA = os.path.join(DATA_PATH, 'vgg')

class Config:
    """Wrapper for the configuration dictionary to ease its use outside this module, providing methods to access
    its data"""

    DATASET = 'dataset'
    DATASET_TRAIN_SET = 'train_set'
    DATASET_VAL_SET = 'val_set'
    DATASET_TEST_SET = 'test_set'
    DATASET_IMAGES_PATH = 'images_path'
    DATASET_QUESTIONS_PATH = 'questions_path'
    DATASET_ANNOTATIONS_PATH = 'annotations_path'


    def __init__(self, config):
        self._config = config

    def to_json(self, json_path):
        """Persists the configuration to the specified JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self._config, f)

    @staticmethod
    def from_json(json_path):
        """Constructs a Config object from the specified configuration JSON file"""
        with open(json_path, 'r') as f:
            return Config(json.load(f))

    def get_raw_config(self):
        """Gets the raw configuration dictionary. It must not be altered after retrieving it, this method is basically
        for validating purposes."""
        return self._config

    def get_train_images_path(self):
        return self._config[self.DATASET][self.DATASET_TRAIN_SET][self.DATASET_IMAGES_PATH]

    def get_train_questions_path(self):
        return self._config[self.DATASET][self.DATASET_TRAIN_SET][self.DATASET_QUESTIONS_PATH]

    def get_train_annotations_path(self):
        return self._config[self.DATASET][self.DATASET_TRAIN_SET][self.DATASET_ANNOTATIONS_PATH]

    def get_val_images_path(self):
        return self._config[self.DATASET][self.DATASET_VAL_SET][self.DATASET_IMAGES_PATH]

    def get_val_questions_path(self):
        return self._config[self.DATASET][self.DATASET_VAL_SET][self.DATASET_QUESTIONS_PATH]

    def get_val_annotations_path(self):
        return self._config[self.DATASET][self.DATASET_VAL_SET][self.DATASET_ANNOTATIONS_PATH]

    def get_test_images_path(self):
        return self._config[self.DATASET][self.DATASET_TEST_SET][self.DATASET_IMAGES_PATH]

    def get_test_questions_path(self):
        return self._config[self.DATASET][self.DATASET_TEST_SET][self.DATASET_QUESTIONS_PATH]


class ConfigError(Exception):
    def __init__(self, invalid_key):
        super(ConfigError, self).__init__('Configuration key {} missing or invalid'.format(invalid_key))


def check_config():
    # If the config file doesn't exist, create it
    if not os.path.isfile(CONFIG_FILE_PATH):
        return create_config()

    # Load config file
    print 'Recovering configuration...'
    config = Config.from_json(CONFIG_FILE_PATH)
    print 'Configuration loaded'

    # Validate config consistence. It's easier with the dictionary containing the configuration rather than the Config
    # object
    validate_config(config.get_raw_config())

    return config


def create_config():
    if not os.path.isdir(CONFIG_PATH):
        os.mkdir(CONFIG_PATH)

    # Print header
    _print_header()

    # Create all the different configurations needed
    # At the moment only the dataset config is needed
    dataset_config = _dataset_config()
    config = {Config.DATASET: dataset_config}
    config = Config(config)

    # Persist config file
    config.to_json(CONFIG_FILE_PATH)
    print 'Saved config file at: {}'.format(CONFIG_FILE_PATH)
    print 'You can change this configuration afterwards by manually modifying the config file'

    return config


def validate_config(config):
    print 'Validating configuration file...'
    if Config.DATASET not in config:
        raise ConfigError(Config.DATASET)
    _validate_dataset_config(config[Config.DATASET])
    print 'Configuration file valid!'


def _print_header():
    print '{:=^50}'.format('')
    print '{:=^50}'.format(' VQA ')
    print '{:=^50}'.format('')
    print ''
    print 'Welcome to VQA module! Here we will setup all the configuration that this software needs to run smoothly'
    print ''


def _dataset_config():
    print '{:=^50}'.format(' Dataset config ')
    print 'TRAIN DATASET'
    train_config = _dataset_paths(Config.DATASET_TRAIN_SET)
    print ''
    print 'VALIDATION DATASET'
    val_config = _dataset_paths(Config.DATASET_VAL_SET)
    print ''
    print 'TEST DATASET'
    test_config = _dataset_paths(Config.DATASET_TEST_SET)
    print '{:=^50}'.format('')
    print ''

    return {Config.DATASET_TRAIN_SET: train_config, Config.DATASET_VAL_SET: val_config,
            Config.DATASET_TEST_SET: test_config}


def _dataset_paths(dataset_type):
    images_path = _get_valid_path('Images path: ', mode='d')
    questions_path = _get_valid_path('Questions file path (with the filename included!): ')

    # For test sets we do not have the
    if dataset_type == Config.DATASET_TEST_SET:
        return {Config.DATASET_IMAGES_PATH: images_path, Config.DATASET_QUESTIONS_PATH: questions_path}

    annotations_path = _get_valid_path('Annotations (answers) file path (with the filename included!): ')
    return {Config.DATASET_IMAGES_PATH: images_path, Config.DATASET_QUESTIONS_PATH: questions_path,
            Config.DATASET_ANNOTATIONS_PATH: annotations_path}


def _get_valid_path(message, mode='f'):
    while True:
        path = raw_input(message)
        if mode == 'f':
            if not os.path.isfile(path):
                print 'Path is not a file! Try again'
            else:
                return path
        elif mode == 'd':
            if not os.path.isdir(path):
                print 'Path is not a directory! Try again'
            else:
                return path
        else:
            raise ValueError('Mode {} is not valid'.format(mode))


def _validate_dataset_config(dataset_config):
    for set_name, set_config in dataset_config.iteritems():
        if Config.DATASET_IMAGES_PATH not in set_config or not os.path.isdir(
                set_config[Config.DATASET_IMAGES_PATH]):
            raise ConfigError(Config.DATASET_IMAGES_PATH)
        if Config.DATASET_QUESTIONS_PATH not in set_config or not os.path.isfile(
                set_config[Config.DATASET_QUESTIONS_PATH]):
            raise ConfigError(Config.DATASET_QUESTIONS_PATH)
        if set_name != Config.DATASET_TEST_SET and \
                (Config.DATASET_ANNOTATIONS_PATH not in set_config or not
                os.path.isfile(set_config[Config.DATASET_ANNOTATIONS_PATH])):
                raise ConfigError(Config.DATASET_ANNOTATIONS_PATH)