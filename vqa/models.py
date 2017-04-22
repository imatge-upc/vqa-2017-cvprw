import os
import h5py
import numpy as np
import json
import scipy.io
import shutil

from dframe.model.model import Model as DFModel

from keras.applications.vgg16 import VGG16
from keras.callbacks import TensorBoard, Callback
from keras.layers import Input, Embedding, merge, Dropout, Dense, LSTM, Convolution2D, MaxPooling2D, Flatten, ZeroPadding2D
from keras.models import Model, Sequential, model_from_json
from keras.optimizers import Adam
from keras.utils.visualize_util import plot
from keras.utils.data_utils import get_file

from vqa import config


class ModelZero(DFModel):
    def __init__(self, vocabulary_size=20000, question_max_len=22, model_path='model_0.p'):
        self.vqa_model = None
        self.EMBED_HIDDEN_SIZE = 100
        self.vocabulary_size = vocabulary_size
        self.question_max_len = question_max_len
        if not os.path.isdir(config.MODELS_PATH):
            os.mkdir(config.MODELS_PATH)
        self.MODEL_PATH = os.path.join(config.MODELS_PATH, model_path)
        self.weights_path = os.path.join(config.MODELS_PATH, 'weights_m0.h5')
        self.build()

    def build(self):
        # Params
        lstm_hidden_units = 256
        # Optimizer
        adam = Adam(lr=1e-4)
        # Create/load model
        try:
            with open(self.MODEL_PATH, 'r') as f:
                print('Loading baseline model...')
                self.vqa_model = model_from_json(f.read())
                print('Baseline model loaded')
                print('Compiling baseline model...')
                self.vqa_model.compile(optimizer=adam, loss='categorical_crossentropy')
                print('Model compiled')
        except IOError:
            print('Creating baseline model...')
            # Image
            image_input = Input(shape=(1024,))
            image_features = Dense(output_dim=lstm_hidden_units, activation='relu')(image_input)
            image_features = Dropout(0.5)(image_features)

            # Question
            question_input = Input(shape=(self.question_max_len,), dtype='int32')
            question_embedded = Embedding(input_dim=self.vocabulary_size, output_dim=self.EMBED_HIDDEN_SIZE,
                                          input_length=self.question_max_len)(question_input)  # Can't use masking
            question_embedded = Dropout(0.5)(question_embedded)
            sentence_embedded = LSTM(lstm_hidden_units, return_sequences=False)(question_embedded)
            sentence_embedded = Dropout(0.5)(sentence_embedded)

            # Merge
            merged = merge([image_features, sentence_embedded], mode='sum')  # Merge for layers, merge for tensors
            output = Dense(output_dim=self.vocabulary_size, activation='softmax')(merged)

            self.vqa_model = Model(input=[image_input, question_input], output=output)
            print('Baseline model created')
            plot(self.vqa_model, show_shapes=True, to_file='model_0.png', show_layer_names=False)
            print('Compiling baseline model...')
            self.vqa_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
            print('Baseline model compiled')

            print('Saving baseline model...')
            model_json = self.vqa_model.to_json()
            with open(self.MODEL_PATH, 'w') as f:
                f.write(model_json)
            print('Baseline model saved')

    def train(self, train_dataset, val_dataset, batch_size, epochs):
        tbcb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
        self.vqa_model.fit_generator(train_dataset.generator(batch_size), samples_per_epoch=train_dataset.len(),
                                     nb_epoch=epochs, validation_data=val_dataset.generator(batch_size),
                                     nb_val_samples=val_dataset.len(), callbacks=[tbcb])
        self.vqa_model.save_weights(self.weights_path)

    def validate(self, dataset, weights_path, batch_size):
        print('Loading weights...')
        self.vqa_model.load_weights(weights_path)
        print('Weights loaded')
        print('Start validation...')
        result = self.vqa_model.evaluate_generator(dataset.generator(batch_size), val_samples=dataset.len())
        print('Validated. Loss: {}'.format(result))
        return result

    def test(self, dataset, weights_path, batch_size):
        print('Loading weights...')
        self.vqa_model.load_weights(weights_path)
        print('Weights loaded')
        print('Predicting...')
        images = []
        questions = []
        features = scipy.io.loadmat(config.PRETRAINED_FEATURES_PATH + '/test_ImageNet_FisherVectors.mat')['features']
        for sample in dataset.get_samples():
            images.append(sample.get_image(features))
            questions.append(sample.get_question())
        results = self.vqa_model.predict([np.array(images), np.array(questions)], batch_size)
        print('Answers predicted')

        print('Transforming results...')
        results = np.argmax(results, axis=1)  # Max index evaluated on rows (1 row = 1 sample)
        results = list(results)
        print('Results transformed')

        print('Building reverse word dictionary...')
        word_dict = {idx: word for word, idx in dataset.tokenizer.word_index.iteritems()}
        print('Reverse dictionary build')

        print('Saving results...')
        results_dict = [{'answer': word_dict[results[idx]], 'question_id': sample._question.id}
                        for idx, sample in enumerate(dataset.get_samples())]
        with open('model_zero_results', 'w') as f:
            json.dump(results_dict, f)
        print('Results saved')

    def predict(self):
        pass


class ModelOne(DFModel):
    def __init__(self, vocabulary_size=20000, question_max_len=22, model_path='model_1.p'):
        self.vqa_model = None
        self.EMBED_HIDDEN_SIZE = 100
        self.vocabulary_size = vocabulary_size
        self.question_max_len = question_max_len
        if not os.path.isdir(config.MODELS_PATH):
            os.mkdir(config.MODELS_PATH)
        self.MODEL_PATH = os.path.join(config.MODELS_PATH, model_path)
        self.weights_path = os.path.join(config.MODELS_PATH, 'weights_m1.h5')
        self.build()

    def build(self):
        # Params
        lstm_hidden_units = 256
        # Optimizer
        adam = Adam(lr=1e-4)
        #VGG-16 Weights Paths:
        weights_path ='https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

        # Create/load model
        try:
            with open(self.MODEL_PATH, 'r') as f:
                print('Loading baseline model...')
                self.vqa_model = model_from_json(f.read())
                print('Baseline model loaded')
                print('Compiling baseline model...')
                self.vqa_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
                print('Model compiled')
        except IOError:
            print('Creating baseline model...')
            # VGG:
            # Set to all layers trainable=False to freeze VGG-16 weights'
            vgg = Sequential()
            vgg.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3), trainable=False))
            vgg.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
            vgg.add(ZeroPadding2D((1, 1), trainable=False))
            vgg.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
            vgg.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

            vgg.add(ZeroPadding2D((1, 1), trainable=False))
            vgg.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
            vgg.add(ZeroPadding2D((1, 1), trainable=False))
            vgg.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
            vgg.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

            vgg.add(ZeroPadding2D((1, 1)))
            vgg.add(Convolution2D(256, 3, 3, activation='relu'))
            vgg.add(ZeroPadding2D((1, 1)))
            vgg.add(Convolution2D(256, 3, 3, activation='relu'))
            vgg.add(ZeroPadding2D((1, 1)))
            vgg.add(Convolution2D(256, 3, 3, activation='relu'))
            vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

            vgg.add(ZeroPadding2D((1, 1)))
            vgg.add(Convolution2D(512, 3, 3, activation='relu'))
            vgg.add(ZeroPadding2D((1, 1)))
            vgg.add(Convolution2D(512, 3, 3, activation='relu'))
            vgg.add(ZeroPadding2D((1, 1)))
            vgg.add(Convolution2D(512, 3, 3, activation='relu'))
            vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

            vgg.add(ZeroPadding2D((1, 1)))
            vgg.add(Convolution2D(512, 3, 3, activation='relu'))
            vgg.add(ZeroPadding2D((1, 1)))
            vgg.add(Convolution2D(512, 3, 3, activation='relu'))
            vgg.add(ZeroPadding2D((1, 1)))
            vgg.add(Convolution2D(512, 3, 3, activation='relu'))
            vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

            vgg.add(Flatten(trainable=False))
            vgg.add(Dense(4096, activation='relu'))
            vgg.add(Dropout(0.5))
            vgg.add(Dense(4096, activation='relu'))
            vgg.add(Dropout(0.5))
            vgg.add(Dense(1000, activation='softmax'))

            # Load VGG-16 weights
            print 'Obtaining VGG weights...'
            prepared_weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    weights_path, cache_subdir='models')
            print 'VGG weights obtained!'
            print 'Preparing and loading VGG weights...'
            vgg.load_weights(prepared_weights_path)
            vgg.layers.pop()
            vgg.layers.pop()
            vgg.layers.pop()
            vgg.outputs = [vgg.layers[-1].output]
            vgg.layers[-1].outbound_nodes = []
            print 'VGG weights prepared and loaded!'

            #Image:
            image_input = Input(shape=(224, 224, 3))
            x = vgg(image_input)
            image_features = Dense(output_dim=lstm_hidden_units, activation='relu')(x)

            # Question
            question_input = Input(shape=(self.question_max_len,), dtype='int32')
            question_embedded = Embedding(input_dim=self.vocabulary_size, output_dim=self.EMBED_HIDDEN_SIZE,
                                          input_length=self.question_max_len)(question_input)  # Can't use masking
            question_embedded = Dropout(0.5)(question_embedded)
            sentence_embedded = LSTM(lstm_hidden_units, return_sequences=False)(question_embedded)
            sentence_embedded = Dropout(0.5)(sentence_embedded)

            # Merge
            merged = merge([image_features, sentence_embedded], mode='sum')  # Merge for layers, merge for tensors
            output = Dense(output_dim=self.vocabulary_size, activation='softmax')(merged)

            self.vqa_model = Model(input=[image_input, question_input], output=output)
            print('Baseline model created')
            plot(self.vqa_model, show_shapes=True, to_file='model_1.png', show_layer_names=False)
            print('Compiling baseline model...')
            self.vqa_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
            print('Baseline model compiled')

            print('Saving baseline model...')
            model_json = self.vqa_model.to_json()
            with open(self.MODEL_PATH, 'w') as f:
                f.write(model_json)
            print('Baseline model saved')

    def train(self, train_dataset, val_dataset, batch_size, epochs):
        tbcb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
        self.vqa_model.fit_generator(train_dataset.generator(batch_size), samples_per_epoch=train_dataset.len(),
                                     nb_epoch=epochs, validation_data=val_dataset.generator(batch_size),
                                     nb_val_samples=val_dataset.len(), callbacks=[tbcb])
        self.vqa_model.save_weights(self.weights_path)

    def validate(self, dataset, weights_path, batch_size):
        print('Loading weights...')
        self.vqa_model.load_weights(weights_path)
        print('Weights loaded')
        print('Start validation...')
        result = self.vqa_model.evaluate_generator(dataset.generator(batch_size), val_samples=dataset.len())
        print('Validated. Loss: {}'.format(result))
        return result

    def test(self, dataset, weights_path, batch_size):
        print('Loading weights...')
        self.vqa_model.load_weights(weights_path)
        print('Weights loaded')
        print('Predicting...')
        images = []
        questions = []
        for sample in dataset.get_samples():
            images.append(sample.get_image())
            questions.append(sample.get_question())
        results = self.vqa_model.predict([np.array(images), np.array(questions)], batch_size)
        print('Answers predicted')

        print('Transforming results...')
        results = np.argmax(results, axis=1)  # Max index evaluated on rows (1 row = 1 sample)
        results = list(results)
        print('Results transformed')

        print('Building reverse word dictionary...')
        word_dict = {idx: word for word, idx in dataset.tokenizer.word_index.iteritems()}
        print('Reverse dictionary build')

        print('Saving results...')
        results_dict = [{'answer': word_dict[results[idx]], 'question_id': sample._question.id}
                        for idx, sample in enumerate(dataset.get_samples())]
        with open('model_zero_results', 'w') as f:
            json.dump(results_dict, f)
        print('Results saved')

    def predict(self):
        pass