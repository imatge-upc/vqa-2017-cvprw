import os
import h5py
import numpy as np
import json
import scipy.io
from dframe.model.model import Model as DFModel
import threading

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Embedding, merge, Dropout, Dense, LSTM, Flatten, BatchNormalization
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.utils.visualize_util import plot

from vqa import config

class ModelZero(DFModel):
    def __init__(self, vocabulary_size=10000, question_max_len=22, model_path='model_0.p'):
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
                                     nb_val_samples=val_dataset.len(), callbacks=[tbcb], max_q_size=batch_size*2)
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
            images.append(features[sample._image.features_idx])
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
    def __init__(self, vocabulary_size=10000, question_max_len=22, model_path='model_1.p'):
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

            image_input = Input(shape=(4096, ))
            x = Dropout(0.5)(image_input)
            x = Dense(output_dim=4096, activation='relu')(x)
            x = Dropout(0.5)(x)
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
                                     nb_val_samples=val_dataset.len(), callbacks=[tbcb], max_q_size=batch_size*2)
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
            images.append(np.reshape(sample.get_image(), 4096))
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
        with open('model_one_results.json', 'w') as f:
            json.dump(results_dict, f)
        print('Results saved')

    def predict(self):
        pass

class ModelTwo(DFModel):
    def __init__(self, vocabulary_size=10000, question_max_len=22, model_path='model_2_test_batchnorm.p'):
        self.vqa_model = None
        self.EMBED_HIDDEN_SIZE = 100
        self.vocabulary_size = vocabulary_size
        self.question_max_len = question_max_len
        if not os.path.isdir(config.MODELS_PATH):
            os.mkdir(config.MODELS_PATH)
        self.MODEL_PATH = os.path.join(config.MODELS_PATH, model_path)
        self.weights_path = os.path.join(config.MODELS_PATH, 'weights_m2_batchnorm.h5')
        self.build()

    def build(self):
        # Params
        lstm_hidden_units = 256
        # Optimizer
        adam = Adam(lr=1e-4)
        # Create/load model
        embedding_matrix = h5py.File(os.path.join(config.DATA_PATH, 'glove/embedding_matrix.h5'))['matrix']
        try:
            with open(self.MODEL_PATH, 'r') as f:
                print('Loading baseline model...')
                self.vqa_model = model_from_json(f.read())
                print('Baseline model loaded')
                print('Compiling baseline model...')
                self.vqa_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
                print('Model compiled')
        except IOError:
            #Image:
            image_input = Input(shape=(25088, ))
            image_features = Dense(output_dim=lstm_hidden_units, activation='relu')(image_input)
            # Question
            question_input = Input(shape=(self.question_max_len,), dtype='int32')
            question_embedded = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                                          input_length=self.question_max_len)(question_input)
            question_embedded = Dropout(0.5)(question_embedded)
            sentence_embedded = LSTM(lstm_hidden_units, return_sequences=False)(question_embedded)
            sentence_embedded = Dropout(0.5)(sentence_embedded)

            # Merge
            merged = merge([image_features, sentence_embedded], mode='sum')  # Merge for layers, merge for tensors
            output = Dense(output_dim=self.vocabulary_size, activation='softmax')(merged)

            self.vqa_model = Model(input=[image_input, question_input], output=output)
            print('Baseline model created')
            plot(self.vqa_model, show_shapes=True, to_file='model_2_batchnorm.png', show_layer_names=False)
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
        checkpoint = ModelCheckpoint(self.weights_path, monitor='val_loss', verbose=0, save_best_only=True,
                                     save_weights_only=True, mode='min', period=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
        self.vqa_model.fit_generator(ThreadSafeIter(train_dataset.generator(batch_size)), samples_per_epoch=train_dataset.len(),
                                     nb_epoch=epochs, validation_data=ThreadSafeIter(val_dataset.generator(batch_size)),
                                     nb_val_samples=val_dataset.len(), callbacks=[tbcb, checkpoint, reduce_lr], max_q_size=batch_size * 2,
                                     nb_worker=10)

    def validate(self, dataset, weights_path, batch_size):
        print('Loading weights...')
        self.vqa_model.load_weights(weights_path)
        print('Weights loaded')
        print('Start validation...')
        result = self.vqa_model.evaluate_generator(dataset.generator(batch_size), val_samples=dataset.len())
        print('Validated. Loss: {}'.format(result))
        return result

    def test(self, dataset, weights_path, batch_size, type='test'):
        print('Loading weights...')
        self.vqa_model.load_weights(weights_path)
        print('Weights loaded')
        print('Predicting...')
        image_data = h5py.File(os.path.join(config.DATA_PATH, 'test-dev_dataset_image_features_vgg16.h5'), 'r')
        features = image_data['image_features']
        results_dict=[]
        print('Building reverse word dictionary...')
        word_dict = {idx: word for word, idx in dataset.tokenizer.word_index.iteritems()}
        print('Reverse dictionary build')

        for sample in dataset.get_samples():
            image=np.reshape(features[sample._image.features_idx], (1, 25088))
            question=np.reshape(sample.get_question(), (1, 22))
            results = self.vqa_model.predict([np.array(image), np.array(question)], batch_size)
            results = np.argmax(results, axis=1)  # Max index evaluated on rows (1 row = 1 sample)
            results_dict.append({'answer': word_dict[int(results)], 'question_id': sample._question.id})

        print('Answers predicted')
        print 'Saving...'
        with open(type + '_model_three_new_results_sum_redlr.json', 'w') as f:
            json.dump(results_dict, f)
        print('Results saved')

    def predict(self):
        pass


class ModelThree(DFModel):
    def __init__(self, vocabulary_size=10000, question_max_len=22, model_path='model_3_sum_batchnorm.p'):
        self.vqa_model = None
        self.EMBED_HIDDEN_SIZE = 100
        self.vocabulary_size = vocabulary_size
        self.question_max_len = question_max_len
        if not os.path.isdir(config.MODELS_PATH):
            os.mkdir(config.MODELS_PATH)
        self.MODEL_PATH = os.path.join(config.MODELS_PATH, model_path)
        self.weights_path = os.path.join(config.MODELS_PATH, 'weights_m3_sum_batchnorm.h5')
        self.build()
        #

    def build(self):
        # Params
        lstm_hidden_units = 256
        # Optimizer
        adam = Adam(lr=1e-4)
        # Create/load model
        embedding_matrix = h5py.File(os.path.join(config.DATA_PATH, 'glove/embedding_matrix.h5'))['matrix']
        try:
            with open(self.MODEL_PATH, 'r') as f:
                print('Loading baseline model...')
                self.vqa_model = model_from_json(f.read())
                print('Baseline model loaded')
                print('Compiling baseline model...')
                self.vqa_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
                print('Model compiled')
        except IOError:
            #Image:
            image_input = Input(shape=(1, 1, 1, 2048, ))
            image_features = Flatten()(image_input)
            image_features = Dense(output_dim=lstm_hidden_units, activation='relu')(image_features)
            # Question
            question_input = Input(shape=(self.question_max_len, ), dtype='int32')
            question_embedded = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                                          input_length=self.question_max_len)(question_input)
            question_embedded = Dropout(0.5)(question_embedded)
            sentence_embedded = LSTM(lstm_hidden_units, return_sequences=False, dropout_W=0.5, dropout_U=0.5)(question_embedded)

            # Merge
            merged = merge([image_features, sentence_embedded], mode='sum')  # Merge for layers, merge for tensors
            merged = BatchNormalization()(merged)
            output = Dense(output_dim=self.vocabulary_size, activation='softmax')(merged)

            self.vqa_model = Model(input=[image_input, question_input], output=output)
            print('Baseline model created')
            plot(self.vqa_model, show_shapes=True, to_file='model_3_sum.png', show_layer_names=False)
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
        checkpoint = ModelCheckpoint(self.weights_path, monitor='val_loss', verbose=0, save_best_only=True,
                                        save_weights_only=True, mode='min', period=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
        self.vqa_model.fit_generator(ThreadSafeIter(train_dataset.generator(batch_size)), samples_per_epoch=train_dataset.len(),
                                     nb_epoch=epochs, validation_data=ThreadSafeIter(val_dataset.generator(batch_size)),
                                     nb_val_samples=val_dataset.len(), callbacks=[tbcb, checkpoint, reduce_lr], max_q_size=batch_size * 2,
                                     nb_worker=10)

    def validate(self, dataset, weights_path, batch_size):
        print('Loading weights...')
        self.vqa_model.load_weights(weights_path)
        print('Weights loaded')
        print('Start validation...')
        result = self.vqa_model.evaluate_generator(dataset.generator(batch_size), val_samples=dataset.len())
        print('Validated. Loss: {}'.format(result))
        return result

    def test(self, dataset, weights_path, batch_size, type='test-dev'):
        print('Loading weights...')
        self.vqa_model.load_weights(weights_path)
        print('Weights loaded')
        print('Predicting...')
        image_data = h5py.File(os.path.join(config.DATA_PATH, type + '_dataset_image_features_resnet.h5'), 'r')
        features = image_data['image_features']
        images = []
        questions = []
        for sample in dataset.get_samples():
            images.append(np.reshape(features[sample._image.features_idx], (1, 1, 1, 2048)))
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
        with open(type + '_model_three_new_results.json', 'w') as f:
            json.dump(results_dict, f)
        print('Results saved')

    def predict(self):
        pass

class ModelFour(DFModel):
    def __init__(self, vocabulary_size=10000, question_max_len=22, model_path='language_only_gloveft.p'):
        self.vqa_model = None
        self.EMBED_HIDDEN_SIZE = 100
        self.vocabulary_size = vocabulary_size
        self.question_max_len = question_max_len
        if not os.path.isdir(config.MODELS_PATH):
            os.mkdir(config.MODELS_PATH)
        self.MODEL_PATH = os.path.join(config.MODELS_PATH, model_path)
        self.weights_path = os.path.join(config.MODELS_PATH, 'language_only_gloveft.h5')
        self.build()

    def build(self):
        # Params
        lstm_hidden_units = 256
        # Optimizer
        adam = Adam(lr=1e-4)
        # Create/load model
        embedding_matrix = h5py.File(os.path.join(config.DATA_PATH, 'glove/embedding_matrix.h5'))['matrix']

        try:
            with open(self.MODEL_PATH, 'r') as f:
                print('Loading language model...')
                self.vqa_model = model_from_json(f.read())
                print('Language model loaded')
                print('Compiling language model...')
                self.vqa_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
                print('Model compiled')
        except IOError:
            # Question
            question_input = Input(shape=(self.question_max_len, ), dtype='int32')
            question_embedded = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                                          input_length=self.question_max_len)(question_input)
            question_embedded = Dropout(0.5)(question_embedded)
            sentence_embedded = LSTM(lstm_hidden_units, return_sequences=False, dropout_W=0.5, dropout_U=0.5)(question_embedded)
            sentence_embedded = Dropout(0.5)(sentence_embedded)
            output = Dense(output_dim=self.vocabulary_size, activation='softmax')(sentence_embedded)

            self.vqa_model = Model(input=question_input, output=output)
            print('Language model created')
            plot(self.vqa_model, show_shapes=True, to_file='language_only.png', show_layer_names=False)
            print('Compiling language model...')
            self.vqa_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
            print('Baseline model compiled')

            print('Saving language model...')
            model_json = self.vqa_model.to_json()
            with open(self.MODEL_PATH, 'w') as f:
                f.write(model_json)
            print('Language model saved')

    def train(self, train_dataset, val_dataset, batch_size, epochs):
        tbcb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
        checkpoint = ModelCheckpoint(self.weights_path, monitor='val_loss', verbose=0, save_best_only=True,
                                        save_weights_only=True, mode='min', period=1)
        self.vqa_model.fit_generator(ThreadSafeIter(train_dataset.generator(batch_size)), samples_per_epoch=train_dataset.len(),
                                     nb_epoch=epochs, validation_data=ThreadSafeIter(val_dataset.generator(batch_size)),
                                     nb_val_samples=val_dataset.len(), callbacks=[tbcb, checkpoint], max_q_size=batch_size * 2,
                                     nb_worker=10)

    def validate(self, dataset, weights_path, batch_size):
        print('Loading weights...')
        self.vqa_model.load_weights(weights_path)
        print('Weights loaded')
        print('Start validation...')
        result = self.vqa_model.evaluate_generator(dataset.generator(batch_size), val_samples=dataset.len())
        print('Validated. Loss: {}'.format(result))
        return result

    def test(self, dataset, weights_path, batch_size, type='test-dev'):
        print('Loading weights...')
        self.vqa_model.load_weights(weights_path)
        print('Weights loaded')
        print('Predicting...')

        images = []
        questions = []
        for sample in dataset.get_samples():
            questions.append(sample.get_question())
        results = self.vqa_model.predict([np.array(questions)], batch_size)
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
        with open(type + '_langonlygloveft.json', 'w') as f:
            json.dump(results_dict, f)
        print('Results saved')

    def predict(self):
        pass

class ModelFive(DFModel):
    def __init__(self, vocabulary_size=10000, question_max_len=22, model_path='model_5_concat_nodense.p'):
        self.vqa_model = None
        self.EMBED_HIDDEN_SIZE = 100
        self.vocabulary_size = vocabulary_size
        self.question_max_len = question_max_len
        if not os.path.isdir(config.MODELS_PATH):
            os.mkdir(config.MODELS_PATH)
        self.MODEL_PATH = os.path.join(config.MODELS_PATH, model_path)
        self.weights_path = os.path.join(config.MODELS_PATH, 'weights_m5_concat_nodense.h5')
        self.build()

    def build(self):
        # Params
        lstm_hidden_units = 256
        # Optimizer
        adam = Adam(lr=1e-4)
        # Create/load model
        embedding_matrix = h5py.File(os.path.join(config.DATA_PATH, 'glove/embedding_matrix.h5'))['matrix']
        try:
            with open(self.MODEL_PATH, 'r') as f:
                print('Loading baseline model...')
                self.vqa_model = model_from_json(f.read())
                print('Baseline model loaded')
                print('Compiling baseline model...')
                self.vqa_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
                print('Model compiled')
        except IOError:
            #Image:
            image_input = Input(shape=(1, 1, 1, 2048, ))
            image_features = Flatten()(image_input)
            image_features = Dropout(0.5)(image_features)
            # Question
            question_input = Input(shape=(self.question_max_len, ), dtype='int32')
            question_embedded = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                                          input_length=self.question_max_len)(question_input)
            question_embedded = Dropout(0.5)(question_embedded)
            sentence_embedded = LSTM(lstm_hidden_units, return_sequences=False, dropout_U=0.5, dropout_W=0.5)(question_embedded)
            sentence_embedded = BatchNormalization()(sentence_embedded)
            # Merge
            merged = merge([image_features, sentence_embedded], mode='concat')
            merged = Dense(output_dim=2304, activation='relu')(merged)
            merged = Dropout(0.25)(merged)
            output = Dense(output_dim=self.vocabulary_size, activation='softmax')(merged)

            self.vqa_model = Model(input=[image_input, question_input], output=output)
            print('Baseline model created')
            plot(self.vqa_model, show_shapes=True, to_file='model_5_concat_nodense.png', show_layer_names=False)
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
        checkpoint = ModelCheckpoint(self.weights_path, monitor='val_loss', verbose=0, save_best_only=True,
                                        save_weights_only=True, mode='min', period=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
        self.vqa_model.fit_generator(ThreadSafeIter(train_dataset.generator(batch_size)), samples_per_epoch=train_dataset.len(),
                                     nb_epoch=epochs, validation_data=ThreadSafeIter(val_dataset.generator(batch_size)),
                                     nb_val_samples=val_dataset.len(), callbacks=[tbcb, checkpoint, reduce_lr], max_q_size=batch_size * 2,
                                     nb_worker=10)

    def validate(self, dataset, weights_path, batch_size):
        print('Loading weights...')
        self.vqa_model.load_weights(weights_path)
        print('Weights loaded')
        print('Start validation...')
        result = self.vqa_model.evaluate_generator(dataset.generator(batch_size), val_samples=dataset.len())
        print('Validated. Loss: {}'.format(result))
        return result

    def test(self, dataset, weights_path, batch_size, type='test-dev'):
        print('Loading weights...')
        self.vqa_model.load_weights(weights_path)
        print('Weights loaded')
        print('Predicting...')
        image_data = h5py.File(os.path.join(config.DATA_PATH, type + '_dataset_image_features_resnet_2.h5'), 'r')
        features = image_data['image_features']
        images = []
        questions = []
        for sample in dataset.get_samples():
            images.append(np.reshape(features[sample._image.features_idx], (1, 1, 1, 2048)))
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
        with open(type + 'model_5_results.json', 'w') as f:
            json.dump(results_dict, f)
        print('Results saved')

    def predict(self):
        pass


class ThreadSafeIter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

