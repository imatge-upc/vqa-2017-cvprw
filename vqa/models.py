import os
import numpy as np

from dframe.model.model import Model as DFModel

from keras.layers import Input, Embedding, merge, Dropout, Dense, BatchNormalization, LSTM
from keras.models import Model, model_from_json
from keras.optimizers import Adam

from vqa import config

class Model_0(DFModel):

    def __init__(self, vocabulary_size=20000, question_max_len=22, model_path='model_0.p'):
        self.EMBED_HIDDEN_SIZE = 100
        self.vocabulary_size = vocabulary_size
        self.vqa_model = None
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
                print('Loading model...')
                vqa_model = model_from_json(f.read())
                print('Model loaded')
                print('Compiling model...')
                vqa_model.compile(optimizer=adam, loss='categorical_crossentropy')
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

            vqa_model = Model(input=[image_input, question_input], output=output)
            print('Baseline model created')

            print('Compiling baseline model...')
            vqa_model.compile(optimizer=adam, loss='categorical_crossentropy')
            print('Baseline model compiled')

            print('Saving baseline model...')
            model_json = vqa_model.to_json()
            with open(self.MODEL_PATH, 'w') as f:
                f.write(model_json)
            print('Baseline model saved')
            self.vqa_model = vqa_model

    def train(self, dataset, batch_size, epochs):
        self.vqa_model.fit_generator(dataset.batch_generator(batch_size), samples_per_epoch=dataset.size(),
                            nb_epoch=epochs)
        self.vqa_model.save_weights(self.weights_path)


    def validate(self, dataset):
        pass

    def test(self, dataset):
        pass

    def predict(self, sample):
        pass

    def batch_generator(self, dataset, batch_size):
        num_samples = len(dataset)
        batch_start = 0
        batch_end = batch_size

        while True:
            # Initialize matrix
            I = np.zeros((batch_size, 1024), dtype=np.float16)
            Q = np.zeros((batch_size, self.question_max_len), dtype=np.int32)
            A = np.zeros((batch_size, self.vocab_size), dtype=np.bool_)
            # Assign each sample in the batch
            for idx, sample in enumerate(dataset):
                I[idx], Q[idx] = sample.get_input(self.question_max_len)
                A[idx] = sample.get_output()

        yield ([I, Q], A)

        # Update interval
        batch_start += batch_size
        # An epoch has finished
        if batch_start >= num_samples:
            batch_start = 0
            # Change the order so the model won't see the samples in the same order in the next epoch
            random.shuffle(self.samples)
        batch_end = batch_start + batch_size
        if batch_end > num_samples:
            batch_end = num_samples