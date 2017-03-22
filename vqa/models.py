import os
import numpy as np

from dframe.model.model import Model as DFModel

from keras.layers import Input, Embedding, merge, Dropout, Dense, LSTM
from keras.models import Model, model_from_json
from keras.optimizers import Adam

from vqa import config


class Model0(DFModel):

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
                print('Model baseline loaded')
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

            print('Compiling baseline model...')
            self.vqa_model.compile(optimizer=adam, loss='categorical_crossentropy')
            print('Baseline model compiled')

            print('Saving baseline model...')
            model_json = self.vqa_model.to_json()
            with open(self.MODEL_PATH, 'w') as f:
                f.write(model_json)
            print('Baseline model saved')

    def train(self, train_dataset, val_dataset, batch_size, epochs):
        self.vqa_model.fit_generator(train_dataset.batch_generator(batch_size), samples_per_epoch=train_dataset.len(),
                            nb_epoch=epochs, validation_data=val_dataset.batch_generator(batch_size)
                                     , nb_val_samples=val_dataset.len())
        self.vqa_model.save_weights(self.weights_path)

    def validate(self, dataset):
        pass

    def test(self, dataset):
        pass

    def predict(self, sample):
        pass
