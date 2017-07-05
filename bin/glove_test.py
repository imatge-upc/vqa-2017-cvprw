import numpy as np
import cPickle as pickle
import h5py

from vqa import config
import os

def read_glove_vectors(filename, vocabulary):
    f = open(filename)
    print 'Reading glove vectors...'
    coefs = np.zeros((len(vocabulary) + 1, 100))
    for i, line in enumerate(f):
        values = line.split()
        word = values[0]
        if word in vocabulary:
            idx = vocabulary.index(word)
            coefs[idx+1] = values[1:]
    f.close()
    coefsm = np.asarray(coefs, dtype='float32')
    print 'Saving embedding matrix'
    with h5py.File(os.path.join(config.DATA_PATH, 'glove/embedding_matrix.h5')) as f:
        f.create_dataset('matrix', data=coefsm)
        f.close()
    return coefsm

def get_tokenizer(tokenizer_path):
    tokenizer = None
    if os.path.isfile(tokenizer_path):
        print 'Loading tokenizer'
        tokenizer = pickle.load(open(tokenizer_path, 'r'))
        print 'Finish loading tokenizer'
    return tokenizer



tokenizer = get_tokenizer(os.path.join(config.DATA_PATH, 'tokenizer.p'))
words = []
vocab = tokenizer.word_index.items()
vocab.sort(key=lambda x:x[1])
for values in vocab:
    words.append(values[0])
matrix = read_glove_vectors(os.path.join(config.DATA_PATH, 'glove/glove.6B.100d.txt'), words)
print matrix.shape[0]
print matrix.shape[1]