"""helper.py

Author: Omer Waseem
Description: various helper functions, see specific functions for details
"""

import numpy as np


def one_hot_encode (entities):
    """One-hot encoding from 1-D vector
    
    arguments: entities
    returns: encoded_vectors
    """
    encoded_vectors = []
    # one-hot formatting: [PER LOC ORG MISC O]
    for ent in entities:
        if ent == 'PER':
            encoded_vectors.append([1, 0, 0, 0, 0])
        elif ent == 'LOC':
            encoded_vectors.append([0, 1, 0, 0, 0])
        elif ent == 'ORG':
            encoded_vectors.append([0, 0, 1, 0, 0])
        elif ent == 'MISC':
            encoded_vectors.append([0, 0, 0, 1, 0])
        else:
            encoded_vectors.append([0, 0, 0, 0, 1])
    return np.array(encoded_vectors)


def one_hot_decode(predictions):
    """Decode one-hot vectors
    
    arguments: predictions
    returns: decoded
    """
    
    decoded = []
    for pred in predictions:
        max_index = np.argmax(pred)
        if max_index == 0:
            decoded.append('PER')
        elif max_index == 1:
            decoded.append('LOC')
        elif max_index == 2:
            decoded.append('ORG')
        elif max_index == 3:
            decoded.append('MISC')
        elif max_index == 4:
            decoded.append('O')
    return decoded


def load_glove_dict (glove_file):
    """GloVe loader
    
    Loads GloVe features from file into a word dictionary
    
    arguments: glove_file
    returns: word_dict
    """
    
    word_dict = {}
    with open(glove_file, 'r') as f:
        for line in f:
            split = line.split()
            word = split[0]
            vector = np.array([float(v) for v in split[1:]])
            word_dict[word] = vector
    return word_dict


def get_glove_vector (g_dict, word):
    """GloVe feature vector for given word
    
    Returns the feature vector of a given word from a given dictionary
    If word is not found in dictionary returns same length vector with 0. as values
    
    arguments: g_dict, word
    returns: vector
    """
        
    try:
        vector = g_dict[word.lower()]
    except KeyError:
        vector_len = len(g_dict['test'])
        vector = np.array([0.]*vector_len)
    return vector


def accuracy (expected, predicted):
    """Simple accuracy calculation
    
    Print accuracy percentage
    
    arguments: expected, predicted
    """
    
    total = 0
    correct = 0
    for i in range(len(expected)):
        total += 1
        if (expected[i] == predicted[i]):
            correct += 1
    print('accuracy = %d / %d = %lf' % (correct, total, correct/total))


def entity_count (expected):
    """Entity counts
    
    Print counts of ORG, PER, LOC, MISC and O entities
    
    arguments: expected
    """
        
    n_org = 0
    n_per = 0
    n_loc = 0
    n_misc = 0
    n_o = 0
    
    for e in expected:
        if e == 'ORG':
            n_org = n_org + 1
        elif e == 'PER':
            n_per = n_per + 1
        elif e == 'LOC':
            n_loc = n_loc + 1
        elif e == 'MISC':
            n_misc = n_misc + 1
        elif e == 'O':
            n_o = n_o + 1
    
    print('ORG:', n_org)
    print('PER:', n_per)
    print('LOC:', n_loc)
    print('MISC:', n_misc)
    print('O:', n_o)
