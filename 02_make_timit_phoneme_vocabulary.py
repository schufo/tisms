import pickle
import os

import timit_utils as tu
import numpy as np


timit_corpus = tu.Corpus('../Datasets/TIMIT/TIMIT/TIMIT')
timit_training_set = timit_corpus.train


def get_timit_train_sentence(idx):
    person_idx = int(np.floor(idx / 10))
    person = timit_training_set.person_by_index(person_idx)
    sentence_idx = idx % 10
    sentence = person.sentence_by_index(sentence_idx)
    audio = sentence.raw_audio
    phonemes = sentence.phones_df.index.values
    return audio, phonemes


def make_timit_vocabulary(path_to_save_files):

    # 0: <pad> (padding token), 1: <S> (silence token), 2: <N> (noise token, indicates noise in clean speech recordings)
    vocabulary = ['<pad>', '<S>', '<N>']

    for idx in range(4620):

        audio, phonemes = get_timit_train_sentence(idx)

        for token in phonemes:
            if token not in vocabulary:
                vocabulary.append(token)

    # dictionary to translate between token and index representation of phonemes
    phoneme2idx = {p: int(idx) for (idx, p) in enumerate(vocabulary)}
    idx2phoneme = {idx: p for (idx, p) in enumerate(vocabulary)}

    pickle_out = open(os.path.join(path_to_save_files, "timit_vocabulary.pickle"), "wb")
    pickle.dump(vocabulary, pickle_out)
    pickle_out.close()

    pickle_out = open(os.path.join(path_to_save_files, "phoneme2idx.pickle"), "wb")
    pickle.dump(phoneme2idx, pickle_out)
    pickle_out.close()

    pickle_out = open(os.path.join(path_to_save_files, "idx2phoneme.pickle"), "wb")
    pickle.dump(idx2phoneme, pickle_out)
    pickle_out.close()

    print('Vocabulary: ', vocabulary)
    print('Vocabulary size: ', len(vocabulary))
    print('phoneme2idx: ', phoneme2idx)
    print('idx2phoneme: ', idx2phoneme)


if __name__ == '__main__':

    path_to_save_files = 'data/'

    make_timit_vocabulary(path_to_save_files)



