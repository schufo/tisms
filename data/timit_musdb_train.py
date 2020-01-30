import os
import pickle

import timit_utils as tu
import numpy as np
import torch
from torch.utils.data import Dataset

timit_corpus = tu.Corpus('../Datasets/TIMIT/TIMIT/TIMIT')
path_to_processed_musdb = '../Datasets/MUSDB_accompaniments'

timit_training_set = timit_corpus.train

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def get_timit_train_sentence(idx):
    # the training set for this project comprises the first 4320 sentences of the TIMIT training partition
    # the persons are not sorted by dialect regions when accessed with .person_by_index, which ensures that all
    # dialect regions are represented in both the training and validation set
    person_idx = int(np.floor(idx / 10))
    person = timit_training_set.person_by_index(person_idx)
    sentence_idx = idx % 10
    sentence = person.sentence_by_index(sentence_idx)
    audio = sentence.raw_audio
    phonemes = sentence.phones_df.index.values
    words = sentence.words_df.index.values

    # the array 'phoneme_times' contains the start values of the phonemes.
    # The last number is the end value of the last phoneme !
    phoneme_times = sentence.phones_df['start'].values
    phoneme_times = np.append(phoneme_times, sentence.phones_df['end'].values[-1])

    return audio, phonemes, phoneme_times, words


class Train(Dataset):

    def __init__(self, transform=None):

        # timit related
        pickle_in = open('data/phoneme2idx.pickle', 'rb')
        self.phoneme2idx = pickle.load(pickle_in)

        # musdb related
        self.musdb_train_path = os.path.join(path_to_processed_musdb, 'train')
        # load pickle file made by pre-processing script
        pickle_in = open(os.path.join(path_to_processed_musdb, 'train/train_file_list.pickle'), 'rb')
        self.train_file_list = pickle.load(pickle_in)

        # make list of shuffled musdb indices to randomly assign a musdb frame to each timit utterance
        musdb_indices_1 = list(np.arange(0, 2259))
        musdb_indices_2 = list(np.arange(0, 2259))
        np.random.seed(1)
        np.random.shuffle(musdb_indices_1)
        np.random.shuffle(musdb_indices_2)
        self.musdb_shuffled_idx = []
        self.musdb_shuffled_idx.extend(musdb_indices_1)
        self.musdb_shuffled_idx.extend(musdb_indices_2)
        self.transform = transform

    def __len__(self):
        return 4320  # number of TIMIT utterances assigned to training set

    def __getitem__(self, idx):

        speech, phonemes, phoneme_times, words = get_timit_train_sentence(idx)
        musdb_accompaniment = np.load(os.path.join(self.musdb_train_path,
                                                   self.train_file_list[self.musdb_shuffled_idx[idx]]))

        # pad the speech signal to same length as music
        speech_len = len(speech)
        music_len = len(musdb_accompaniment)
        padding_at_start = np.random.randint(0, music_len - speech_len)
        padding_at_end = music_len - padding_at_start - speech_len
        speech_padded = np.pad(array=speech, pad_width=(padding_at_start, padding_at_end),
                               mode='constant', constant_values=0)

        phoneme_int = np.array([self.phoneme2idx[p] for p in phonemes])


        # add a silence token (idx=1) to start and end of phoneme sequence
        phoneme_int = np.pad(phoneme_int, (1, 1), mode='constant', constant_values=1)

        sample = {'speech': speech_padded, 'music': musdb_accompaniment, 'speech_start': padding_at_start,
                  'speech_len': speech_len, 'phonemes': phoneme_int, 'phoneme_times': phoneme_times,
                  'perfect_alphas': None}

        if self.transform:
            sample = self.transform(sample)

        return sample
