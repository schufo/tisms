import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tensorboardX import SummaryWriter
from sacred import Experiment
from sacred.observers import FileStorageObserver

import utils.data_set_utls as utls
from utils import fct
from utils import build_models


ex = Experiment('tisms')
ex.observers.append(FileStorageObserver.create('sacred_experiment_logs'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

torch.manual_seed(0)
torch.cuda.manual_seed(0)


@ex.config
def configuration():

    tag = 'OA'
    model = 'InformedSeparatorWithPerfectAttention'
    data_set = 'timit_musdb'

    seed = 1

    snr_train = 'random'
    snr_val = -5
    fft_len = 512
    hop_len = 256
    window = 'hamming'

    loss_function = 'L1'
    batch_size_train = 32
    batch_size_val = 40
    epochs = 3000
    lr_switch1 = 0
    lr_switch2 = 200

    text_feature_size = 63
    vocabulary_size = 63
    mix_encoder_layers = 2
    side_info_encoder_layers = 1
    side_info_encoder_bidirectional = True
    target_decoder_layers = 2

    perfect_alphas = True

    optimizer_name = 'Adam'
    learning_rate = 0.0001
    weight_decay = 0

    models_directory = 'trained_models'

    comment = 'optimal attention weights'


@ex.capture
def make_model(model, fft_len, text_feature_size, mix_encoder_layers,
               side_info_encoder_layers, target_decoder_layers, side_info_encoder_bidirectional=True):

    mix_features_size = int(fft_len / 2 + 1)

    if model == 'InformedSeparatorWithAttention':
        separator = build_models.make_informed_separator_with_attention(mix_features_size,
                                                                        text_feature_size,
                                                                        mix_encoder_layers,
                                                                        side_info_encoder_layers,
                                                                        target_decoder_layers,
                                                                        side_info_encoder_bidirectional)

    elif model == 'InformedSeparatorWithPerfectAttention':
        separator = build_models.make_informed_separator_with_perfect_attention(mix_features_size,
                                                                                text_feature_size,
                                                                                mix_encoder_layers,
                                                                                side_info_encoder_layers,
                                                                                target_decoder_layers,
                                                                                side_info_encoder_bidirectional)

    return separator


@ex.capture
def experiment_tag(tag):
    return tag


@ex.capture
def make_experiment_dir(models_directory, tag):
    experiment_dir = os.path.join(models_directory, tag)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    return experiment_dir


@ex.capture
def make_data_sets(data_set, snr_train, snr_val, fft_len, hop_len, window, batch_size_train, batch_size_val, perfect_alphas):

    if data_set == 'timit_musdb':
        import data.timit_musdb_train as train_set
        import data.timit_musdb_val as val_set

        if perfect_alphas:
            timit_musdb_train = train_set.Train(transform=transforms.Compose([utls.MixSNR(snr_train),
                                                                              utls.StftOnFly(fft_len=fft_len,
                                                                                             hop_len=hop_len,
                                                                                             window=window),
                                                                              utls.NormalizeWithOwnMax(),
                                                                              utls.MakePerfectAttentionWeights(fft_len, hop_len)]))

            timit_musdb_val = val_set.Val(transform=transforms.Compose([utls.MixSNR(snr_val),
                                                                        utls.StftOnFly(fft_len=fft_len, hop_len=hop_len,
                                                                                       window=window),
                                                                        utls.NormalizeWithOwnMax(),
                                                                        utls.MakePerfectAttentionWeights(fft_len, hop_len)]))


        else:
            timit_musdb_train = train_set.Train(transform=transforms.Compose([utls.MixSNR(snr_train),
                                                                          utls.StftOnFly(fft_len=fft_len,
                                                                                         hop_len=hop_len,
                                                                                         window=window),
                                                                          utls.NormalizeWithOwnMax()]))

            timit_musdb_val = val_set.Val(transform=transforms.Compose([utls.MixSNR(snr_val),
                                                                    utls.StftOnFly(fft_len=fft_len, hop_len=hop_len,
                                                                                   window=window),
                                                                    utls.NormalizeWithOwnMax()]))


    dataloader_train = DataLoader(timit_musdb_train, batch_size=batch_size_train, shuffle=True, num_workers=4,
                                  worker_init_fn=utls.worker_init_fn, collate_fn=utls.collate_with_phonemes)

    dataloader_val = DataLoader(timit_musdb_val, batch_size=batch_size_val, shuffle=True, num_workers=4,
                                worker_init_fn=utls.worker_init_fn, collate_fn=utls.collate_with_phonemes)

    return dataloader_train, dataloader_val


@ex.capture
def idx2onehot(phonemes_batched, vocabulary_size):

    """

    Parameters
    ----------
    phonemes_batched: sequence of phoneme indices, shape: (batch size, sequence length)
    vocabulary_size: int

    Returns
    -------

    """
    batch_of_one_hot_sentences = []
    for sentence_idx in range(phonemes_batched.size()[0]):
        sentence = phonemes_batched[sentence_idx, :]
        sentence_one_hot_encoded = utls.idx2one_hot(sentence, vocabulary_size)
        batch_of_one_hot_sentences.append(sentence_one_hot_encoded)
    batch_of_one_hot_sentences = torch.from_numpy(np.asarray(batch_of_one_hot_sentences)).type(torch.float32)
    return batch_of_one_hot_sentences


@ex.capture
def make_optimizer(model_to_train, learning_rate, weight_decay):
    optimizer = torch.optim.Adam(model_to_train.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


@ex.capture
def config2main(epochs, lr_switch1, lr_switch2, loss_function):
    return epochs, lr_switch1, lr_switch2, loss_function


@ex.automain
def train_model():

    epochs, lr_switch1, lr_switch2, loss_function = config2main()

    tag = experiment_tag()

    writer = SummaryWriter(logdir=os.path.join('tensorboard', tag))

    experiment_dir = make_experiment_dir()

    dataloader_train, dataloader_val = make_data_sets()

    model_to_train = make_model()

    optimizer = make_optimizer(model_to_train)

    def factor_fn(epoch):
        if epoch < lr_switch1:
            factor = 0.01
        elif epoch < lr_switch2:
            factor = 0.1
        else:
            factor = 1
        return factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor_fn, last_epoch=-1)

    loss_fn = nn.L1Loss(reduction='sum')

    best_val_cost = 1000000000

    counter = 0

    for i in range(epochs):
        cost = 0.
        val_cost = 0.
        np.random.seed(1)

        # training loop
        for i_batch, sample_batched in enumerate(dataloader_train):

            phonemes_idx = sample_batched['phonemes']

            alphas = sample_batched['perfect_alphas']

            # output has shape (batch_size, sequence_len, vocabulary_size)
            one_hot_phonemes = idx2onehot(phonemes_idx)

            batch_cost = fct.train_with_perfect_attention(model_to_train.to(device), loss_fn, optimizer,
                                             sample_batched['mix'].to(device), one_hot_phonemes.to(device),
                                             sample_batched['target'].to(device), alphas.to(device))

            cost += batch_cost

        writer.add_scalar('Train_cost', cost, i + 1)

        # validation loop
        for i_batchV, sample_batchedV in enumerate(dataloader_val):

            phonemes_idxV = sample_batchedV['phonemes']

            # output has shape (batch_size, sequence_len, vocabulary_size)
            one_hot_phonemesV = idx2onehot(phonemes_idxV)

            with torch.no_grad():

                prediction, alphas = fct.predict_with_perfect_attention(model_to_train.to(device),
                                                                sample_batchedV['mix'].to(device),
                                                                one_hot_phonemesV.to(device),
                                                                sample_batchedV['perfect_alphas'].to(device))

                val_loss = loss_fn(prediction, sample_batchedV['target'].to(device))

            val_cost += val_loss.item()

        print("Epoch: {}, Training cost: {} Validation cost: {}".format(i + 1, cost, val_cost))
        writer.add_scalar('Validation_cost', val_cost, i + 1)

        scheduler.step()

        if i+1 % 100 == 0:
            torch.save({
                'experiment_tag': tag,
                'epoch': i,
                'model_state_dict': model_to_train.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_cost': cost,
            }, os.path.join(experiment_dir, 'model_epoch_{}.pt'.format(i+1)))

        if val_cost < best_val_cost:
            best_val_cost = val_cost
            counter = 0

            print('Epoch: ', i + 1, 'val cost: ', best_val_cost)

            torch.save({
                'experiment_tag': tag,
                'epoch': i,
                'model_state_dict': model_to_train.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_cost': cost,
            }, os.path.join(experiment_dir, 'model_best_val_cost.pt'))

            print("Model has been saved!")

        elif val_cost > best_val_cost:
            counter += 1

        if counter > 200:

            print("No improvement of validation cost for {} epochs".format(counter))

            break

    torch.save({
        'experiment_tag': tag,
        'epoch': i,
        'model_state_dict': model_to_train.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_cost': cost,
    }, os.path.join(experiment_dir, 'model_last_train_epoch.pt'))

