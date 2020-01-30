import os
import json

import torch
from torchvision import transforms
import numpy as np
from sacred import Experiment

import utils.data_set_utls as utls
from utils import build_models
from utils import fct

ex = Experiment('eval_alignment')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

torch.manual_seed(0)
torch.cuda.manual_seed(0)

@ex.config
def configuration():

    tag = 'test'
    seed = 1
    model_state_dict_name = 'model_best_val_cost.pt'

    side_info_type = 'phonemes'  # 'phonemes' or 'ones'
    data_set = 'timit_musdb'

    text_feature_size = 63
    vocabulary_size = 63
    phoneme_embedding_size = 257
    mix_encoder_layers = 2
    side_info_encoder_layers = 2
    target_decoder_layers = 2
    align_phonemes = False
    side_info_encoder_bidirectional = True

    snr_train = 'random'
    eval_dir = 'evaluation'

    ex.add_config('configs/{}/config.json'.format(tag))
    test_snr = -5


@ex.capture
def make_data_set(data_set, test_snr, fft_len, hop_len, window):

    if data_set == 'timit_musdb':
        import data.timit_musdb_test as test_set

        timit_musdb_test = test_set.Test(transform=transforms.Compose([utls.MixSNR(test_snr),
                                                                       utls.StftOnFly_testset(fft_len=fft_len,
                                                                                              hop_len=hop_len,
                                                                                              window=window),
                                                                       utls.NormalizeWithOwnMax()]))

    return timit_musdb_test


@ex.capture
def make_model(model, fft_len, text_feature_size, mix_encoder_layers, side_info_encoder_layers, target_decoder_layers,
               side_info_encoder_bidirectional):

    mix_features_size = int(fft_len / 2 + 1)

    if model == 'InformedSeparatorWithAttention':
        separator = build_models.make_informed_separator_with_attention(mix_features_size,
                                                                        text_feature_size,
                                                                        mix_encoder_layers,
                                                                        side_info_encoder_layers,
                                                                        target_decoder_layers,
                                                                        side_info_encoder_bidirectional)

    elif model == 'InformedSeparatorWithSplitAttention':
        separator = build_models.make_informed_separator_with_split_attention(mix_features_size,
                                                                              text_feature_size,
                                                                              mix_encoder_layers,
                                                                              side_info_encoder_layers,
                                                                              target_decoder_layers,
                                                                              side_info_encoder_bidirectional)

    return separator


@ex.capture
def load_state_dict(models_directory, tag, model_state_dict_name):
    checkpoint = torch.load(os.path.join(models_directory, tag, model_state_dict_name))
    return checkpoint['model_state_dict'], checkpoint['experiment_tag']


@ex.capture
def config2main(tag, test_snr, fft_len, hop_len, window, text_feature_size, side_info_type, vocabulary_size):

    return tag, test_snr, fft_len, hop_len, window, text_feature_size, side_info_type, vocabulary_size


@ex.capture
def make_eval_dir(eval_dir, tag, test_snr):
    model_eval_dir = os.path.join(eval_dir, tag + "_snr{}".format(test_snr))
    if not os.path.exists(model_eval_dir):
        os.mkdir(model_eval_dir)
    return model_eval_dir


@ex.automain
def eval_model():

    tag, test_snr, fft_len, hop_len, window, text_feature_size, side_info_type, vocabulary_size = config2main()

    test_set = make_data_set()

    model_to_evaluate = make_model()

    state_dict, training_tag = load_state_dict()

    model_to_evaluate.load_state_dict(state_dict)

    model_to_evaluate.to(device)

    ae_all_snippets = []  # alignment error

    num_predicted_phonemes = 0
    num_phonemes_in_10ms_window = 0
    num_phonemes_in_20ms_window = 0
    num_phonemes_in_25ms_window = 0
    num_phonemes_in_50ms_window = 0
    num_phonemes_in_75ms_window = 0
    num_phonemes_in_100ms_window = 0
    num_phonemes_in_200ms_window = 0

    for i in range(len(test_set)):
        sample = test_set[i]

        mix_spec = sample['mix']
        phoneme_times = sample['phoneme_times']  # start of phonemes, last number=last phoneme's end (rel. speech_start)
        speech_start = sample['speech_start']  # start of the speech recording in the mix (!= first phoneme start)

        phonemes_idx = torch.from_numpy(sample['phonemes'])

        # output has shape (batch_size, sequence_len, vocabulary_size)
        side_info = torch.from_numpy(utls.idx2one_hot(phonemes_idx, vocabulary_size), ).type(torch.float32)
        side_info = torch.unsqueeze(side_info, dim=0)

        with torch.no_grad():
            predicted_speech_spec, alphas = fct.predict_with_attention(model_to_evaluate,
                                                                   torch.unsqueeze(torch.from_numpy(mix_spec), dim=0).to(device),
                                                                   side_info=side_info.to(device))

        alphas = alphas.detach().cpu().numpy()[0, :, :].T

        phoneme_idx_sequence, phoneme_onsets = fct.viterbi_alignment_from_attention(alphas, hop_len)
        phoneme_onsets = np.asarray(phoneme_onsets) + hop_len/32000
        phoneme_onsets_truth = np.asarray([(x+speech_start)/16000 for x in phoneme_times][:-1])  # delete end time of last phoneme

        number_of_phonemes = len(phoneme_onsets_truth)
        absolute_errors = abs(phoneme_onsets_truth - phoneme_onsets)
        absolute_error_snippet = np.mean(abs(phoneme_onsets_truth - phoneme_onsets))

        # ae_all_snippets.append(absolute_error_snippet)
        ae_all_snippets.append(absolute_error_snippet)

        # compute % correct phonemes within a tolerance
        correct_phonemes_in_10ms_window = (absolute_errors < 0.01).sum()
        num_phonemes_in_10ms_window += correct_phonemes_in_10ms_window
        correct_phonemes_in_20ms_window = (absolute_errors < 0.02).sum()
        num_phonemes_in_20ms_window += correct_phonemes_in_20ms_window
        correct_phonemes_in_25ms_window = (absolute_errors < 0.025).sum()
        num_phonemes_in_25ms_window += correct_phonemes_in_25ms_window
        correct_phonemes_in_50ms_window = (absolute_errors < 0.05).sum()
        num_phonemes_in_50ms_window += correct_phonemes_in_50ms_window
        correct_phonemes_in_75ms_window = (absolute_errors < 0.075).sum()
        num_phonemes_in_75ms_window += correct_phonemes_in_75ms_window
        correct_phonemes_in_100ms_window = (absolute_errors < 0.1).sum()
        num_phonemes_in_100ms_window += correct_phonemes_in_100ms_window
        correct_phonemes_in_200ms_window = (absolute_errors < 0.2).sum()
        num_phonemes_in_200ms_window += correct_phonemes_in_200ms_window

        num_predicted_phonemes += number_of_phonemes

    mean_abs_error_mean = np.mean(np.asarray(ae_all_snippets))
    mean_abs_error_median = np.median(np.asarray(ae_all_snippets))

    percent_correct_in_10ms_tolerance = num_phonemes_in_10ms_window / num_predicted_phonemes
    percent_correct_in_20ms_tolerance = num_phonemes_in_20ms_window / num_predicted_phonemes
    percent_correct_in_25ms_tolerance = num_phonemes_in_25ms_window / num_predicted_phonemes
    percent_correct_in_50ms_tolerance = num_phonemes_in_50ms_window / num_predicted_phonemes
    percent_correct_in_75ms_tolerance = num_phonemes_in_75ms_window / num_predicted_phonemes
    percent_correct_in_100ms_tolerance = num_phonemes_in_100ms_window / num_predicted_phonemes
    percent_correct_in_200ms_tolerance = num_phonemes_in_200ms_window / num_predicted_phonemes

    print("mean absolute error over test set: ", mean_abs_error_mean)
    print("median absolute error over test set: ", mean_abs_error_median)

    model_eval_dir = make_eval_dir()

    np.save(os.path.join(model_eval_dir, 'mean_abs_error_all_test_examples_snr{}.npy'.format(test_snr)),
            ae_all_snippets)

    eval_align_summary_dict = {'tag': tag, 'test_snr': test_snr, 'fft_len': fft_len,
                               'hop_len': hop_len, 'mean_abs_error_mean': mean_abs_error_mean,
                               'mean_abs_error_median': mean_abs_error_median,
                               'in_10ms_tol': percent_correct_in_10ms_tolerance,
                               'in_20ms_tol': percent_correct_in_20ms_tolerance,
                               'in_25ms_tol': percent_correct_in_25ms_tolerance,
                               'in_50ms_tol': percent_correct_in_50ms_tolerance,
                               'in_75ms_tol': percent_correct_in_75ms_tolerance,
                               'in_100ms_tol': percent_correct_in_100ms_tolerance,
                               'in_200ms_tol': percent_correct_in_200ms_tolerance
                               }

    print(eval_align_summary_dict)

    with open(os.path.join(model_eval_dir, 'eval_align_summary_snr{}.json'.format(test_snr)), 'w') as outfile:
        json.dump(eval_align_summary_dict, outfile)
