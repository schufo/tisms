import os
import json

import torch
from torchvision import transforms
import numpy as np
import librosa as lb
from sacred import Experiment

import mir_eval as me
from pystoi.stoi import stoi as eval_stoi
from pypesq import pesq as eval_pesq

import utils.data_set_utls as utls
from utils import build_models
from utils import fct

ex = Experiment('eval_separation')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

torch.manual_seed(0)
torch.cuda.manual_seed(0)


@ex.config
def configuration():

    tag = 'tag to evaluate'
    seed = 1
    model_state_dict_name = 'model_best_val_cost.pt'

    side_info_type = 'phonemes'  # 'phonemes' or 'ones'
    test_data_set = 'timit_musdb'

    side_info_encoder_bidirectional = True

    perfect_alphas = False
    vocabulary_size = None
    eval_dir = 'evaluation'

    ex.add_config('configs/{}/config.json'.format(tag))

    test_snr = -5


@ex.capture
def make_data_set(test_data_set, test_snr, fft_len, hop_len, window, perfect_alphas):

    if test_data_set == 'timit_musdb':
        import data.timit_musdb_test as test_set

        if perfect_alphas:
            timit_musdb_test = test_set.Test(transform=transforms.Compose([utls.MixSNR(test_snr),
                                                                           utls.StftOnFly_testset(fft_len=fft_len,
                                                                                                  hop_len=hop_len,
                                                                                                  window=window),
                                                                           utls.NormalizeWithOwnMax(),
                                                                           utls.MakePerfectAttentionWeights(fft_len,
                                                                                                            hop_len)]))

        else:
            timit_musdb_test = test_set.Test(transform=transforms.Compose([utls.MixSNR(test_snr),
                                                                           utls.StftOnFly_testset(fft_len=fft_len,
                                                                                                  hop_len=hop_len,
                                                                                                  window=window),
                                                                           utls.NormalizeWithOwnMax()]))

    return timit_musdb_test



@ex.capture
def make_model(model, fft_len, text_feature_size, mix_encoder_layers, side_info_encoder_layers,
               target_decoder_layers, side_info_encoder_bidirectional):

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
def config2main(tag, test_snr, fft_len, hop_len, window, text_feature_size, side_info_type, vocabulary_size, perfect_alphas):

    return tag, test_snr, fft_len, hop_len, window, text_feature_size, side_info_type, vocabulary_size, perfect_alphas


@ex.capture
def make_eval_dir(eval_dir, tag, test_snr):
    model_eval_dir = os.path.join(eval_dir, tag + "_snr{}".format(test_snr))
    if not os.path.exists(model_eval_dir):
        os.mkdir(model_eval_dir)
    return model_eval_dir


@ex.automain
def eval_model():

    tag, test_snr, fft_len, hop_len, window, text_feature_size, side_info_type, vocabulary_size, perfect_alphas = config2main()

    test_set = make_data_set()

    model_to_evaluate = make_model()

    state_dict, training_tag = load_state_dict()

    model_to_evaluate.load_state_dict(state_dict)

    model_to_evaluate.to(device)

    print('state dict loaded')

    sdr_speech_all_snippets = []
    sdr_music_all_snippets = []
    sar_speech_all_snippets = []
    sar_music_all_snippets = []
    sir_speech_all_snippets = []
    sir_music_all_snippets = []

    stoi_speech_all_snippets = []
    pesq_speech_all_snippets = []

    pes_speech_all_snippets = []
    eps_speech_all_snippets = []

    for i in range(len(test_set)):
        sample = test_set[i]

        mix_spec = sample['mix']
        mix_phase = sample['mix_phase']

        mix_time_domain = np.expand_dims(sample['mix_time'], axis=0)
        true_speech_time_domain = np.expand_dims(sample['speech_time'], axis=0)
        true_music_time_domain = np.expand_dims(sample['music_time'], axis=0)

        mix_length = mix_spec.shape[0]

        if side_info_type == 'ones':
            side_info = torch.ones((1, mix_length, 1))
        elif side_info_type == 'phonemes':
            phonemes_idx = torch.from_numpy(sample['phonemes'])

            # output has shape (batch_size, sequence_len, vocabulary_size)
            side_info = torch.from_numpy(utls.idx2one_hot(phonemes_idx, vocabulary_size)).type(torch.float32)
            side_info = torch.unsqueeze(side_info, dim=0)

        if perfect_alphas:
            with torch.no_grad():
                predicted_speech_spec, alphas = fct.predict_with_perfect_attention(model_to_evaluate,
                                                                                   torch.unsqueeze(torch.from_numpy(
                                                                                       mix_spec), dim=0).to(device),
                                                                                   side_info.to(device),
                                                                                   torch.unsqueeze(
                                                                                       torch.from_numpy(
                                                                                           sample['perfect_alphas'])
                                                                                           .type(torch.float32), dim=0)
                                                                                   .to(device))

        else:
            with torch.no_grad():
                predicted_speech_spec, alphas = fct.predict_with_attention(model_to_evaluate,
                                                                           torch.unsqueeze(torch.from_numpy(mix_spec),
                                                                                           dim=0).to(device),
                                                                           side_info=side_info.to(device))

        complex_predicted_speech = predicted_speech_spec.detach().cpu().numpy() * np.exp(1j * mix_phase)

        predicted_speech_time_domain = lb.core.istft(complex_predicted_speech.T, hop_length=hop_len, win_length=fft_len,
                                                     window=window, center=False) * 70

        pes, eps = fct.eval_source_separation_silent_parts(
            true_speech_time_domain.flatten(), predicted_speech_time_domain.flatten(), window_size=16000,
            hop_size=16000)

        if len(pes) != 0:
            pes_speech_all_snippets.append(np.mean(pes))
        if len(eps) != 0:
            eps_speech_all_snippets.append(np.mean(eps))

        if sum(predicted_speech_time_domain) == 0:
            print("all-zero prediction:", i)
            continue
        if sum(true_music_time_domain[0, :]) == 0:
            print("all-zero music snippet", i)
            continue

        # STOI implementation from https://github.com/mpariente/pystoi
        stoi = eval_stoi(true_speech_time_domain.flatten(), predicted_speech_time_domain, fs_sig=16000)

        stoi_speech_all_snippets.append(stoi)

        # PESQ implementation from of https://github.com/vBaiCai/python-pesq
        pesq = eval_pesq(true_speech_time_domain.flatten(), predicted_speech_time_domain, 16000)

        pesq_speech_all_snippets.append(pesq)

        predicted_speech_time_domain = np.expand_dims(predicted_speech_time_domain, axis=0)

        predicted_music_time_domain = mix_time_domain - predicted_speech_time_domain

        true_sources = np.concatenate((true_speech_time_domain, true_music_time_domain), axis=0)
        predicted_sources = np.concatenate((predicted_speech_time_domain, predicted_music_time_domain), axis=0)

        me.separation.validate(true_sources, predicted_sources)

        sdr, sir, sar, perm = me.separation.bss_eval_sources_framewise(true_sources, predicted_sources,
                                                                       window=1 * 16000, hop=1 * 16000)

        # evaluation metrics for the current test snippet
        sdr_speech = sdr[0]
        sdr_music = sdr[1]
        sir_speech = sir[0]
        sir_music = sir[1]
        sar_speech = sar[0]
        sar_music = sar[1]

        # compute median over evaluation frames for current test snippet, ignore nan values
        sdr_speech_median_snippet = np.median(sdr_speech[~np.isnan(sdr_speech)])
        sdr_music_median_snippet = np.median(sdr_music[~np.isnan(sdr_music)])
        sar_speech_median_snippet = np.median(sar_speech[~np.isnan(sar_speech)])
        sar_music_median_snippet = np.median(sar_music[~np.isnan(sar_music)])
        sir_speech_median_snippet = np.median(sir_speech[~np.isnan(sir_speech)])
        sir_music_median_snippet = np.median(sir_music[~np.isnan(sir_music)])

        # append median of current snippet to list
        sdr_speech_all_snippets.append(sdr_speech_median_snippet)
        sdr_music_all_snippets.append(sdr_music_median_snippet)
        sar_speech_all_snippets.append(sar_speech_median_snippet)
        sar_music_all_snippets.append(sar_music_median_snippet)
        sir_speech_all_snippets.append(sir_speech_median_snippet)
        sir_music_all_snippets.append(sir_music_median_snippet)

    model_eval_dir = make_eval_dir()

    np.save(os.path.join(model_eval_dir, 'sdr_speech.npy'), sdr_speech_all_snippets)
    np.save(os.path.join(model_eval_dir, 'sdr_music.npy'), sdr_music_all_snippets)
    np.save(os.path.join(model_eval_dir, 'sar_speech.npy'), sar_speech_all_snippets)
    np.save(os.path.join(model_eval_dir, 'sar_music.npy'), sar_music_all_snippets)
    np.save(os.path.join(model_eval_dir, 'sir_speech.npy'), sir_speech_all_snippets)
    np.save(os.path.join(model_eval_dir, 'sir_music.npy'), sir_music_all_snippets)
    np.save(os.path.join(model_eval_dir, 'stoi_speech.npy'), stoi_speech_all_snippets)
    np.save(os.path.join(model_eval_dir, 'pesq_speech.npy'), pesq_speech_all_snippets)
    np.save(os.path.join(model_eval_dir, 'pes_speech.npy'), pes_speech_all_snippets)
    np.save(os.path.join(model_eval_dir, 'eps_speech.npy'), eps_speech_all_snippets)

    eval_summary_dict = {'tag': tag, 'test_snr': test_snr, 'fft_len': fft_len, 'hop_len': hop_len,
                         'SDR speech mean': np.mean(np.asarray(sdr_speech_all_snippets)[~np.isnan(sdr_speech_all_snippets)]),
                         'SDR speech median': np.median(np.asarray(sdr_speech_all_snippets)[~np.isnan(sdr_speech_all_snippets)]),
                         'SDR music mean': np.mean(np.asarray(sdr_music_all_snippets)[~np.isnan(sdr_music_all_snippets)]),
                         'SDR music median': np.median(np.asarray(sdr_music_all_snippets)[~np.isnan(sdr_music_all_snippets)]),
                         'SAR speech mean': np.mean(np.asarray(sar_speech_all_snippets)[~np.isnan(sar_speech_all_snippets)]),
                         'SAR speech median': np.median(np.asarray(sar_speech_all_snippets)[~np.isnan(sar_speech_all_snippets)]),
                         'SAR music mean': np.mean(np.asarray(sar_music_all_snippets)[~np.isnan(sar_music_all_snippets)]),
                         'SAR music median': np.median(np.asarray(sar_music_all_snippets)[~np.isnan(sar_music_all_snippets)]),
                         'SIR speech mean': np.mean(np.asarray(sir_speech_all_snippets)[~np.isnan(sir_speech_all_snippets)]),
                         'SIR speech median': np.median(np.asarray(sir_speech_all_snippets)[~np.isnan(sir_speech_all_snippets)]),
                         'SIR music mean': np.mean(np.asarray(sir_music_all_snippets)[~np.isnan(sir_music_all_snippets)]),
                         'SIR music median': np.median(np.asarray(sir_music_all_snippets)[~np.isnan(sir_music_all_snippets)]),
                         'STOI speech mean': np.mean(np.asarray(stoi_speech_all_snippets)),
                         'STOI speech median': np.median(np.asarray(stoi_speech_all_snippets)),
                         'PESQ speech mean': np.mean(np.asarray(pesq_speech_all_snippets)),
                         'PESQ speech median': np.median(np.asarray(pesq_speech_all_snippets)),
                         'EPS speech mean': np.mean(np.asarray(eps_speech_all_snippets)[~np.isnan(eps_speech_all_snippets)]),
                         'EPS speech median': np.median(np.asarray(eps_speech_all_snippets)[~np.isnan(eps_speech_all_snippets)]),
                         'PES speech mean': np.mean(np.asarray(pes_speech_all_snippets)[~np.isnan(pes_speech_all_snippets)]),
                         'PES speech median': np.median(np.asarray(pes_speech_all_snippets)[~np.isnan(pes_speech_all_snippets)])
                         }

    with open(os.path.join(model_eval_dir, 'eval_summary.json'), 'w') as outfile:
        json.dump(eval_summary_dict, outfile)

    print(eval_summary_dict)


