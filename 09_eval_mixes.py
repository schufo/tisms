import os
import json
import sys

sys.path.append('/tsi/doctorants/kschulze/SpeechMusicSeparation/sms')
sys.path.append('/tsi/doctorants/kschulze/envs/sms_env/lib/python3.6/site-packages/pypesq-1.0-py3.6-linux-x86_64.egg')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import librosa as lb
from sacred import Experiment
from sacred.observers import FileStorageObserver
import mir_eval as me
from pystoi.stoi import stoi as eval_stoi
from pypesq import pesq as eval_pesq

import src.utilities.data_set_utls as utls
from src.models import build_models
from src.models import fct

ex = Experiment('test_eval_script')


torch.manual_seed(0)
torch.cuda.manual_seed(0)


@ex.config
def configuration():

    tag = 'mix_evaluation'
    seed = 1
    model_state_dict_name = 'model_best_val_cost.pt'

    side_info_type = 'phonemes'  # 'phonemes' or 'ones'
    data_set = 'accompaniments'  # 'with_vocals' or 'accompaniments'

    text_feature_size = 63
    vocabulary_size = 63
    phoneme_embedding_size = 257
    mix_encoder_layers = 2
    side_info_encoder_layers = 2
    target_decoder_layers = 2
    align_phonemes = False
    sigma = None
    fft_len = 512
    hop_len = 256
    window = 'hamming'


    eval_dir = '/tsi/doctorants/kschulze/SpeechMusicSeparation/sms/reports/evaluations'


    test_snr = -5
    train_snr = None
    snr = None


@ex.capture
def make_data_set(data_set, test_snr, fft_len, hop_len, window, align_phonemes):

    if data_set == 'accompaniments':
        import src.data.timit_musdb_test as test_set


        if align_phonemes:
            timit_musdb_test = test_set.Test(transform=transforms.Compose([utls.MixSNR(test_snr),
                                                                       utls.StftOnFly_testset(fft_len=fft_len,
                                                                                      hop_len=hop_len,
                                                                                      window=window),
                                                                       utls.NormalizeWithOwnMax(),
                                                                       utls.AlignPhonemes(fft_len, hop_len)]))
            print('aligned phonemes')

        else:
            timit_musdb_test = test_set.Test(transform=transforms.Compose([utls.MixSNR(test_snr),
                                                                       utls.StftOnFly_testset(fft_len=fft_len,
                                                                                              hop_len=hop_len,
                                                                                              window=window),
                                                                       utls.NormalizeWithOwnMax()]))

    if data_set == 'with_vocals':
        import src.data.timit_musdb_test_with_vocals as test_set

        if align_phonemes:
            timit_musdb_test = test_set.Test(transform=transforms.Compose([utls.MixSNR(test_snr),
                                                                       utls.StftOnFly_testset(fft_len=fft_len,
                                                                                      hop_len=hop_len,
                                                                                      window=window),
                                                                       utls.NormalizeWithOwnMax(),
                                                                       utls.AlignPhonemes(fft_len, hop_len)]))
            print('aligned phonemes')

        else:
            timit_musdb_test = test_set.Test(transform=transforms.Compose([utls.MixSNR(test_snr),
                                                                       utls.StftOnFly_testset(fft_len=fft_len,
                                                                                              hop_len=hop_len,
                                                                                              window=window),
                                                                       utls.NormalizeWithOwnMax()]))

    return timit_musdb_test



@ex.capture
def make_model(model, fft_len, text_feature_size, phoneme_embedding_size, mix_encoder_layers,
               side_info_encoder_layers, target_decoder_layers, sigma):

    mix_features_size = int(fft_len / 2 + 1)

    if model == 'InformedSeparatorWithAttention':
        separator = build_models.make_informed_separator_with_attention(mix_features_size, text_feature_size,
                                                                        mix_encoder_layers, side_info_encoder_layers,
                                                                        target_decoder_layers)
    elif model == 'InformedSeparatorWithAttentionAndEmbedding':
        separator = build_models.make_informed_separator_with_attention_and_embedding(mix_features_size,
                                                                                      text_feature_size,
                                                                                      mix_encoder_layers,
                                                                                      side_info_encoder_layers,
                                                                                      target_decoder_layers,
                                                                                      phoneme_embedding_size)
    elif model == 'InformedSeparatorWithAttentionSkip':
        separator = build_models.make_informed_separator_with_attention_skip(mix_features_size, text_feature_size,
                                                                            mix_encoder_layers,
                                                                            side_info_encoder_layers,
                                                                            target_decoder_layers)
    elif model == 'InformedSeparatorWithLocMonAttention':
        separator = build_models.make_informed_separator_with_loc_mon_attention(mix_features_size, text_feature_size,
                                                                        mix_encoder_layers, side_info_encoder_layers,
                                                                        target_decoder_layers, sigma=sigma)


    return separator


@ex.capture
def load_state_dict(models_directory, tag, model_state_dict_name):
    checkpoint = torch.load(os.path.join(models_directory, tag, model_state_dict_name))
    return checkpoint['model_state_dict'], checkpoint['experiment_tag']

@ex.capture
def config2main(tag, test_snr, fft_len, hop_len, window, text_feature_size, side_info_type, vocabulary_size):

    return tag, test_snr, fft_len, hop_len, window, text_feature_size, side_info_type, vocabulary_size


@ex.capture
def make_eval_dir(eval_dir, tag):
    model_eval_dir = os.path.join(eval_dir, tag)
    if not os.path.exists(model_eval_dir):
        os.mkdir(model_eval_dir)
        # print("Directory ", experiment_dir, " Created ")
    else:
        # print("Directory ", experiment_dir, " already exists")
        pass
    return model_eval_dir


@ex.automain
def eval_model():

    tag, test_snr, fft_len, hop_len, window, text_feature_size, side_info_type, vocabulary_size = config2main()

    print(tag, fft_len, hop_len, window, text_feature_size)

    test_set = make_data_set()

    sdr_speech_all_snippets = []
    sdr_music_all_snippets = []
    sar_speech_all_snippets = []
    sar_music_all_snippets = []
    sir_speech_all_snippets = []
    sir_music_all_snippets = []

    stoi_speech_all_snippets = []
    pesq_speech_all_snippets = []

    for i in range(len(test_set)):
        sample = test_set[i]

        mix_spec = sample['mix']
        mix_phase = sample['mix_phase']

        mix_time_domain = np.expand_dims(sample['mix_time'], axis=0)
        true_speech_time_domain = np.expand_dims(sample['speech_time'], axis=0)
        true_music_time_domain = np.expand_dims(sample['music_time'], axis=0)

        mix_length = mix_spec.shape[0]

        # predict speech and music with mix
        predicted_speech_time_domain = np.expand_dims(mix_time_domain.flatten(), axis=0)
        predicted_music_time_domain = np.expand_dims(mix_time_domain.flatten(), axis=0)

        if sum(predicted_speech_time_domain[0]) == 0:
            print("all-zero prediction:", i)
            continue
        if sum(true_music_time_domain[0, :]) == 0:
            print("all-zero music snippet", i)
            continue

        stoi = eval_stoi(true_speech_time_domain.flatten(), mix_time_domain.flatten(), fs_sig=16000)

        stoi_speech_all_snippets.append(stoi)

        pesq = eval_pesq(true_speech_time_domain.flatten(), mix_time_domain.flatten(), 16000)

        pesq_speech_all_snippets.append(pesq)


        true_sources = np.concatenate((true_speech_time_domain, true_music_time_domain), axis=0)
        predicted_sources = np.concatenate((predicted_speech_time_domain, predicted_music_time_domain), axis=0)

        me.separation.validate(true_sources, predicted_sources)

        sdr, sir, sar, perm = me.separation.bss_eval_sources_framewise(true_sources, predicted_sources,
                                                                       window=1 * 16000, hop=1 * 16000)

        # I am using the implementation of https://github.com/vBaiCai/python-pesq for evaluation of PESQ
        # (There is also this: https://github.com/ludlows/python-pesq)
        # for STOI I am using https://github.com/mpariente/pystoi


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

    # # convert lists to json format
    # sdr_speech_json = json.dumps(sdr_speech_all_snippets)
    # sdr_music_json = json.dumps(sdr_music_all_snippets)
    # sar_speech_json = json.dumps(sar_speech_all_snippets)
    # sar_music_json = json.dumps(sar_music_all_snippets)
    # sir_speech_json = json.dumps(sir_speech_all_snippets)
    # sir_music_json = json.dumps(sir_music_all_snippets)
    #
    # stoi_speech_json = json.dumps(stoi_speech_all_snippets)
    # pesq_speech_json = json.dumps(pesq_speech_all_snippets)


    np.save(os.path.join(model_eval_dir, 'sdr_speech.npy'), sdr_speech_all_snippets)
    np.save(os.path.join(model_eval_dir, 'sdr_music.npy'), sdr_music_all_snippets)
    np.save(os.path.join(model_eval_dir, 'sar_speech.npy'), sar_speech_all_snippets)
    np.save(os.path.join(model_eval_dir, 'sar_music.npy'), sar_music_all_snippets)
    np.save(os.path.join(model_eval_dir, 'sir_speech.npy'), sir_speech_all_snippets)
    np.save(os.path.join(model_eval_dir, 'sir_music.npy'), sir_music_all_snippets)
    np.save(os.path.join(model_eval_dir, 'stoi_speech.npy'), stoi_speech_all_snippets)
    np.save(os.path.join(model_eval_dir, 'pesq_speech.npy'), pesq_speech_all_snippets)



    # # save lists with eval results as json files
    # with open(os.path.join(model_eval_dir, 'sdr_speech.json'), 'w') as outfile:
    #     json.dump(sdr_speech_json, outfile)
    # with open(os.path.join(model_eval_dir, 'sdr_music.json'), 'w') as outfile:
    #     json.dump(sdr_music_json, outfile)
    # with open(os.path.join(model_eval_dir, 'sar_speech.json'), 'w') as outfile:
    #     json.dump(sar_speech_json, outfile)
    # with open(os.path.join(model_eval_dir, 'sar_music.json'), 'w') as outfile:
    #     json.dump(sar_music_json, outfile)
    # with open(os.path.join(model_eval_dir, 'sir_speech.json'), 'w') as outfile:
    #     json.dump(sir_speech_json, outfile)
    # with open(os.path.join(model_eval_dir, 'sir_music.json'), 'w') as outfile:
    #     json.dump(sir_music_json, outfile)
    # with open(os.path.join(model_eval_dir, 'stoi_speech.json'), 'w') as outfile:
    #     json.dump(stoi_speech_json, outfile)
    # with open(os.path.join(model_eval_dir, 'pesq_speech.json'), 'w') as outfile:
    #     json.dump(pesq_speech_json, outfile)
    # # #
    # print("SDR speech: mean:", np.mean(sdr_speech_all_snippets), "median:", np.median(sdr_speech_all_snippets))
    # print("SDR music: mean: ", np.mean(sdr_music_all_snippets), "median:", np.median(sdr_music_all_snippets))
    # print("SAR speech: mean:", np.mean(sar_speech_all_snippets), "median:", np.median(sar_speech_all_snippets))
    # print("SAR music: mean: ", np.mean(sar_music_all_snippets), "median:", np.median(sar_music_all_snippets))
    # print("SIR speech: mean:", np.mean(sir_speech_all_snippets), "median:", np.median(sir_speech_all_snippets))
    # print("SIR music: mean: ", np.mean(sir_music_all_snippets), "median:", np.median(sir_music_all_snippets))

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
                         'PESQ speech median': np.median(np.asarray(pesq_speech_all_snippets))
                         }

    with open(os.path.join(model_eval_dir, 'eval_summary.json'), 'w') as outfile:
        json.dump(eval_summary_dict, outfile)

    print(eval_summary_dict)


    # sd.play(predicted_speech_time_domain.flatten(), 16000)
    # sd.wait()
    # sd.play(true_speech_time_domain.flatten(), 16000)
    # sd.wait()
    # sd.play(mix_time_domain.flatten(), 16000)
    # sd.wait()
    # sd.play(predicted_music_time_domain.flatten(), 16000)
    # sd.wait()
    # sd.play(true_music_time_domain.flatten(), 16000)
    # sd.wait()



