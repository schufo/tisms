import torch
import numpy as np
import librosa as lb
from torch.nn.utils.rnn import pad_sequence


# PyTorch related utilities
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def collate_with_phonemes(sample_list):

    batch_size = len(sample_list)

    # make list of phonemes, mix, speech, music of the batch
    list_of_phoneme_sequences = [torch.from_numpy(sample_list[n]['phonemes']) for n in range(batch_size)]
    list_of_mix_specs = [torch.from_numpy(sample_list[n]['mix']) for n in range(batch_size)]
    list_of_speech_specs = [torch.from_numpy(sample_list[n]['target']) for n in range(batch_size)]
    list_of_music_specs = [torch.from_numpy(sample_list[n]['music']) for n in range(batch_size)]
    list_of_perfect_alphas = [torch.from_numpy(sample_list[n]['perfect_alphas'].T) for n in range(batch_size)]


    # pad phonemes to length of longest phoneme sequence in batch and stack them along dim=0
    phonemes_batched = pad_sequence(list_of_phoneme_sequences, batch_first=True, padding_value=0).type(torch.float32)

    alphas_batched = pad_sequence(list_of_perfect_alphas, batch_first=True, padding_value=0).type(torch.float32)
    alphas_batched = torch.transpose(alphas_batched, 1, 2)

    # stack other elements in batch that have the same size across individual samples
    mix_batched = torch.stack(list_of_mix_specs, dim=0)
    speech_batched = torch.stack(list_of_speech_specs, dim=0)
    music_batched = torch.stack(list_of_music_specs, dim=0)

    samples_batched = {'target': speech_batched, 'music': music_batched, 'mix': mix_batched,
                       'phonemes': phonemes_batched, 'perfect_alphas': alphas_batched}

    return samples_batched

# transformations that can be applied when creating an instance of a data set

class MixSNR(object):
    """
    The energy ratio of speech and music is measured over samples where the speech is active only
    """

    def __init__(self, desired_snr):
        self.snr_desired = desired_snr

    def __call__(self, sample):
        music = sample['music']
        speech = sample['speech']
        speech_len = sample['speech_len']
        speech_start = sample['speech_start']
        phonemes = sample['phonemes']

        speech_energy = sum(speech ** 2)
        music_energy_at_speech_overlap = sum(music[speech_start: speech_start + speech_len] ** 2)

        target_snr = self.snr_desired

        if self.snr_desired == 'random':
            target_snr = np.random.uniform(-8, 0)

        if music_energy_at_speech_overlap > 0.1:
            snr_current = 10 * np.log10(speech_energy / music_energy_at_speech_overlap)
            snr_difference = target_snr - snr_current
            scaling = (10 ** (snr_difference / 10))
            speech_scaled = speech * np.sqrt(scaling)
            mix = speech_scaled + music
            mix_max = abs(mix).max()
            mix = mix / mix_max
            speech_scaled = speech_scaled / mix_max
            music = music / mix_max
        else:
            mix = speech + music
            mix_max = abs(mix).max()
            mix = mix / mix_max
            speech_scaled = speech / mix_max
            music = music / mix_max

        sample = {'mix': mix, 'target': speech_scaled, 'music': music, 'phonemes': phonemes,
                  'speech_start': speech_start, 'speech_len': speech_len, 'phoneme_times': sample['phoneme_times'],
                  'perfect_alphas': np.ones((1, 1))}
        return sample


class Stft_torch(object):

    def __init__(self, fft_len, hop_len, device):
        self.fft_len = fft_len
        self.hop_len = hop_len
        self.device = device

    def __call__(self, sample):

        mix = torch.from_numpy(sample['mix']).to(self.device)
        music = torch.from_numpy(sample['music']).to(self.device)
        speech = torch.from_numpy(sample['speech']).to(self.device)

        window = torch.hamming_window(self.fft_len, periodic=False).double().to(self.device)

        mix_stft = torch.stft(mix, n_fft=self.fft_len, hop_length=self.hop_len, win_length=self.fft_len, window=window, center=False)

        music_stft = torch.stft(music, n_fft=self.fft_len, hop_length=self.hop_len, win_length=self.fft_len,
                                   window=window, center=False)
        speech_stft = torch.stft(speech, n_fft=self.fft_len, hop_length=self.hop_len, win_length=self.fft_len,
                                   window=window, center=False)


        mix_mag_spec = torch.sqrt(mix_stft[:, :, 0]**2 + mix_stft[:, :, 1]**2)
        music_mag_spec = torch.sqrt(music_stft[:, :, 0]**2 + music_stft[:, :, 1]**2)
        speech_mag_spec = torch.sqrt(speech_stft[:, :, 0]**2 + speech_stft[:, :, 1]**2)

        print(mix_mag_spec.transpose(0, 1).size())

        sample['target'] = speech_mag_spec.transpose(0, 1)
        sample['music'] = music_mag_spec.transpose(0, 1)
        sample['mix'] = mix_mag_spec.transpose(0, 1)
        return sample


class StftOnFly(object):

    def __init__(self, fft_len, hop_len, window):
        self.fft_len = fft_len
        self.hop_len = hop_len
        self.window = window

    def __call__(self, sample):
        mix = sample['mix']
        music = sample['music']
        speech = sample['target']

        mix_stft = lb.core.stft(mix, n_fft=self.fft_len, hop_length=self.hop_len, win_length=self.fft_len,
                                window=self.window, center=False)
        music_stft = lb.core.stft(music, n_fft=self.fft_len, hop_length=self.hop_len, win_length=self.fft_len,
                                  window=self.window, center=False)
        speech_stft = lb.core.stft(speech, n_fft=self.fft_len, hop_length=self.hop_len, win_length=self.fft_len,
                                   window=self.window, center=False)

        mix_mag_spec = abs(mix_stft).T
        music_mag_spec = abs(music_stft).T
        speech_mag_spec = abs(speech_stft).T

        sample['target'] = speech_mag_spec
        sample['music'] = music_mag_spec
        sample['mix'] = mix_mag_spec
        return sample

class StftOnFly_testset(object):

    def __init__(self, fft_len, hop_len, window):
        self.fft_len = fft_len
        self.hop_len = hop_len
        self.window = window

    def __call__(self, sample):

        # time domain signals
        mix = sample['mix']
        music = sample['music']
        speech = sample['target']

        mix_stft = lb.core.stft(mix, n_fft=self.fft_len, hop_length=self.hop_len, win_length=self.fft_len,
                                window=self.window, center=False)
        music_stft = lb.core.stft(music, n_fft=self.fft_len, hop_length=self.hop_len, win_length=self.fft_len,
                                  window=self.window, center=False)
        speech_stft = lb.core.stft(speech, n_fft=self.fft_len, hop_length=self.hop_len, win_length=self.fft_len,
                                   window=self.window, center=False)

        mix_mag_spec = abs(mix_stft).T
        music_mag_spec = abs(music_stft).T
        speech_mag_spec = abs(speech_stft).T

        mix_phase = np.angle(mix_stft).T

        sample['target'] = speech_mag_spec
        sample['music'] = music_mag_spec
        sample['mix'] = mix_mag_spec
        sample['mix_phase'] = mix_phase
        sample['mix_time'] = mix
        sample['music_time'] = music
        sample['speech_time'] = speech
        return sample

class NormalizeWithOwnMax(object):

    def __call__(self, sample):
        mix_spec = sample['mix']
        music_spec = sample['music']
        speech_spec = sample['target']

        mix_max = mix_spec.max()
        music_max = music_spec.max()
        speech_max = speech_spec.max()

        mix_norm_spec = mix_spec / mix_max
        if music_max > 0:
            music_norm_spec = music_spec / music_max
        else:
            music_norm_spec = music_spec
        speech_norm_spec = speech_spec / speech_max

        sample['target'] = speech_norm_spec
        sample['music'] = music_norm_spec
        sample['mix'] = mix_norm_spec
        return sample


class MakePerfectAttentionWeights(object):

    def __init__(self, fft_len, hop_len):
        self.fft_len = fft_len
        self.hop_len = hop_len

    def __call__(self, sample):
        phonemes = sample['phonemes']  # sequence of phoneme indices
        phoneme_times = sample['phoneme_times']  # start of phonemes, last number=last phoneme's end (rel. speech_start)
        speech_start = sample['speech_start']  # start of the speech recording in the mix (!= first phoneme start)
        speech_len = sample['speech_len']  # length of the speech recording
        time_frames = sample['mix'].shape[0]  # number of time frames in spectrograms

        sequence_of_phoneme_sample_idx = np.zeros((time_frames,), dtype=int)

        # assign phoneme indices to frames where they are active at least over half of the frame length
        for n in range(0, len(phoneme_times) - 1):
            phoneme_start_frame = int(np.floor((speech_start + phoneme_times[n]) / self.hop_len))
            phoneme_end_frame = int(np.floor((speech_start + phoneme_times[n + 1]) / self.hop_len))

            if phoneme_start_frame < phoneme_end_frame:
                sequence_of_phoneme_sample_idx[phoneme_start_frame: phoneme_end_frame] = n + 1
            # elif phoneme_start_frame == phoneme_end_frame:
            #     pass

        # assign idx of last silence token to silent frames at end of speech frames
        sequence_of_phoneme_sample_idx[phoneme_end_frame:] = n + 2

        alphas = idx2one_hot(torch.from_numpy(sequence_of_phoneme_sample_idx), len(phonemes))

        sample['perfect_alphas'] = alphas

        return sample


class AlignPhonemes(object):

    def __init__(self, fft_len, hop_len):
        self.fft_len = fft_len
        self.hop_len = hop_len

    def __call__(self, sample):
        phonemes = sample['phonemes']  # sequence of phoneme indices
        phoneme_times = sample['phoneme_times']  # start of phonemes, last number=last phoneme's end (rel. speech_start)
        speech_start = sample['speech_start']  # start of the speech recording in the mix (!= first phoneme start)
        speech_len = sample['speech_len']  # length of the speech recording
        time_frames = sample['mix'].shape[0]  # number of time frames in spectrograms

        aligned_phoneme_idx_sequence = np.ones((time_frames,), dtype=int)

        # assign noise token (index 2) to frames where speech recording does not contain voice yet
        speech_recording_start_frame = int(np.floor(speech_start / self.hop_len))
        phoneme_start_frame = int(np.floor((speech_start + phoneme_times[0]) / self.hop_len))
        if speech_recording_start_frame < phoneme_start_frame:
           aligned_phoneme_idx_sequence[speech_recording_start_frame: phoneme_start_frame] = 2

        # assign phoneme indices to frames where they are active at least over half of the frame length
        for n in range(0, len(phoneme_times) - 1):
            phoneme_start_frame = int(np.floor((speech_start + phoneme_times[n]) / self.hop_len))
            phoneme_end_frame = int(np.floor((speech_start + phoneme_times[n + 1]) / self.hop_len))

            if phoneme_start_frame < phoneme_end_frame:
                aligned_phoneme_idx_sequence[phoneme_start_frame: phoneme_end_frame] = phonemes[n + 1]

        # assign noise token (index 2) to frames where speech is not active but noise is still in ground truth
        last_phoneme_end_frame = int(np.floor((speech_start + phoneme_times[-1]) / self.hop_len))
        speech_recording_end_frame = int(np.floor((speech_start + speech_len) / self.hop_len))
        if last_phoneme_end_frame < speech_recording_end_frame:
            aligned_phoneme_idx_sequence[last_phoneme_end_frame: speech_recording_end_frame] = 2
        # elif last_phoneme_end_frame == speech_recording_end_frame:
        #     pass

        sample['phonemes'] = aligned_phoneme_idx_sequence

        return sample

def idx2one_hot(idx_sentence, vocabulary_size):
    sentence_one_hot_encoded = []
    for idx in idx_sentence:
        phoneme = [0 for _ in range(vocabulary_size)]
        phoneme[idx.type(torch.int)] = 1
        sentence_one_hot_encoded.append(np.array(phoneme))
    return np.array(sentence_one_hot_encoded)

