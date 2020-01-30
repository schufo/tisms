"""
basic functions needed to train and test deep learning models with PyTorch
"""
import torch
import numpy as np
import sys


def make_fake_side_info(voice_spectro_tensor):
    voice_energy = torch.sum(voice_spectro_tensor, dim=2, keepdim=True)

    fake_side_info = torch.ones_like(voice_energy)

    return fake_side_info


def viterbi_alignment_from_attention(attention_weights, hop_len):

    """
    :param attention_weights: shape (M, N)
    :param hop_len: int
    :return:
    """

    M = attention_weights.shape[0]
    N = attention_weights.shape[1]

    # transition probabilities are zero everywhere except when going back to the same state (m --> m)
    # or moving to next state (m --> m+1). First dimension (vertical) is the starting state,
    # second dimension (horizontally) is the arriving state

    # initialize transition probabilities to 0.5 for both allowed cases
    trans_p = np.zeros((M, M))
    for m in range(M):
        trans_p[m, m] = 0.5
        if m < M - 1:
            trans_p[m, m+1] = 0.5

    # initialization
    delta = np.zeros((M, N))  # shape: (states, time_steps), contains delta_n(m)
    delta[0, 0] = 1  # delta_0(0) = 1 first state (silence token) must be active at first time step

    psi = np.zeros((M, N))  # state that is most likely predecessor of state m at time step n

    # recurrence
    for n in range(1, N):
        for m in range(M):

            delta_m_n_candidates = []
            for m_candidate in range(M):
                delta_m_n_candidates.append(delta[m_candidate, n-1] * trans_p[m_candidate, m])

            delta[m, n] = max(delta_m_n_candidates) * (attention_weights[m, n] * 2 + 1)

            psi[m, n] = np.argmax(delta_m_n_candidates)

    np.set_printoptions(threshold=sys.maxsize)

    optimal_state_sequence = np.zeros((1, N))
    optimal_state_sequence[0, N-1] = int(M - 1)  # force the last state (silent token) to be active at last time step

    for n in range(N-2, 0, -1):

        optimal_state_sequence[0, n] = (psi[int(optimal_state_sequence[0, n+1]), n+1])

    # compute index of list elements whose right neighbor is different from itself
    last_idx_before_change = [i for i, (x, y) in enumerate(zip(optimal_state_sequence[0, :-1], optimal_state_sequence[0, 1:])) if x != y]

    # compute phoneme onset times from idx of last time frame previous phoneme
    phoneme_onsets_prediction = [(n + 1) * hop_len / 16000 for n in last_idx_before_change]

    phoneme_onsets_prediction = phoneme_onsets_prediction[:-1]  # remove onset prediction of silence token

    # the optimal_state_sequence is a sequence of phoneme indices with length N
    return optimal_state_sequence.astype(int), phoneme_onsets_prediction


def train_with_attention(model, loss_function, optimizer, mix_inputs, side_info, targets):

    model.train()
    optimizer.zero_grad()

    # Forward
    output_of_network, _ = model(mix_inputs, side_info)
    loss = loss_function(output_of_network, targets)

    # Backward
    loss.backward()

    # Update parameters
    optimizer.step()

    # return a number that represents the loss
    return loss.item()


def train_with_perfect_attention(model, loss_function, optimizer, mix_inputs, side_info, targets, alphas):

    model.train()
    optimizer.zero_grad()

    # Forward
    output_of_network, _ = model(mix_inputs, side_info, alphas)
    loss = loss_function(output_of_network, targets)

    # Backward
    loss.backward()

    # Update parameters
    optimizer.step()

    # return a number that represents the loss
    return loss.item()


def predict_with_attention(model, mix_input, side_info):

    model.eval()
    prediction, alphas = model(mix_input, side_info)
    return prediction, alphas


def predict_with_perfect_attention(model, mix_input, side_info, alphas_in):

    model.eval()
    prediction, alphas = model(mix_input, side_info, alphas_in)
    return prediction, alphas


def eval_source_separation_silent_parts(true_source, predicted_source, window_size, hop_size):

    num_eval_windows = int(np.ceil((len(true_source) - abs(hop_size - window_size)) / hop_size)) -1

    list_prediction_energy_at_true_silence = []
    list_true_energy_at_predicted_silence = []

    for ii in range(num_eval_windows):

        prediction_window = predicted_source[ii * window_size: ii * window_size + window_size]
        true_window = true_source[ii * window_size: ii * window_size + window_size]

        # compute predicted energy for silent true source (PESTS)
        if sum(abs(true_window)) == 0:
            prediction_energy_at_true_silence = 10 * np.log10(sum(prediction_window**2) + 10**(-12))
            list_prediction_energy_at_true_silence.append(prediction_energy_at_true_silence)
        else:
            # list_prediction_energy_at_true_silence.append(np.nan)
            pass

        # compute energy of true source when silence (all zeros) is predicted and true source is not silent//
        # True Energy at Wrong Silence Prediction (TEWSP)
        if sum(abs(prediction_window)) == 0 and sum(abs(true_window)) != 0:
            true_source_energy_at_silent_prediction = 10 * np.log10(sum(true_window**2) + 10**(-12))
            list_true_energy_at_predicted_silence.append(true_source_energy_at_silent_prediction)
        else:
            # list_true_energy_at_predicted_silence.append(np.nan)
            pass

    return np.asarray(list_prediction_energy_at_true_silence), np.asarray(list_true_energy_at_predicted_silence)
