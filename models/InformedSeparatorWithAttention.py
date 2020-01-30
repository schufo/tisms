"""
Copyright (c) 2019 Kilian Schulze-Forster


This is a PyTorch implementation of the audio source separation model proposed in the paper "Weakly Informed Audio
Source Separation" by Kilian Schulze-Forster, Clement Doire, Gaël Richard, Roland Badeau.

The following Python packages are required:

numpy==1.15.4
torch==1.0.1.post2

To train the model, you can create an instance of the class InformedSeparatorWithAttention.

In the experiments, the model was used with the following parameters:

separator = InformedSeparatorWithAttention(mix_features=513,
                                           mix_encoding_size=513,
                                           mix_encoder_layers=2,
                                           side_info_features=1,
                                           side_info_encoding_size=513,
                                           side_info_encoder_layers=2,
                                           connector_output_size=513,
                                           target_decoding_hidden_size=513,
                                           target_decoding_features=513,
                                           target_decoder_layers=2)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InformedSeparatorWithAttention(nn.Module):

    def __init__(self, mix_features, mix_encoding_size, mix_encoder_layers, side_info_features,
                 side_info_encoding_size, side_info_encoder_layers, connector_output_size, target_decoding_hidden_size,
                 target_decoding_features, target_decoder_layers, side_info_encoder_bidirectional=True):

        """
        :param mix_features: number of features of the mixture representation, F
        :param mix_encoding_size: number of features of the mixture encoding, E
        :param mix_encoder_layers: number of layers of the mixture encoder
        :param side_info_features: number of features of the side information representation, D
        :param side_info_encoding_size: number of features of the side information encoding, J
        :param side_info_encoder_layers: number of layers of the side information encoder
        :param connector_output_size: number of features of the first layer in the target source decoder
        :param target_decoding_hidden_size: number of features of the target source hidden representation q^{(2)}
        :param target_decoding_features: number of features of the target source decoding, F
        :param target_decoder_layers: number of LSTM layers in the target source decoder
        """

        super(InformedSeparatorWithAttention, self).__init__()

        self.mix_encoder = MixEncoder(mix_features, mix_encoding_size, mix_encoder_layers)

        self.side_info_encoder = SideInfoEncoder(side_info_features, side_info_encoding_size, side_info_encoder_layers,
                                                 side_info_encoder_bidirectional)

        if side_info_encoder_bidirectional:

            self.attention = AttentionMechanism(2 * side_info_encoding_size, 2 * mix_encoding_size)

            self.connection = ConnectionLayer(2 * side_info_encoding_size + 2 * mix_encoding_size, connector_output_size)

            self.target_decoder = TargetDecoder(connector_output_size, target_decoding_hidden_size, target_decoding_features,
                                            target_decoder_layers)
        else:
            self.attention = AttentionMechanism(side_info_encoding_size, 2 * mix_encoding_size)

            self.connection = ConnectionLayer(side_info_encoding_size + 2 * mix_encoding_size,
                                              connector_output_size)

            self.target_decoder = TargetDecoder(connector_output_size, target_decoding_hidden_size,
                                                target_decoding_features,
                                                target_decoder_layers)

        self.side_info_encoding = None
        self.mix_encoding = None
        self.combined_hidden_representation = None

    def forward(self, mix_input, side_info):
        """
        :param mix_input: mixture representation, shape: (batch_size, N, F)
        :param side_info: side information representation, shape: (batch_size, M, D)
        :return: target_prediction: prediction of the target source magnitude spectrogram, shape: (batch_size, N, F)
                 alphas: attention weights, shape: (batch_size, N, M)
        """

        self.mix_encoding = self.mix_encoder(mix_input)

        self.side_info_encoding = self.side_info_encoder(side_info)

        context_vector, alphas = self.attention(self.side_info_encoding, self.mix_encoding)

        self.combined_hidden_representation = self.connection(context_vector, self.mix_encoding)

        target_prediction = self.target_decoder(self.combined_hidden_representation)

        return target_prediction, alphas


class MixEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, layers):
        """
        :param input_size: number of features of the mixture representation, F
        :param hidden_size: number of features of the mixture encoding, E
        :param layers: number of LSTM layers
        """

        super(MixEncoder, self).__init__()
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers,
                            bidirectional=True, batch_first=True)

    def forward(self, mix_input):
        """
        :param mix_input: mixture representation, shape: (batch_size, N, F)
        :return: mix_encoding: shape: (batch_size, N, 2*E)
        """

        mix_encoding, h_n_c_n = self.LSTM(mix_input)
        return mix_encoding


class SideInfoEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, layers, bidirectional):
        """
        :param input_size: number of features of the side information representation, D
        :param hidden_size: desired number of features of the side information encoding, J
        :param layers: number of LSTM layers
        :param bidirectional: boolean, default True
        """

        super(SideInfoEncoder, self).__init__()
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers,
                            bidirectional=bidirectional, batch_first=True)

    def forward(self, inputs):
        """
        :param inputs: side information representation, shape: (batch_size, M, D)
        :return: side info encoding, shape (batch_size, M, 2*J)
        """

        # encoding has shape (batch_size, sequence_len, 2*hidden_size)

        encoding, h_n_c_n = self.LSTM(inputs)

        return encoding


class AttentionMechanism(nn.Module):

    def __init__(self, side_info_encoding_size, mix_encoding_size):
        """
        :param side_info_encoding_size: number of features of the side information encoding, 2*J
        :param mix_encoding_size: number of features of the mixture encoding, 2*E

        """
        super(AttentionMechanism, self).__init__()
        self.side_info_encoding_size = side_info_encoding_size
        self.mix_encoding_size = mix_encoding_size

        # make weight matrix Ws and initialize it
        w_s_init = torch.empty(self.mix_encoding_size, self.side_info_encoding_size)
        k = np.sqrt(1 / self.side_info_encoding_size)
        nn.init.uniform_(w_s_init, -k, k)
        self.w_s = nn.Parameter(w_s_init, requires_grad=True)

    def forward(self, side_info_encoding, mix_encoding):

        """
        :param side_info_encoding: output of side information encoder, shape; (batch_size, M, 2*J)
        :param mix_encoding: output of the mixture encoder, shape: (batch_size, N, 2*E)
        :return: context: matrix containing context vectors for each time step of the mixture encoding,
                          shape: (batch_size, N, 2*J)
                 alphas: matrix of attention weights, shape: (batch_size, N, M)
        """

        batch_size = mix_encoding.size()[0]

        current_device = side_info_encoding.device

        # compute score = g_n * W_s * h_m in two steps (equation 3 in the paper)
        side_info_transformed = torch.bmm(self.w_s.expand(batch_size, -1, -1).to(current_device),
                                          torch.transpose(side_info_encoding, 1, 2))

        scores = torch.bmm(mix_encoding, side_info_transformed)

        # compute the attention weights of all side information steps for all time steps of the target source decoder
        alphas = F.softmax(scores, dim=2)  # shape: (batch_size, N, M)

        # compute context vector for each time step of target source decoder
        context = torch.bmm(torch.transpose(side_info_encoding, 1, 2), torch.transpose(alphas, 1, 2))

        # make shape: (batch_size, N, 2*J)
        context = torch.transpose(context, 1, 2)

        return context, alphas


class ConnectionLayer(nn.Module):
    """
    This layer is part of the target source decoder in the architecture description in the paper (equation 1)
    """

    def __init__(self, input_size, output_size):
        """
        :param input_size: sum of features of the context vector and mixture encoding (2*J + 2*E)
        :param output_size: desired number of features of the hidden representation q^{(1)}
        """

        super(ConnectionLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.Tanh = nn.Tanh()

    def forward(self, context, mix_encoding):
        """
        :param context: context vector from attention mechanism, shape: (batch_size, N, 2*J)
        :param mix_encoding: output of mixture encoder, shape: (batch_size, N, 2*E)
        :return: output (hidden representation q^{(1)}), shape: (batch_size, N, output_size)
        """

        concat = torch.cat((context, mix_encoding), dim=2)
        output = self.Tanh(self.fc(concat))
        return output


class TargetDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, layers):
        """
        :param input_size: number of features of the connection layer output
        :param hidden_size: desired number of features of the hidden representation q^{(2)}
        :param output_size: number of features of target source estimation, F
        :param layers: number of LSTM layers
        """

        super(TargetDecoder, self).__init__()

        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.ReLU = nn.ReLU()

    def forward(self, inputs):
        """
        :param inputs: hidden representation q^{(1)}, shape: (batch_size, N, number_of_features)
        :return: output: prediction of the target source magnitude spectrogram, shape: (batch_size, N, F)
        """

        lstm_out, h_n_c_n = self.LSTM(inputs)
        output = self.ReLU(self.fc(lstm_out))
        return output
