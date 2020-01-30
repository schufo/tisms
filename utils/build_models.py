

def make_informed_separator_with_attention(mix_features_size, text_feature_size, mix_encoder_layers,
                                           side_info_encoder_layers, target_decoder_layers, side_info_encoder_bidirectional=True):

    from models import InformedSeparatorWithAttention as Model

    network = Model.InformedSeparatorWithAttention(mix_features=mix_features_size,
                                                   mix_encoding_size=mix_features_size,
                                                   mix_encoder_layers=mix_encoder_layers,
                                                   side_info_features=text_feature_size,
                                                   side_info_encoding_size=mix_features_size,
                                                   side_info_encoder_layers=side_info_encoder_layers,
                                                   connector_output_size=mix_features_size,
                                                   target_decoding_hidden_size=mix_features_size,
                                                   target_decoding_features=mix_features_size,
                                                   target_decoder_layers=target_decoder_layers,
                                                   side_info_encoder_bidirectional=side_info_encoder_bidirectional)
    return network


def make_informed_separator_with_perfect_attention(mix_features_size, text_feature_size, mix_encoder_layers,
                                           side_info_encoder_layers, target_decoder_layers, side_info_encoder_bidirectional):

    from models import InformedSeparatorWithPerfectAttention as Model

    network = Model.InformedSeparatorWithPerfectAttention(mix_features=mix_features_size,
                                                   mix_encoding_size=mix_features_size,
                                                   mix_encoder_layers=mix_encoder_layers,
                                                   side_info_features=text_feature_size,
                                                   side_info_encoding_size=mix_features_size,
                                                   side_info_encoder_layers=side_info_encoder_layers,
                                                   connector_output_size=mix_features_size,
                                                   target_decoding_hidden_size=mix_features_size,
                                                   target_decoding_features=mix_features_size,
                                                   target_decoder_layers=target_decoder_layers,
                                                side_info_encoder_bidirectional=side_info_encoder_bidirectional)
    return network



def make_informed_separator_with_split_attention(mix_features_size, text_feature_size, mix_encoder_layers,
                                           side_info_encoder_layers, target_decoder_layers, side_info_encoder_bidirectional=True):

    from models import InformedSeparatorWithSplitAttention as Model

    network = Model.InformedSeparatorWithSplitAttention(mix_features=mix_features_size,
                                                   mix_encoding_size=mix_features_size,
                                                   mix_encoder_layers=mix_encoder_layers,
                                                   side_info_features=text_feature_size,
                                                   side_info_encoding_size=mix_features_size,
                                                   side_info_encoder_layers=side_info_encoder_layers,
                                                   connector_output_size=mix_features_size,
                                                   target_decoding_hidden_size=mix_features_size,
                                                   target_decoding_features=mix_features_size,
                                                   target_decoder_layers=target_decoder_layers,
                                                   side_info_encoder_bidirectional=side_info_encoder_bidirectional)
    return network
