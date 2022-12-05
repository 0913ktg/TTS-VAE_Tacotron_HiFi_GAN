import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)+'/'))

import torch
import torch.nn as nn

from tacotron2.model import Tacotron2
from tacotron2.loss_function import Tacotron2Loss
from tacotron2.data_function import TextMelCollate, TextMelLoader
from tacotron2.data_function import batch_to_gpu as batch_to_gpu_tacotron2

def get_loss_function(loss_function, sigma=1.0):
    if loss_function == 'Tacotron2':
        loss = Tacotron2Loss()
    else:
        raise NotImplementedError(
            "unknown loss function requested: {}".format(loss_function))

    loss.cuda()
    return loss

def get_collate_function(model_name, n_frames_per_step=1):
    if model_name == 'Tacotron2':
        collate_fn = TextMelCollate(n_frames_per_step)
    else:
        raise NotImplementedError(
            "unknown collate function requested: {}".format(model_name))

    return collate_fn


def get_data_loader(model_name, dataset_path, audiopaths_and_text, args, speaker_ids=None, emotion_ids=None):
    if model_name == 'Tacotron2':
        # if speaker_ids is not None:
        #     data_loader = TextMelLoader(dataset_path, audiopaths_and_text, args, speaker_ids=speaker_ids)
        # else:
        #     data_loader = TextMelLoader(dataset_path, audiopaths_and_text, args)
        data_loader = TextMelLoader(dataset_path, audiopaths_and_text, args, speaker_ids=speaker_ids, emotion_ids=emotion_ids)
    else:
        raise NotImplementedError(
            "unknown data loader requested: {}".format(model_name))

    return data_loader


def get_batch_to_gpu(model_name):
    if model_name == 'Tacotron2':
        batch_to_gpu = batch_to_gpu_tacotron2
    else:
        raise NotImplementedError(
            "unknown batch_to_gpu requested: {}".format(model_name))
    return batch_to_gpu

def model_parser(model_name, parser, add_help=False):
    if model_name == 'Tacotron2':
        from tacotron2.arg_parser import tacotron2_parser
        return tacotron2_parser(parser, add_help)    
    else:
        raise NotImplementedError(model_name)


def batchnorm_to_float(module):
    """Converts batch norm to FP32"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_float(child)
    return module


def init_bn(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if module.affine:
            module.weight.data.uniform_()
    for child in module.children():
        init_bn(child)


def get_model(model_name, model_config, cpu_run,
              uniform_initialize_bn_weight=False, forward_is_infer=False):
    """ Code chooses a model based on name"""
    model = None
    if model_name == 'Tacotron2':
        if forward_is_infer:
            class Tacotron2__forward_is_infer(Tacotron2):
                def forward(self, inputs, input_lengths, ref_mel, emotion_id, style_png):
                    return self.infer(inputs, input_lengths, ref_mel, emotion_id, style_png)
            model = Tacotron2__forward_is_infer(**model_config)
        else:
            model = Tacotron2(**model_config)    
    else:
        raise NotImplementedError(model_name)

    if uniform_initialize_bn_weight:
        init_bn(model)

    if not cpu_run:
        model = model.cuda()
    return model


def get_model_config(model_name, args):
    """ Code chooses a model based on name"""
    if model_name == 'Tacotron2':
        model_config = dict(
            # optimization
            mask_padding=args.mask_padding,
            # audio
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=args.n_symbols,
            symbols_embedding_dim=args.symbols_embedding_dim,
            # encoder
            encoder_kernel_size=args.encoder_kernel_size,
            encoder_n_convolutions=args.encoder_n_convolutions,
            encoder_embedding_dim=args.encoder_embedding_dim,
            # attention
            attention_rnn_dim=args.attention_rnn_dim,
            attention_dim=args.attention_dim,
            # attention location
            attention_location_n_filters=args.attention_location_n_filters,
            attention_location_kernel_size=args.attention_location_kernel_size,
            # decoder
            n_frames_per_step=args.n_frames_per_step,
            decoder_rnn_dim=args.decoder_rnn_dim,
            prenet_dim=args.prenet_dim,
            max_decoder_steps=args.max_decoder_steps,
            gate_threshold=args.gate_threshold,
            p_attention_dropout=args.p_attention_dropout,
            p_decoder_dropout=args.p_decoder_dropout,
            # postnet
            postnet_embedding_dim=args.postnet_embedding_dim,
            postnet_kernel_size=args.postnet_kernel_size,
            postnet_n_convolutions=args.postnet_n_convolutions,
            decoder_no_early_stopping=args.decoder_no_early_stopping,
            # GST
            E=args.E,
            ref_enc_filters=args.ref_enc_filters,
            ref_enc_size=args.ref_enc_size,
            ref_enc_strides=args.ref_enc_strides,
            ref_enc_pad=args.ref_enc_pad,
            ref_enc_gru_size=args.ref_enc_gru_size,

            token_num=args.token_num,
            num_heads=args.num_heads,
            n_mels=args.n_mels,

            n_speakers=args.n_speakers,
            speaker_embedding_dim=args.speaker_embedding_dim,
            n_emotions=args.n_emotions,
            emotion_embedding_dim=args.emotion_embedding_dim,
            z_latent_dim=args.z_latent_dim
        )
        return model_config    
    else:
        raise NotImplementedError(model_name)
