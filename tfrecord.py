import numpy as np 
import os
import multiprocessing
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tqdm import tqdm


class TFRecordCreator:
    def __init__(self, metadata_filename, hparams):
        self._hparams = hparams
        self._metadata_filename = metadata_filename
        self._pad = 0.
        self._basedir = os.path.dirname(metadata_filename)
        
    def _get_example(self, audio, mel, speaker_id=None):
        audio_list = tf.train.FloatList(value=audio)
        audio_len_list = tf.train.Int64List(value=[np.int32(audio.shape[0])])
        mel_shape_list = tf.train.Int64List(value=np.int32(mel.shape))
        mel_list = tf.train.FloatList(value=mel.flatten())

        if speaker_id is not None:
            speaker_id_list =  tf.train.Int64List(value=[speaker_id])

        feature_key_value_pair = {
            'audio': tf.train.Feature(float_list=audio_list),
            'audio_len': tf.train.Feature(int64_list=audio_len_list),
            'mel_shape': tf.train.Feature(int64_list=mel_shape_list),
            'mel': tf.train.Feature(float_list=mel_list),
        }

        if speaker_id is not None:
            feature_key_value_pair['speaker_id'] = tf.train.Feature(int64_list=speaker_id_list)

        features = tf.train.Features(feature=feature_key_value_pair)
        example = tf.train.Example(features=features)
        return example


    def _adjust_time_resolution(self, audio, mel, speaker_id=None):
        if audio.shape[0] < self._hparams.max_time_steps:
            audio_pad = self._hparams.max_time_steps - audio.shape[0]
            mel_pad = audio_pad // self._hparams.hop_size
            audio = np.pad(audio, (0, audio_pad), mode='constant', constant_values=self._pad)
            mel = np.pad(mel, ((0, mel_pad), (0, 0)), mode='constant', constant_values=self._pad)
        
        self._assert_ready_for_upsample(audio, mel)
        return audio, mel, speaker_id


    def _assert_ready_for_upsample(self, x, c):
        assert len(x) % len(c) == 0 and len(x) // len(c) == self._hparams.hop_size


    def _py_load_sample(self, audio_filename, mel_filename, speaker_id):
        mel = np.load(os.path.join(self._basedir, 'mels', mel_filename))
        audio = np.load(os.path.join(self._basedir, 'audios', audio_filename))
        speaker_id = np.int32(speaker_id)

        return audio, mel, speaker_id
    

    def _write_tfrecord(self, output_filename, meta):
        with tf.python_io.TFRecordWriter(os.path.join(self._basedir, output_filename)) as tfwriter:
            for m in tqdm(meta):
                audio_filename, mel_filename, _, speaker_id, _ = m
                audio, mel, speaker_id = self._py_load_sample(audio_filename, mel_filename, speaker_id)
                if self._hparams.gin_channels > 0:
                    example =self._get_example(audio, mel, speaker_id)
                else:
                    example = self._get_example(audio, mel)
                tfwriter.write(example.SerializeToString())


    def create_tfrecords(self):
        with open(self._metadata_filename, encoding='utf-8') as f:
            metadata = [line.strip().split('|') for line in f]

        indices = np.arange(len(metadata))
        train_indices, test_indices = train_test_split(indices,
            test_size=self._hparams.test_size, random_state=self._hparams.split_random_state)

        train_meta = list(np.array(metadata)[train_indices])
        test_meta = list(np.array(metadata)[test_indices])

        self._write_tfrecord('train.tfrecord', train_meta)
        self._write_tfrecord('test.tfrecord', test_meta)