import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
import multiprocessing

_buffer_size = 500
_pad = 0

class Dataset:
    def __init__(self,  train_tfrecord, test_tfrecord, hparams):
        self._hparams = hparams
        self._train_tfrecord = train_tfrecord
        self._test_tfrecord = test_tfrecord
        
        #pad input sequences with the <pad_token> 0 ( _ )
        self._pad = 0.
        #explicitely setting the padding to a value that doesn't originally exist in the spectogram
        #to avoid any possible conflicts, without affecting the output range of the model too much
        
        n_cpu = multiprocessing.cpu_count()
        buffer_size = 64

        with tf.device('/cpu:0'):
            self._filenames = tf.placeholder(tf.string, shape=[None])
            dataset = tf.data.TFRecordDataset(self._filenames)
            dataset = dataset.map(self._load_sample, n_cpu)
            dataset = dataset.cache()
            dataset = dataset.repeat()
            dataset = dataset.padded_batch(hparams.tacotron_batch_size, 
                                           padded_shapes=([-1, self._hparams.num_mels], 
                                           [-1],
                                           [1]),
                                            padding_values=(self._pad, self._pad, 0))     
            dataset = dataset.shuffle(buffer_size)
            dataset = dataset.map(self._postprocess_batch, n_cpu)
            dataset = dataset.prefetch(hparams.num_gpus)

        self._train_iterator = dataset.make_initializable_iterator()
        self.inputs = []
        self.local_conditions = []
        self.speaker_ids = []
        for _ in range(hparams.num_gpus):
            train_batch = self._train_iterator.get_next()
            self.inputs.append(train_batch[0])
            self.local_conditions.append(train_batch[1])
            self.speaker_ids.append(train_batch[2])
                
        self._test_iterator = dataset.make_initializable_iterator()
        test_batch = self._test_iterator.get_next()
        self.eval_inputs = test_batch[0]
        self.eval_local_conditions = test_batch[1]
        self.eval_speaker_ids = test_batch[2]

    def _load_sample(self, data_record):
        features = {
            'audio': tf.VarLenFeature(tf.float32),
            'audio_len': tf.FixedLenFeature([], tf.int64),
            'mel_shape': tf.FixedLenFeature([2], tf.int64),
            'mel': tf.VarLenFeature(tf.float32)
        }

        if self._hparams.gin_channels > 0:
            features['speaker_id'] = tf.FixedLenFeature([], tf.int64)

        sample = tf.parse_single_example(data_record, features)
        audio = tf.sparse.to_dense(sample['audio'])
        # audio = tf.cast(audio, tf.float32)
        audio_len =  tf.cast(sample['audio_len'], tf.int32)
        audio = tf.reshape(audio, [audio_len])

        mel_shape = tf.cast(sample['mel_shape'], tf.int32)
        mel = tf.sparse.to_dense(sample['mel'])
        mel = tf.reshape(mel, [mel_shape[0], mel_shape[1]])
        speaker_id = tf.cast(sample['speaker_id'], tf.int32) if self._hparams.gin_channels > 0 else 0
        speaker_id = tf.reshape(speaker_id, [1])

        max_time_frames = self._hparams.max_time_steps // self._hparams.hop_size
        start = tf.random.uniform(1, 0, tf.shape(mel)[0] - max_time_frames, dtype=tf.int32)
        time_start = start[0] * self._hparams.hop_size
        audio = audio[time_start: time_start + max_time_frames * self._hparams.hop_size]
        mel = mel[start[0]: start[0] + max_time_frames, :]

        return mel, audio, speaker_id


    def _postprocess_batch(self, mels, audios, speaker_ids):
        return mels, audios, tf.squeeze(speaker_ids)

        
    def initialize(self, sess):
        # audio_filename, mel_filename, time_steps, N, speaker_id, text
        sess.run(self._train_iterator.initializer, feed_dict={
            self._filenames: [self._train_tfrecord]
        })

        sess.run(self._test_iterator.initializer, feed_dict={
            self._filenames: [self._test_tfrecord]
        })
