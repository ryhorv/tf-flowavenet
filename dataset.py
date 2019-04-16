import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
import multiprocessing


class Dataset:
    def __init__(self,  train_tfrecord, test_tfrecord, hparams):
        self._hparams = hparams
        self._train_tfrecord = train_tfrecord
        self._test_tfrecord = test_tfrecord      
        
        self._max_time_frames = self._hparams.max_time_steps // self._hparams.hop_size
        self._max_time_steps = self._max_time_frames * self._hparams.hop_size
        
        self._pad = 0.    
        n_cpu = multiprocessing.cpu_count()
        buffer_size = 64

        with tf.device('/cpu:0'):
            self._filenames = tf.placeholder(tf.string, shape=[None])
            self._num_workers = tf.placeholder(tf.int64, shape=None, name='num_workers')
            self._worker_id = tf.placeholder(tf.int64, shape=None, name='worker_id')
            
            dataset = tf.data.TFRecordDataset(self._filenames)
            dataset = dataset.shard(self._num_workers, self._worker_id)
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size))
            dataset = dataset.map(self._load_sample, n_cpu)
            # dataset = dataset.apply(tf.data.experimental.ignore_errors())
            dataset = dataset.batch(self._hparams.batch_size)
            dataset = dataset.prefetch(1)

        self._train_iterators = []
        self.inputs = []
        self.local_conditions = []
        self.speaker_ids = []
        for _ in range(hparams.num_gpus):
            train_iterator = dataset.make_initializable_iterator()
            with tf.device(None):
                train_batch = train_iterator.get_next()
                self.local_conditions.append(train_batch[0])
                self.inputs.append(train_batch[1])
                self.speaker_ids.append(train_batch[2])
                self._train_iterators.append(train_iterator)
                
        self._test_iterator = dataset.make_initializable_iterator()
        test_batch = self._test_iterator.get_next()
        self.eval_local_conditions = test_batch[0]
        self.eval_inputs = test_batch[1]
        self.eval_speaker_ids = test_batch[2]
        
        if self._hparams.gin_channels <= 0:
            self.speaker_ids = [None] * hparams.num_gpus
            self.eval_speaker_ids = None

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
        audio = tf.reshape(audio, [audio_len, 1])

        mel_shape = tf.cast(sample['mel_shape'], tf.int32)
        mel = tf.sparse.to_dense(sample['mel'])
        mel = tf.reshape(mel, [mel_shape[0], mel_shape[1]])
        speaker_id = tf.cast(sample['speaker_id'], tf.int32) if self._hparams.gin_channels > 0 else 0

        
        start = tf.random.uniform([1], 0, tf.shape(mel)[0] - self._max_time_frames, dtype=tf.int32)
        time_start = start[0] * self._hparams.hop_size
        audio = audio[time_start:time_start + self._max_time_steps]
        mel = mel[start[0]:start[0] + self._max_time_frames]
        
        audio.set_shape([None, 1])
        mel.set_shape([None, self._hparams.num_mels])
        
        if self._hparams.dtype == tf.float16:
            audio = tf.cast(audio, tf.float16)
            mel = tf.cast(mel, tf.float16)

        return mel, audio, speaker_id


    def _postprocess_batch(self, mels, audios, speaker_ids):
        return mels, audios, tf.squeeze(speaker_ids)

        
    def initialize(self, sess):
        # audio_filename, mel_filename, time_steps, N, speaker_id, text
        for i, train_iterator in enumerate(self._train_iterators):
            sess.run(train_iterator.initializer, feed_dict={
                self._filenames: [self._train_tfrecord],
                self._num_workers: self._hparams.num_gpus,
                self._worker_id: i
            })

        sess.run(self._test_iterator.initializer, feed_dict={
            self._filenames: [self._test_tfrecord],
            self._num_workers: 1,
            self._worker_id: 0
        })
