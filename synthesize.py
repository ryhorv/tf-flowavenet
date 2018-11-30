import tensorflow as tf
import os
from model import FloWaveNet
from hparams import hparams
import argparse
import numpy as np
import audio
from tqdm import tqdm

def get_model(hparams):
    with tf.variable_scope('vocoder'):
        with tf.name_scope('tower_0'):
#             z = tf.placeholder(tf.float32, shape=[None, None, 1], name='z')
            lc = tf.placeholder(tf.float32, shape=[None, None, hparams.num_mels])
            z = tf.random_normal([tf.shape(lc)[0], tf.shape(lc)[1] * audio.get_hop_size(hparams), 1]) * hparams.temp

            model = FloWaveNet(in_channel=1,
                               cin_channel=hparams.num_mels,
                               n_block=hparams.n_block,
                               n_flow=hparams.n_flow,
                               n_layer=hparams.n_layer,
                               affine=hparams.affine,
                               causal=hparams.causality, scope='FloWaveNet')

            predictions = model.reverse(z, lc)
            predictions = tf.squeeze(predictions)
            
            return predictions, lc
        
def synthesize(args, hparams):
    predictions, lc_phr = get_model(hparams)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    try:
        checkpoint_state = tf.train.get_checkpoint_state(args.saved_dir)

        if (checkpoint_state and checkpoint_state.model_checkpoint_path):
            print('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
            saver.restore(sess, checkpoint_state.model_checkpoint_path)

    except tf.errors.OutOfRangeError as e:
        print('Cannot restore checkpoint: {}'.format(e))
        return
    
    mel_filenames = [f for f in os.listdir(args.mels_dir) if f.endswith('.npy')]
    
    for mel_filename in tqdm(mel_filenames):
        mel_path = os.path.join(args.mels_dir, mel_filename)
        mel = np.load(mel_path)[np.newaxis, ...]
        
        result = sess.run(predictions, feed_dict={lc_phr: mel})
        audio_filename = mel_filename[:-4] + '.wav'
        audio_path = os.path.join(args.output_dir, audio_filename)
        audio.save_wav(result, audio_path, sr=hparams.sample_rate)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_dir', default='logs/pretrained/', help='Folder with model checkpoint')
    parser.add_argument('--mels_dir', default='mels/', help='folder to contain mels to synthesize audio from using the model')
    parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized audio files')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    synthesize(args, hparams)
    
if __name__ == '__main__':
    main()