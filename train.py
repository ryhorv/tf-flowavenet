import os
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

import tensorflow as tf
import time
from dataset import Dataset
from model import FloWaveNet
from hparams import hparams
import argparse
import numpy as np
from utils import fp16_dtype_getter, average_gradients
  
    
def get_optimizer(hparams, global_step):
    with tf.name_scope('optimizer'):
        learning_rate = tf.constant(0.001)
        learning_rate = tf.cond(tf.less(global_step, 200000), true_fn=lambda: learning_rate, false_fn=lambda: tf.constant(0.001 / 2))
        learning_rate = tf.cond(tf.less(global_step, 400000), true_fn=lambda: learning_rate, false_fn=lambda: tf.constant(0.001 / 4))
        learning_rate = tf.cond(tf.less(global_step, 600000), true_fn=lambda: learning_rate, false_fn=lambda: tf.constant(0.001 / 6))

        optimizer = tf.train.AdamOptimizer(learning_rate)
        
        return optimizer, learning_rate
        
        
def clip_gradients(grad_vars):
    with tf.name_scope('gradient_clipping'):      
        grads, variables = zip(*grad_vars)
        clipped_grads, global_norm = tf.clip_by_global_norm(grads, 1)
        grad_vars = list(zip(clipped_grads, variables))        
        return grad_vars, global_norm
    

def build_model(dataset, hparams, global_step, init):
    tower_gradvars = []
    train_model = None
    train_losses = []
    train_predictd_wavs = None
    train_target_wavs = None
    
    consolidation_device  = '/cpu:0' if hparams.ps_device_type == 'CPU' and hparams.num_gpus > 1 else '/gpu:0'
    for i in range(hparams.num_gpus):
        if hparams.num_gpus > 1:
            worker_device = '/gpu:%d' % i
            if hparams.ps_device_type == 'CPU':
                device_setter = tf.train.replica_device_setter(ps_tasks=1, worker_device=worker_device, ps_device="/cpu:0")
            elif hparams.ps_device_type == 'GPU':
                device_setter = tf.train.replica_device_setter(ps_tasks=1, worker_device=worker_device, ps_device="/gpu:0")
        else:
            device_setter = '/gpu:0'

        with tf.variable_scope('vocoder', reuse=tf.AUTO_REUSE, custom_getter=fp16_dtype_getter):  
            with tf.name_scope('tower_%d' % i) as name_scope:
                with tf.device(device_setter):
                    model = FloWaveNet(hparams, init=init)
                    log_p, logdet = model.forward(dataset.inputs[i], dataset.local_conditions[i], dataset.speaker_ids[i])
                    
                    with tf.name_scope('loss'):
                        loss = -(log_p + logdet)
                        
                    with tf.name_scope('gradients'):
                        variables = tf.trainable_variables()
                        scaled_loss = tf.scalar_mul(hparams.scale, loss)
                        grads = tf.gradients(scaled_loss, variables)
                        grad_vars = list(zip(grads, variables)) 
                        tower_gradvars.append(grad_vars)
                   
                    if i == 0:
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
                        train_model = model
                        train_losses = [loss, log_p, logdet]

    
    with tf.device(consolidation_device):
        grad_vars = average_gradients(tower_gradvars)
        grad_vars = [(tf.scalar_mul(1./hparams.scale, g), v) for g, v in grad_vars if g is not None]
        clipped_grad_vars, grad_global_norm = clip_gradients(grad_vars)
        optimizer, lr = get_optimizer(hparams, global_step)    
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(clipped_grad_vars, global_step=global_step)

    return train_op, train_model, train_losses, lr, grad_global_norm

def get_test_losses(model, dataset, hparams):
    log_p, logdet = model.forward(dataset.eval_inputs, dataset.eval_local_conditions, dataset.eval_speaker_ids)
    with tf.name_scope('loss'):
        loss = -(log_p + logdet)                
    
    losses = [loss, log_p, logdet]
    return losses
    
def get_summary_op(train_losses, test_losses, learning_rate, grad_global_norm, is_training):
    losses = tf.cond(is_training, true_fn=lambda: train_losses, false_fn=lambda: test_losses)
    train_summaries = []
    test_summaries = []

    total_loss = tf.summary.scalar('losses/total_loss', losses[0])
    train_summaries.append(total_loss)
    test_summaries.append(total_loss)

    log_p = tf.summary.scalar('losses/log_p', losses[1])
    train_summaries.append(log_p)
    test_summaries.append(log_p)

    logdet = tf.summary.scalar('losses/logdet', losses[2])
    train_summaries.append(logdet)
    test_summaries.append(logdet)

    train_summaries.append(tf.summary.scalar('learning_rate', learning_rate))
    train_summaries.append(tf.summary.scalar('gradient_global_norm', grad_global_norm))

    train_op = tf.summary.merge(train_summaries)
    test_op = tf.summary.merge(test_summaries)

    return train_op, test_op

def predict_random_samples(model, metadata_path, hparams):
    basedir = os.path.dirname(metadata_path)
    with open(metadata_path, 'rt', encoding='utf-8') as f:
        meta = f.read().strip().split('\n')
        meta = [m.split('|') for m in meta]
            
    max_time_frames = int(hparams.eval_max_time_steps // hparams.hop_size)
    random_index = np.random.choice(len(meta))
    sample = meta[random_index]
    wav = np.load(os.path.join(basedir, 'audios', sample[0]))[:max_time_frames*hparams.hop_size]
    lc = np.load(os.path.join(basedir, 'mels', sample[1]))[:max_time_frames]
        
    target_wavs = tf.convert_to_tensor(wav[np.newaxis, :, np.newaxis], dtype=tf.float32)
    lc = tf.convert_to_tensor(lc[np.newaxis, ...], dtype=tf.float32)
        
    z = tf.random_normal(target_wavs.shape) * hparams.temp
    speaker_ids = tf.constant([int(sample[3])], dtype=tf.int32)
        
    predicted_wavs = model.reverse(z, lc, speaker_ids)
    predicted_wavs = tf.squeeze(predicted_wavs, axis=-1)
        
    target_wavs = tf.squeeze(target_wavs, axis=-1)        
    return predicted_wavs, target_wavs

def get_eval_summary_op(model, metadata_path, hparams):
    predicted_wavs, target_wavs = predict_random_samples(model, metadata_path, hparams)
    
    summaries = []
    summaries.append(tf.summary.audio('predictions', tf.cast(predicted_wavs, tf.float32), sample_rate=hparams.sample_rate))
    summaries.append(tf.summary.audio('targets', tf.cast(target_wavs, tf.float32), sample_rate=hparams.sample_rate))

    summary_op = tf.summary.merge(summaries)
    return summary_op
    

def train(log_dir, args, hparams, input_path):
    tf.set_random_seed(hparams.tf_random_seed)
    save_dir = os.path.join(log_dir, 'pretrained')
    train_logdir = os.path.join(log_dir, 'train')
    test_logdir = os.path.join(log_dir, 'test')
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(train_logdir, exist_ok=True)
    os.makedirs(test_logdir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, 'flowavenet_model.ckpt')
    input_path = os.path.join(args.base_dir, input_path)
    train_tfrecord = os.path.join(args.base_dir, 'training_data/train.tfrecord')
    test_tfrecord = os.path.join(args.base_dir, 'training_data/test.tfrecord')
    metadata_filename = os.path.join(args.base_dir, 'training_data/train.txt')

    print('Checkpoint_path: {}'.format(checkpoint_path))
    print('Loading training data from: {}'.format(input_path))

    #Start by setting a seed for repeatability
    tf.set_random_seed(hparams.tf_random_seed)

    with tf.name_scope('dataset'):
        dataset = Dataset(train_tfrecord, test_tfrecord, hparams)

    #Set up model
    init = tf.placeholder_with_default(False, shape=None, name='init')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op, model, train_losses, lr, grad_global_norm = build_model(dataset, hparams, global_step, init)
    test_losses = get_test_losses(model, dataset, hparams)
    
    is_training = tf.placeholder(tf.bool, name='is_training')
    
    train_summary_op, test_summary_op = get_summary_op(train_losses, test_losses, lr, grad_global_norm, is_training)
    eval_summary_op = get_eval_summary_op(model, metadata_filename, hparams)

    step = 0
    saver = tf.train.Saver(var_list=tf.global_variables())

    print('FloWaveNet training set to a maximum of {} steps'.format(args.train_steps))
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # config.intra_op_parallelism_threads = 14
    # config.inter_op_parallelism_threads = 4

    #Train
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        test_writer = tf.summary.FileWriter(test_logdir)
            
        sess.run(tf.global_variables_initializer())
        
        #initializing dataset        
        dataset.initialize(sess)

        #saved model restoring
        if args.restore:
            # Restore saved model if the user requested it, default = True
            try:
                checkpoint_state = tf.train.get_checkpoint_state(save_dir)

                if (checkpoint_state and checkpoint_state.model_checkpoint_path):
                    print('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
                    saver.restore(sess, checkpoint_state.model_checkpoint_path)
                else:
                    print('Init ActNorm layer...', end='')
                    step, init_loss, _ = sess.run([global_step, train_losses[0], train_op], feed_dict={init: True})
                    print(" OK. Init loss: {:.5f}".format(init_loss))

            except tf.errors.OutOfRangeError as e:
                print('Cannot restore checkpoint: {}'.format(e))
        else:
            print('Starting new training!')
            print('Init ActNorm layer...', end='')
            step, init_loss, _ = sess.run([global_step, train_losses[0], train_op], feed_dict={init: True})
            print(" OK. Init loss: {:.5f}".format(init_loss))

        
        # Training loop
        while step < args.train_steps:
            try:
                start_time = time.time()
                step, total_loss, log_p_loss, logdet_loss, opt = sess.run([global_step, train_losses[0], train_losses[1], train_losses[2], train_op])
                step_duration = (time.time() - start_time)
                message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, log_p={:.5f}, logdet={:.5f}]'.format(step, step_duration, total_loss, log_p_loss, logdet_loss)
                print(message, end='\r')
            except tf.errors.InvalidArgumentError as e:
                print(e)
                print('Continue training')
                                    

            if step % args.summary_interval == 0:
                print('\nWriting summary at step {}'.format(step))
                train_writer.add_summary(sess.run(train_summary_op, feed_dict={is_training: True}), step)
                test_writer.add_summary(sess.run(test_summary_op, feed_dict={is_training: False}), step)
                
            if step % args.checkpoint_interval == 0 or step == args.train_steps:
                saver.save(sess, checkpoint_path, global_step=global_step)

            if step % args.eval_interval == 0:
                print('\nEvaluating at step {}'.format(step))
                train_writer.add_summary(sess.run(eval_summary_op, feed_dict={is_training: True}), step)
                train_writer.flush()

        return save_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--input', default='training_data/train.txt')
    parser.add_argument('--input_dir', default='training_data/', help='folder to contain inputs sentences/targets')
    parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
    parser.add_argument('--summary_interval', type=int, default=500,
        help='Steps between running summary ops')
    parser.add_argument('--checkpoint_interval', type=int, default=2000,
        help='Steps between writing checkpoints')
    parser.add_argument('--eval_interval', type=int, default=5000,
        help='Steps between eval on test data')
    parser.add_argument('--train_steps', type=int, default=2000000, help='total number of model training steps')
    args = parser.parse_args()

    logdir = os.path.join(args.base_dir, 'logs')
    os.makedirs(logdir, exist_ok=True)
    train(logdir, args, hparams, args.input)
    
if __name__ == "__main__":
    main()
