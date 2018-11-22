import tensorflow as tf
import os
import time
from dataset import Dataset
from model import FloWaveNet
import utils
from hparams import hparams
import argparse


def get_optimizer(hparams, global_step):
    with tf.name_scope('optimizer'):
        learning_rate = tf.constant(0.001)
        learning_rate = tf.cond(tf.less(global_step, 200000), true_fn=lambda: learning_rate, false_fn=lambda: tf.constant(0.001 / 2))
        learning_rate = tf.cond(tf.less(global_step, 400000), true_fn=lambda: learning_rate, false_fn=lambda: tf.constant(0.001 / 4))
        learning_rate = tf.cond(tf.less(global_step, 600000), true_fn=lambda: learning_rate, false_fn=lambda: tf.constant(0.001 / 6))

        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer, learning_rate
    
def compute_gradients(loss, vars):
    with tf.name_scope('gradients'):
        grads = tf.gradients(loss, vars)
        with tf.name_scope('gradient_clipping'):
            grad_vars = []
            for grad, var in zip(grads, vars):
                if grad is not None:
                    name = grad.name
                    name = name.replace(':', '_')
                    clipped_grad = tf.clip_by_norm(grad, 1, name=name+'_norm')
                    grad_vars.append((clipped_grad, var))
                else:
                    grad_vars.append((grad, var))
                    
            return grad_vars


def get_train_model(dataset, hparams, global_step):
    tower_gradvars = []
    train_model = None
    train_losses = []
    train_predictd_wavs = None
    train_target_wavs = None

    for i in range(hparams.num_gpus):
        if hparams.num_gpus > 1:
            worker_device = '/gpu:%d' % i
            if hparams.ps_device_type == 'CPU':
                device_setter = utils.local_device_setter(worker_device=worker_device)
            elif hparams.ps_device_type == 'GPU':
                device_setter = utils.local_device_setter(
                    ps_device_type='gpu',
                    worker_device=worker_device,
                    ps_strategy=None)
        else:
            device_setter = '/gpu:0'

        with tf.variable_scope('vocoder', reuse=tf.AUTO_REUSE):  
            with tf.name_scope('tower_%d' % i) as name_scope:
                with tf.device(device_setter):
                    model = FloWaveNet(in_channel=1,
                                    cin_channel=hparams.num_mels,
                                    n_block=hparams.n_block,
                                    n_flow=hparams.n_flow,
                                    n_layer=hparams.n_layer,
                                    affine=hparams.affine,
                                    causal=hparams.causality)

                    log_p, logdet = model.forward(dataset.inputs[i], dataset.local_conditions[i])
                    
                    with tf.name_scope('loss'):
                        loss = -(log_p + logdet)


                    grad_vars = compute_gradients(loss, tf.trainable_variables())
                    tower_gradvars.append(grad_vars)

                    if i == 0:
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
                        train_model = model
                        train_losses = [-(log_p + logdet), log_p, logdet]
                        
#                         lc = dataset.local_conditions[i]
#                         target = dataset.inputs[i]
#                         z = tf.random_normal(tf.shape(target))
        
#                         train_predictd_wavs = train_model.reverse(z, lc)
#                         train_predictd_wavs = tf.squeeze(train_predictd_wavs)
        
#                         train_target_wavs = tf.squeeze(target)

    consolidation_device  = '/cpu:0' if hparams.ps_device_type == 'CPU' and hparams.num_gpus > 1 else '/gpu:0'
    with tf.device(consolidation_device):
        grad_vars = utils.average_gradients(tower_gradvars)
        optimizer, lr = get_optimizer(hparams, global_step)
            
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(grad_vars, global_step=global_step)

    return train_op, train_model, train_losses, lr, train_predictd_wavs, train_target_wavs, tower_gradvars[0]

def get_test_model(dataset, hparams):
    with tf.variable_scope('vocoder', reuse=tf.AUTO_REUSE):
        test_model = FloWaveNet(in_channel=1,
                                cin_channel=hparams.num_mels,
                                n_block=hparams.n_block,
                                n_flow=hparams.n_flow,
                                n_layer=hparams.n_layer,
                                affine=hparams.affine,
                                causal=hparams.causality)
        target = dataset.eval_inputs
        lc = dataset.eval_local_conditions
        log_p, logdet = test_model.forward(target, lc)
        
        with tf.name_scope('loss'):
            loss = -(log_p + logdet)
                
        losses = [loss, log_p, logdet]
        return losses, None, None
    
def get_summary_op(train_losses, test_losses, learning_rate, is_training, grad_vars):
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
    
    with tf.name_scope('max_grad_val'):
        max_abs_grads = []
        for grad, var in grad_vars:
            if grad is not None:
                train_summaries.append(tf.summary.histogram(grad.name, grad))
                max_abs_grads.append(tf.reduce_max(tf.abs(grad)))

        max_abs_value = tf.reduce_max(max_abs_grads)
            
    train_summaries.append(tf.summary.scalar('max_abs_grad', max_abs_value))
    
    
    train_op = tf.summary.merge(train_summaries)
    test_op = tf.summary.merge(test_summaries)
            
    return train_op, test_op
    

def train(log_dir, args, hparams, input_path):
    save_dir = os.path.join(log_dir, 'pretrained')
    train_logdir = os.path.join(log_dir, 'train')
    test_logdir = os.path.join(log_dir, 'test')
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(train_logdir, exist_ok=True)
    os.makedirs(test_logdir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, 'flowavenet_model.ckpt')
    input_path = os.path.join(args.base_dir, input_path)

    print('Checkpoint_path: {}'.format(checkpoint_path))
    print('Loading training data from: {}'.format(input_path))

    #Start by setting a seed for repeatability
    tf.set_random_seed(hparams.tf_random_seed)

    with tf.name_scope('dataset') as scope:
        dataset = Dataset(input_path, args.input_dir, hparams)

    #Set up model
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op, train_model, train_losses, lr, train_predictd_wavs, train_target_wavs, gradvars = get_train_model(dataset, hparams, global_step)
    test_losses, test_predicted_wavs, test_target_wavs = get_test_model(dataset, hparams)
    
#     _, init_ops = train_model.initialize(dataset.inputs[0], dataset.local_conditions[0])

    
    is_training = tf.placeholder(tf.bool, name='is_training')
    
    train_summary_op, test_summary_op = get_summary_op(train_losses, test_losses, lr, is_training, gradvars)
    
    # eval_summary = get_eval_summary_op(is_training, 
    #                     outputs['mel_outputs'], outputs['mel_targets'], outputs['alignments'], outputs['targets_lengths'], 
    #                     test_outputs['mel_outputs'], test_outputs['mel_targets'], test_outputs['alignments'], test_outputs['targets_lengths'], hparams)

    #book keeping
    step = 0
    saver = tf.train.Saver()

    print('FloWaveNet training set to a maximum of {} steps'.format(args.train_steps))
    
    #Memory allocation on the memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    #Train
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        test_writer = tf.summary.FileWriter(test_logdir)
            
        sess.run(tf.global_variables_initializer())
        #initializing dataset
        
        dataset.initialize(sess)
#         sess.run(init_ops)


        #saved model restoring
        if args.restore:
            # Restore saved model if the user requested it, default = True
            try:
                checkpoint_state = tf.train.get_checkpoint_state(save_dir)

                if (checkpoint_state and checkpoint_state.model_checkpoint_path):
                    print('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
                    saver.restore(sess, checkpoint_state.model_checkpoint_path)

            except tf.errors.OutOfRangeError as e:
                print('Cannot restore checkpoint: {}'.format(e))
        else:
            print('Starting new training!')

        # summary_writer.add_summary(sess.run(stats), step)
        # save_log(sess, step, model, plot_dir, wav_dir, hparams=hparams)
        # eval_step(sess, step, eval_model, eval_plot_dir, eval_wav_dir, summary_writer=summary_writer , hparams=model._hparams)

        # #Training loop
        # sess.graph.finalize()
        while step < args.train_steps:
            start_time = time.time()
            step, total_loss, log_p_loss, logdet_loss, opt = sess.run([global_step, train_losses[0], train_losses[1], train_losses[2], train_op])
            step_duration = (time.time() - start_time)

            message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, log_p={:.5f}, logdet={:.5f}]'.format(step, step_duration, total_loss, log_p_loss, logdet_loss)
            print(message, end='\r')                

            if step % args.summary_interval == 0:
                print('\nWriting summary at step {}'.format(step))
                train_writer.add_summary(sess.run(train_summary_op, feed_dict={is_training: True}), step)
                test_writer.add_summary(sess.run(test_summary_op, feed_dict={is_training: False}), step)

            if step % args.checkpoint_interval == 0 or step == args.train_steps:
                saver.save(sess, checkpoint_path, global_step=global_step)

        #     if step % args.eval_interval == 0:
        #         log('\nEvaluating at step {}'.format(step))
        #         train_writer.add_summary(sess.run(eval_summary, feed_dict={is_training: True}), step)
        #         test_writer.add_summary(sess.run(eval_summary, feed_dict={is_training: False}), step)

        return save_dir



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--input', default='training_data/train.txt')
    parser.add_argument('--input_dir', default='training_data/', help='folder to contain inputs sentences/targets')
    parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
    parser.add_argument('--summary_interval', type=int, default=500,
        help='Steps between running summary ops')
    parser.add_argument('--checkpoint_interval', type=int, default=5000,
        help='Steps between writing checkpoints')
    parser.add_argument('--eval_interval', type=int, default=2000,
        help='Steps between eval on test data')
    parser.add_argument('--train_steps', type=int, default=2000000, help='total number of model training steps')
    args = parser.parse_args()

    logdir = os.path.join(args.base_dir, 'logs')
    os.makedirs(logdir, exist_ok=True)
#     print('выаыв')
    train(logdir, args, hparams, args.input)
    
if __name__ == "__main__":
    main()
