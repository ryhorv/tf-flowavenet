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


def get_train_model(dataset, hparams, global_step):
    tower_gradvars = []
    train_model = None
    train_loss = None

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

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):  
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
                    loss = -(log_p + logdet)

                    vars = tf.trainable_variables()
                    grads = tf.gradients(loss, vars)

                    with tf.name_scope('gradient_clipping'):
                        grad_vars = []
                        for grad, var in zip(grads, vars):
                            if grad is not None:
                                clipped_grad = tf.clip_by_norm(grad, 1)
                                grad_vars.append((clipped_grad, var))
                            else:
                                print(var)
                                grad_vars.append((grad, var))

                    tower_gradvars.append(grad_vars)

                    if i == 0:
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
                        train_model = model
                        train_loss = loss

    consolidation_device  = '/cpu:0' if hparams.ps_device_type == 'CPU' and hparams.num_gpus > 1 else '/gpu:0'
    with tf.device(consolidation_device):
        grad_vars = utils.average_gradients(tower_gradvars)
        optimizer, lr = get_optimizer(hparams, global_step)
            
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(grad_vars, global_step=global_step)


    return train_op, train_model, train_loss, lr

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
    # tf.set_random_seed(hparams.tacotron_random_seed)

    with tf.name_scope('dataset') as scope:
        dataset = Dataset(input_path, args.input_dir, hparams)

    #Set up model
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op, train_model, train_loss, l = get_train_model(dataset, hparams, global_step)

    
    # is_training = tf.placeholder(tf.bool, name='is_training')
    
    # train_summary_op, test_summary_op = get_summary_op(is_training, losses, test_losses, learning_rate)
    
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
        try:
            train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
            # test_writer = tf.summary.FileWriter(test_logdir)
            
            sess.run(tf.global_variables_initializer())

            #saved model restoring
            if args.restore:
                # Restore saved model if the user requested it, default = True
                try:
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)

                    if (checkpoint_state and checkpoint_state.model_checkpoint_path):
                        print('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path), slack=True)
                        saver.restore(sess, checkpoint_state.model_checkpoint_path)
                        # load_averaged_model(sess, sh_saver, checkpoint_state.model_checkpoint_path)

                except tf.errors.OutOfRangeError as e:
                    print('Cannot restore checkpoint: {}'.format(e), slack=True)
            else:
                print('Starting new training!', slack=True)

            #initializing dataset
            dataset.initialize(sess)

            # summary_writer.add_summary(sess.run(stats), step)
            # save_log(sess, step, model, plot_dir, wav_dir, hparams=hparams)
            # eval_step(sess, step, eval_model, eval_plot_dir, eval_wav_dir, summary_writer=summary_writer , hparams=model._hparams)

            # #Training loop
            # sess.graph.finalize()
            while step < args.train_steps:
                start_time = time.time()
                step, loss, opt = sess.run([global_step, train_loss, train_op])
                step_duration = (time.time() - start_time)

                message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}]'.format(step, step_duration, loss)
                print(message, end='\r')
                

            #     if step % args.summary_interval == 0:
            #         log('\nWriting summary at step {}'.format(step))
            #         train_writer.add_summary(sess.run(train_summary_op, feed_dict={is_training: True}), step)
            #         test_writer.add_summary(sess.run(test_summary_op, feed_dict={is_training: False}), step)

                if step % args.checkpoint_interval == 0 or step == args.train_steps:
                    saver.save(sess, checkpoint_path, global_step=global_step)

            #     if step % args.eval_interval == 0:
            #         log('\nEvaluating at step {}'.format(step))
            #         train_writer.add_summary(sess.run(eval_summary, feed_dict={is_training: True}), step)
            #         test_writer.add_summary(sess.run(eval_summary, feed_dict={is_training: False}), step)

            # log('Tacotron training complete after {} global steps'.format(args.tacotron_train_steps), slack=True)
            return save_dir

        except Exception as e:
            print('Exiting due to exception: {}'.format(e), slack=True)
            # traceback.print_exc()



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
