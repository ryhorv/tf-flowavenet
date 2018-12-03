import tensorflow as tf
from modules import WaveNet
from math import log, pi
from convolutional import Conv2DTranspose

logabs = lambda x: tf.log(tf.abs(x))

class ActNorm:
    def __init__(self, in_channel, logdet=True, scope='ActNorm'):
        with tf.variable_scope(scope) as vs:
            self._vs = vs
            self._scope = scope
            self._loc = tf.get_variable('loc', shape=[1, 1, in_channel], initializer=tf.initializers.zeros())
            self._scale = tf.get_variable('scale', shape=[1, 1, in_channel], initializer=tf.initializers.ones())
            
#             self._is_initialized = tf.get_variable('is_initialized', shape=None, name=)
#             self._is_initialized = tf.Variable(False, trainable=False, name='is_initialized')

            self._logdet = logdet

    def initialize(self, x):
        mean, var = tf.nn.moments(x, axes=[0, 1], keep_dims=True)
        mean = tf.stop_gradient(mean)
        std = tf.stop_gradient(tf.sqrt(var))
                
        init_op = []
        init_op.append(self._loc.assign(-mean))
        init_op.append(self._scale.assign(1 / (std + 1e-6)))
        init_op.append(self._is_initialized.assign(True))
        
        return self.forward(x), init_op
        
#         _init_op = []
#         _init_op.append(self._loc.assign(self._loc))
#         _init_op.append(self._scale.assign(self._scale))
#         _init_op.append(self._is_initialized.assign(self._is_initialized))
#         return tf.cond(self._is_initialized, true_fn=lambda: _init_op, false_fn=lambda: init_op)


    def forward(self, x):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
#                 with tf.control_dependencies(self.initialize(x)):
                log_abs = logabs(self._scale)
                logdet = tf.reduce_mean(log_abs)

                if self._logdet:
                    return self._scale * (x + self._loc), logdet
                else:
                    return self._scale * (x + self._loc)

    def reverse(self, output):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                
                return output / self._scale - self._loc

    def __call__(self, x):
        return self.forward(x)


class AffineCoupling:
    def __init__(self, in_channel, cin_channel, filter_size=256, num_layer=6, affine=True, causal=False, scope='AffineCoupling'):
        with tf.variable_scope(scope) as vs:
            self._vs = vs
            self._scope = scope
            self._in_channel = in_channel
            self._affine = affine
            self._net = WaveNet(in_channels=in_channel // 2, out_channels=in_channel if self._affine else in_channel // 2,
                            num_blocks=1, num_layers=num_layer, residual_channels=filter_size,
                            gate_channels=filter_size, skip_channels=filter_size,
                            kernel_size=3, cin_channels=cin_channel // 2, causal=causal)
                            

    def forward(self, x, c=None):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                in_a, in_b = tf.split(x, axis=2, num_or_size_splits=2)
                # in_a.set_shape([None, None, self._in_channel//2])
                # in_b.set_shape([None, None, self._in_channel - (self._in_channel//2)])

                c_a, c_b = tf.split(c, axis=2, num_or_size_splits=2)
                # c_a.set_shape([None, None, tf.shape(c)[2] // 2])
                # c_b.set_shape([None, None, tf.shape(c)[2] - (tf.shape(c)[2] // 2)])

                if self._affine:
                    log_s, t = tf.split(self._net(in_a, c_a), axis=2, num_or_size_splits=2)                    
                    out_b = (in_b - t) * tf.exp(-log_s)
                    logdet = tf.reduce_mean(-log_s) / 2
                else:
                    net_out = self._net(in_a, c_a)
                    out_b = in_b + net_out
                    logdet = None

                return tf.concat([in_a, out_b], 2), logdet

    def reverse(self, output, c=None):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                out_a, out_b = tf.split(output, axis=2, num_or_size_splits=2)
                c_a, c_b = tf.split(c, axis=2, num_or_size_splits=2)

                if self._affine:
                    # log_s, t = self.net(out_a, c_a).chunk(2, 1)
                    log_s, t = tf.split(self._net(out_a, c_a), axis=2, num_or_size_splits=2)
                    in_b = out_b * tf.exp(log_s) + t
                else:
                    net_out = self._net(out_a, c_a)
                    in_b = out_b - net_out

                return tf.concat([out_a, in_b], 2)

    def __call__(self, x, c=None):
        return self.forward(x, c)

def change_order(x, c=None):
    x_a, x_b = tf.split(x, axis=2, num_or_size_splits=2)
    c_a, c_b = tf.split(c, axis=2, num_or_size_splits=2)

    return tf.concat([x_b, x_a], 2), tf.concat([c_b, c_a], 2)

class Flow:
    def __init__(self, in_channel, cin_channel, filter_size, num_layer, affine=True, causal=False, scope='Flow'):
        with tf.variable_scope(scope) as vs:
            self._vs = vs
            self._scope = scope
            self._actnorm = ActNorm(in_channel)
            self._coupling = AffineCoupling(in_channel, cin_channel, filter_size=filter_size,
                                       num_layer=num_layer, affine=affine, causal=causal)
            
    def initialize(self, x, c):
        [out, logdet], init_ops = self._actnorm.initialize(x)
        out, det = self._coupling(out, c)
        out, c = change_order(out, c)

        if det is not None:
            logdet = logdet + det
            
        return [out, c, logdet], init_ops

    def forward(self, x, c=None):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                out, logdet = self._actnorm(x)
                out, det = self._coupling(out, c)
                out, c = change_order(out, c)

                if det is not None:
                    logdet = logdet + det

                return out, c, logdet

    def reverse(self, output, c=None):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                output, c = change_order(output, c)
                x = self._coupling.reverse(output, c)
                x = self._actnorm.reverse(x)
                return x, c

    def __call__(self, x, c=None):
        return self.forward(x, c)

class Block:
    def __init__(self, in_channel, cin_channel, n_flow, n_layer, affine=True, causal=False, scope='Block'):
        with tf.variable_scope(scope) as vs:
            self._vs = vs
            self._scope = scope
            squeeze_dim = in_channel * 2
            squeeze_dim_c = cin_channel * 2

            self._flows = []
            for i in range(n_flow):
                self._flows.append(Flow(squeeze_dim, squeeze_dim_c, filter_size=256, num_layer=n_layer, affine=affine,
                                    causal=causal, scope='Flow_%d' % i))
                
    def initialize(self, x, c):
        shape = tf.shape(x)
                
        with tf.name_scope('squeeze_x'):
            squeezed_x = tf.reshape(x, [shape[0], shape[1] // 2, 2, x.shape[2]])
            squeezed_x = tf.transpose(squeezed_x, [0, 1, 3, 2])
            out = tf.reshape(squeezed_x, [shape[0], shape[1] // 2, 2 * x.shape[2]])
                    
        with tf.name_scope('squeeze_c'):
            squeezed_c = tf.reshape(c, [shape[0], shape[1] // 2, 2, c.shape[2]])
            squeezed_c = tf.transpose(squeezed_c, [0, 1, 3, 2])
            c = tf.reshape(squeezed_c, [shape[0], shape[1] // 2, 2 * c.shape[2]])
                
        init_ops = []
        logdet = []
        for flow in self._flows:   
            [out, c, det], new_init_ops = flow.initialize(out, c)
            init_ops += new_init_ops
            logdet.append(det)
                    
        logdet = tf.add_n(logdet)
                
        return [out, c, logdet], init_ops
                
                

    def forward(self, x, c):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                
                shape = tf.shape(x)
                with tf.name_scope('squeeze_x'):
                    squeezed_x = tf.reshape(x, [shape[0], shape[1] // 2, 2, x.shape[2]])
                    squeezed_x = tf.transpose(squeezed_x, [0, 1, 3, 2])
                    out = tf.reshape(squeezed_x, [shape[0], shape[1] // 2, 2 * x.shape[2]])

                with tf.name_scope('squeeze_c'):
                    squeezed_c = tf.reshape(c, [shape[0], shape[1] // 2, 2, c.shape[2]])
                    squeezed_c = tf.transpose(squeezed_c, [0, 1, 3, 2])
                    c = tf.reshape(squeezed_c, [shape[0], shape[1] // 2, 2 * c.shape[2]])

                logdet = []
                for flow in self._flows:
                    out, c, det = flow(out, c)
                    logdet.append(det)

                logdet = tf.add_n(logdet)  
                return out, c, logdet

    def reverse(self, output, c):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                x = output

                for flow in self._flows[::-1]:
                    x, c = flow.reverse(x, c)

                shape = tf.shape(x)

                with tf.name_scope('unsqueezed_x'):
                    unsqueezed_x = tf.reshape(x, [shape[0], shape[1], x.shape[2] // 2, 2])
                    unsqueezed_x = tf.transpose(unsqueezed_x, [0, 1, 3, 2])
                    unsqueezed_x = tf.reshape(unsqueezed_x, [shape[0], shape[1] * 2, x.shape[2] // 2])

                with tf.name_scope('unsqueezed_c'):
                    unsqueezed_c = tf.reshape(c, [shape[0], shape[1], c.shape[2] // 2, 2])
                    unsqueezed_c = tf.transpose(unsqueezed_c, [0, 1, 3, 2])
                    unsqueezed_c = tf.reshape(unsqueezed_c, [shape[0], shape[1] * 2, c.shape[2] // 2])

                return unsqueezed_x, unsqueezed_c

    def __call__(self, x, c):
        return self.forward(x, c)

class FloWaveNet:
    def __init__(self, in_channel, cin_channel, n_block, n_flow, n_layer, affine=True, causal=False, scope='FloWaveNet'):
        with tf.variable_scope(scope) as vs:
            self._vs = vs
            self._scope = scope
            self._blocks = []
            self._n_block = n_block
            for i in range(self._n_block):
                self._blocks.append(Block(in_channel, cin_channel, n_flow, n_layer, affine=affine,
                                        causal=causal, scope='Block_%d' % i))
                in_channel *= 2
                cin_channel *= 2

            self._upsample_conv = []
            for s in [16, 16]:
                convt = Conv2DTranspose(filters=1, 
                                        kernel_size=(2 * s, 3), 
                                        padding='same', 
                                        strides=(s, 1), 
                                        activation=lambda x: tf.nn.leaky_relu(x, 0.4),
                                        kernel_initializer=tf.initializers.he_uniform(),
                                        bias_initializer=tf.initializers.zeros())
#                 convt = Conv2DTranspose(filters=1, 
#                                         kernel_size=(2 * s, 3), 
#                                         padding='same', 
#                                         strides=(s, 1), 
#                                         activation=lambda x: tf.nn.leaky_relu(x, 0.4),
#                                         kernel_initializer=tf.initializers.constant(0.001),
#                                         bias_initializer=tf.initializers.constant(0.001))
                self._upsample_conv.append(convt) 
                
                
    def initialize(self, x, c):
        logdet = []
        out = x
        c = self.upsample(c)
                
        init_ops = []
        for block in self._blocks:
            [out, c, logdet_new], new_init_ops  = block.initialize(out, c)
            init_ops += new_init_ops
            logdet.append(logdet_new)
                    
        logdet = tf.add_n(logdet)
        log_p = tf.reduce_mean(0.5 * (- log(2.0 * pi) - tf.pow(out, 2)))
        return [log_p, logdet], init_ops


    def forward(self, x, c):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                
                logdet = []
                out = x
                c = self.upsample(c)
                for block in self._blocks:
                    out, c, logdet_new = block(out, c)
                    logdet.append(logdet_new)

                logdet = tf.add_n(logdet)
                log_p = tf.reduce_mean(0.5 * (- log(2.0 * pi) - tf.pow(out, 2)))
                return log_p, logdet

    def reverse(self, z, c):  
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                # c = tf.cond(tf.equal(tf.shape(z)[1], tf.shape(c)[1]), 
                #             true_fn=lambda: c, 
                #             false_fn=lambda: self.upsample(c))
                c = self.upsample(c)
                x = z

                x_channels = 1
                c_channels = 80
                for _ in range(self._n_block):
                    # b_size, T, _ = tf.shape(x)
                    shape = tf.shape(x)
                    # x_channels = x.shape[2]
                    x = tf.reshape(x, [shape[0], shape[1] // 2, 2, x_channels])
                    x = tf.transpose(x, [0, 1, 3, 2])
                    x = tf.reshape(x, [shape[0], shape[1] // 2, 2 * x_channels])

                    # c_channels = c.shape[2]
                    c = tf.reshape(c, [shape[0], shape[1] // 2, 2, c_channels])
                    c = tf.transpose(c, [0, 1, 3, 2])
                    c = tf.reshape(c, [shape[0], shape[1]  // 2, 2 * c_channels])
                    c_channels = c_channels * 2
                    x_channels = x_channels * 2

                for i, block in enumerate(self._blocks[::-1]):
                    x, c = block.reverse(x, c)
                return x

    def check_recon(self, x, c):
        with tf.name_scope('check_recon'):
            c = self.upsample(c)
            out = x
            for block in self._blocks:
                out, c, _ = block(out, c)
            for i, block in enumerate(self._blocks[::-1]):
                out, c = block.reverse(out, c)
            return out

    def upsample(self, c):
        with tf.name_scope('upsample'):
            c = tf.expand_dims(c, 3)
            for f in self._upsample_conv:
                c = f(c)
            c = tf.squeeze(c, 3)
            return c