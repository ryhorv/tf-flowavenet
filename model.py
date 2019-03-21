import tensorflow as tf
from modules import WaveNet
from math import log, pi
from convolutional import Conv2DTranspose


class ActNorm:
    """
    This layer is implemented based on the implementation from the tensor2tensor library 
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/glow_ops_test.py
    """
    def __init__(self, in_channel, logdet=True, init=False, logscale=3., scope='ActNorm', training_dtype=tf.float32):
        with tf.variable_scope(scope) as vs:
            self._vs = vs
            self._scope = scope
            self._logscale = logscale
            self._in_channel = in_channel
            self._init = init
            self._logdet = logdet
            self._training_dtype = training_dtype
        
    def assign(self, w, initial_value):
        if initial_value.dtype != w.dtype:
            initial_value = tf.cast(initial_value, dtype=w.dtype)

        w = w.assign(initial_value)
        with tf.control_dependencies([w]):
            return w

    def get_variable_ddi(self, name, shape, initial_value, dtype=tf.float32, trainable=True, init=False):
        """Wrapper for data-dependent initialization."""
        # If init is a tensor bool, w is returned dynamically.
        w = tf.get_variable(name, shape=shape, dtype=dtype, initializer=None, trainable=trainable)
        if isinstance(init, bool):
            if init:
                result = self.assign(w, initial_value)
            result = w
        else:
            result = tf.cond(init, lambda: self.assign(w, initial_value), lambda: w)

        return tf.cast(result, dtype=self._training_dtype) if result.dtype != self._training_dtype else result
        
    def actnorm_center(self, x, reverse=False, init=False):
        """Add a bias to x.
        Initialize such that the output of the first minibatch is zero centered
        per channel.
        Args:
            name: scope
            x: 3-D Tensor.
            reverse: Forward or backward operation.
            init: data-dependent initialization.
        Returns:
            x_center: (x + b), if reverse is True and (x - b) otherwise.
      """
        x_mean = tf.reduce_mean(x, axis=[0, 1], keepdims=True)
        b = self.get_variable_ddi('b', [1, 1, self._in_channel], initial_value=-x_mean, init=init)
        if not reverse:
            x += b
        else:
            x -= b
        return x
        
    def actnorm_scale(self, x, logscale_factor=3., reverse=False, init=False):
        """Per-channel scaling of x."""
        x_var = tf.reduce_mean(x**2, axis=[0, 1], keepdims=True)
        logdet_factor = 1
        var_shape = (1, 1, self._in_channel)
        
        init_value = tf.log(1.0 / (tf.sqrt(x_var) + 1e-7)) / logscale_factor
        logs = self.get_variable_ddi('logs', var_shape, initial_value=init_value, init=init)
        logs = logs * logscale_factor

        # Function and reverse function.
        if not reverse:
            x = x * tf.exp(logs)
        else:
            x = x * tf.exp(-logs)

        # Objective calculation, h * w * sum(log|s|)
        dlogdet = tf.reduce_mean(logs) * logdet_factor
        if reverse:
            dlogdet *= -1
        return x, dlogdet


    def forward(self, x):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                x = self.actnorm_center(x, reverse=False, init=self._init)
                x, objective = self.actnorm_scale(x, reverse=False, init=self._init)
                if self._logdet:
                    return x, objective
                else:
                    return x
                    

    def reverse(self, x):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                output, objective = self.actnorm_scale(x, reverse=True, init=self._init)
                output = self.actnorm_center(output, reverse=True, init=self._init)
                return output

    def __call__(self, x):
        return self.forward(x)


class AffineCoupling:
    def __init__(self, in_channel, cin_channel, filter_size=256, num_layer=6, affine=True, causal=False, scope='AffineCoupling', training_dtype=tf.float32):
        with tf.variable_scope(scope) as vs:
            self._vs = vs
            self._scope = scope
            self._in_channel = in_channel
            self._affine = affine
            self._net = WaveNet(in_channels=in_channel // 2, out_channels=in_channel if self._affine else in_channel // 2,
                            num_blocks=1, num_layers=num_layer, residual_channels=filter_size,
                            gate_channels=filter_size, skip_channels=filter_size,
                            kernel_size=3, cin_channels=cin_channel // 2, causal=causal, training_dtype=training_dtype)
                            

    def forward(self, x, c, g=None):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                in_a, in_b = tf.split(x, axis=2, num_or_size_splits=2)
                c_a, c_b = tf.split(c, axis=2, num_or_size_splits=2)

                if g is not None:
                    g_a, g_b = tf.split(g, axis=2, num_or_size_splits=2)
                else:
                    g_a = None

                if self._affine:
                    log_s, t = tf.split(self._net(in_a, c_a, g_a), axis=2, num_or_size_splits=2)                    
                    out_b = (in_b - t) * tf.exp(-log_s)
                    logdet = tf.reduce_mean(-log_s) / 2
                else:
                    net_out = self._net(in_a, c_a, g_a)
                    out_b = in_b + net_out
                    logdet = None

                return tf.concat([in_a, out_b], 2), logdet

    def reverse(self, output, c, g=None):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                out_a, out_b = tf.split(output, axis=2, num_or_size_splits=2)
                c_a, c_b = tf.split(c, axis=2, num_or_size_splits=2)

                if g is not None:
                    g_a, g_b = tf.split(g, axis=2, num_or_size_splits=2)
                else:
                    g_a = None

                if self._affine:
                    log_s, t = tf.split(self._net(out_a, c_a, g_a), axis=2, num_or_size_splits=2)
                    in_b = out_b * tf.exp(log_s) + t
                else:
                    net_out = self._net(out_a, c_a, g_a)
                    in_b = out_b - net_out

                return tf.concat([out_a, in_b], 2)

    def __call__(self, x, c, g=None):
        return self.forward(x, c, g)

def change_order(x, c, g=None):
    x_a, x_b = tf.split(x, axis=2, num_or_size_splits=2)
    c_a, c_b = tf.split(c, axis=2, num_or_size_splits=2)

    if g is not None:
        g_a, g_b = tf.split(g, axis=2, num_or_size_splits=2)
        return tf.concat([x_b, x_a], 2), tf.concat([c_b, c_a], 2), tf.concat([g_b, g_a], 2)

    return tf.concat([x_b, x_a], 2), tf.concat([c_b, c_a], 2), None

class Flow:
    def __init__(self, in_channel, cin_channel, filter_size, num_layer, init, affine=True, causal=False, scope='Flow', training_dtype=tf.float32):
        with tf.variable_scope(scope) as vs:
            self._vs = vs
            self._scope = scope
            self._actnorm = ActNorm(in_channel, init=init, training_dtype=training_dtype)
            self._coupling = AffineCoupling(in_channel, cin_channel, filter_size=filter_size,
                                       num_layer=num_layer, affine=affine, causal=causal, training_dtype=training_dtype)

    def forward(self, x, c, g=None):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                out, logdet = self._actnorm(x)
                out, det = self._coupling(out, c, g)
                out, c, g = change_order(out, c, g)
                if det is not None:
                    logdet = logdet + det

                return out, c, g, logdet

    def reverse(self, output, c, g=None):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                output, c, g = change_order(output, c, g)
                x = self._coupling.reverse(output, c, g)
                x = self._actnorm.reverse(x)
                return x, c, g

    def __call__(self, x, c, g=None):
        return self.forward(x, c, g)

class Block:
    def __init__(self, in_channel, cin_channel, n_flow, n_layer, init, affine=True, causal=False, scope='Block', training_dtype=tf.float32):
        with tf.variable_scope(scope) as vs:
            self._vs = vs
            self._scope = scope
            squeeze_dim = in_channel * 2
            squeeze_dim_c = cin_channel * 2

            self._flows = []
            for i in range(n_flow):
                self._flows.append(Flow(squeeze_dim, squeeze_dim_c, init=init, filter_size=256, num_layer=n_layer, affine=affine,
                                    causal=causal, scope='Flow_%d' % i, training_dtype=training_dtype))
                

    def forward(self, x, c, g=None):
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

                if g is not None:
                    with tf.name_scope('squeeze_g'):
                        squeezed_g = tf.reshape(g, [shape[0], shape[1] // 2, 2, g.shape[2]])
                        squeezed_g = tf.transpose(squeezed_g, [0, 1, 3, 2])
                        g = tf.reshape(squeezed_g, [shape[0], shape[1] // 2, 2 * g.shape[2]])

                logdet = []
                for flow in self._flows:
                    out, c, g, det = flow(out, c, g)
                    logdet.append(det)

                logdet = tf.add_n(logdet)  
                return out, c, g, logdet

    def reverse(self, output, c, g=None):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                x = output

                for flow in self._flows[::-1]:
                    x, c, g = flow.reverse(x, c, g)

                shape = tf.shape(x)

                with tf.name_scope('unsqueezed_x'):
                    unsqueezed_x = tf.reshape(x, [shape[0], shape[1], x.shape[2] // 2, 2])
                    unsqueezed_x = tf.transpose(unsqueezed_x, [0, 1, 3, 2])
                    unsqueezed_x = tf.reshape(unsqueezed_x, [shape[0], shape[1] * 2, x.shape[2] // 2])

                with tf.name_scope('unsqueezed_c'):
                    unsqueezed_c = tf.reshape(c, [shape[0], shape[1], c.shape[2] // 2, 2])
                    unsqueezed_c = tf.transpose(unsqueezed_c, [0, 1, 3, 2])
                    unsqueezed_c = tf.reshape(unsqueezed_c, [shape[0], shape[1] * 2, c.shape[2] // 2])

                if g is not None:
                    with tf.name_scope('unsqueezed_g'):
                        unsqueezed_g = tf.reshape(g, [shape[0], shape[1], g.shape[2] // 2, 2])
                        unsqueezed_g = tf.transpose(unsqueezed_g, [0, 1, 3, 2])
                        unsqueezed_g = tf.reshape(unsqueezed_g, [shape[0], shape[1] * 2, g.shape[2] // 2])
                else:
                    unsqueezed_g = None
                    
                return unsqueezed_x, unsqueezed_c, unsqueezed_g

    def __call__(self, x, c, g=None):
        return self.forward(x, c, g)

class FloWaveNet:
    def __init__(self, hparams, init=False, scope='FloWaveNet'):
        with tf.variable_scope(scope) as vs:
            self._vs = vs
            self._scope = scope
            self._blocks = []
            self._n_block = hparams.n_block
            self._cin_channels = hparams.num_mels
            self._hparams = hparams
            self._dtype = hparams.dtype

            in_channels = 1
            cin_channels = self._cin_channels
            for i in range(self._n_block):
                self._blocks.append(Block(in_channels, cin_channels, hparams.n_flow, hparams.n_layer, init=init, affine=hparams.affine,
                                        causal=hparams.causality, scope='Block_%d' % i, training_dtype=self._dtype))
                in_channels *= 2
                cin_channels *= 2

            self._upsample_conv = []
            for s in hparams.upsample_scales:
                convt = Conv2DTranspose(filters=1, 
                                        kernel_size=(2 * s, 3), 
                                        padding='same', 
                                        strides=(s, 1), 
                                        activation=lambda x: tf.nn.leaky_relu(x, 0.4),
                                        kernel_initializer=tf.initializers.he_uniform(),
                                        bias_initializer=tf.initializers.zeros())

                self._upsample_conv.append(convt) 

            if hparams.gin_channels > 0:
                self.speaker_embeddings = tf.get_variable('speaker_embeddings', [hparams.n_speakers, hparams.gin_channels], dtype=tf.float32)
                
                
    def forward(self, x, c, g=None):
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                if g is None and self._hparams.gin_channels > 0:
                    raise ValueError('g is None')
                    
                x = tf.cast(x, dtype=self._dtype) if x.dtype != self._dtype else x
                c = tf.cast(c, dtype=self._dtype) if c.dtype != self._dtype else c

                logdet = []
                out = x
                c = self.upsample(c)

                if g is not None and self._hparams.gin_channels > 0:
                    g_embeddings = tf.nn.embedding_lookup(self.speaker_embeddings, g)
                    g_embeddings = tf.cast(g_embeddings, dtype=self._dtype) if g_embeddings.dtype != self._dtype else g_embeddings
                    g_embeddings = tf.expand_dims(g_embeddings, axis=1)
                    g_embeddings = tf.tile(g_embeddings, (1, tf.shape(c)[1], 1))
                else:
                    g_embeddings = None
                    
                for block in self._blocks:
                    out, c, g_embeddings, logdet_new = block(out, c, g_embeddings)
                    logdet.append(logdet_new)

                logdet = tf.add_n(logdet)
                log_p = tf.reduce_mean(0.5 * (- log(2.0 * pi) - tf.pow(out, 2)))
                
                logdet = tf.cast(logdet, dtype=tf.float32)
                log_p = tf.cast(log_p, dtype=tf.float32)
                return log_p, logdet

            
    def reverse(self, z, c, g=None):  
        with tf.variable_scope(self._vs, auxiliary_name_scope=False) as vs1:
            with tf.name_scope(vs1.original_name_scope):
                if g is None and self._hparams.gin_channels > 0:
                    raise ValueError('g is None')

                z = tf.cast(z, dtype=self._dtype) if z.dtype != self._dtype else z
                c = tf.cast(c, dtype=self._dtype) if c.dtype != self._dtype else c

                c = self.upsample(c)

                if g is not None and self._hparams.gin_channels > 0:
                    g_embeddings = tf.nn.embedding_lookup(self.speaker_embeddings, g)
                    g_embeddings = tf.cast(g_embeddings, dtype=self._dtype) if g_embeddings.dtype != self._dtype else g_embeddings
                    g_embeddings = tf.expand_dims(g_embeddings, axis=1)
                    g_embeddings = tf.tile(g_embeddings, (1, tf.shape(c)[1], 1))
                else:
                    g_embeddings = None

                x = z
                x_channels = 1
                c_channels = self._cin_channels
                g_channels = self._hparams.gin_channels

                for _ in range(self._n_block):
                    shape = tf.shape(x)
                    x = tf.reshape(x, [shape[0], shape[1] // 2, 2, x_channels])
                    x = tf.transpose(x, [0, 1, 3, 2])
                    x = tf.reshape(x, [shape[0], shape[1] // 2, 2 * x_channels])
                    
                    c = tf.reshape(c, [shape[0], shape[1] // 2, 2, c_channels])
                    c = tf.transpose(c, [0, 1, 3, 2])
                    c = tf.reshape(c, [shape[0], shape[1]  // 2, 2 * c_channels])
                    
                    if g_embeddings is not None:
                        g_embeddings = tf.reshape(g_embeddings, [shape[0], shape[1] // 2, 2, g_channels])
                        g_embeddings = tf.transpose(g_embeddings, [0, 1, 3, 2])
                        g_embeddings = tf.reshape(g_embeddings, [shape[0], shape[1]  // 2, 2 * g_channels])
                        g_channels = g_channels * 2


                    c_channels = c_channels * 2
                    x_channels = x_channels * 2

                for i, block in enumerate(self._blocks[::-1]):
                    x, c, g_embeddings = block.reverse(x, c, g_embeddings)
                return x

    def upsample(self, c):
        with tf.name_scope('upsample'):
            c = tf.expand_dims(c, 3)
            for f in self._upsample_conv:
                c = f(c)
            c = tf.squeeze(c, 3)
            return c