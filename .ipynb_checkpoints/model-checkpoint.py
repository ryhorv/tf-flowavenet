import tensorflow as tf
from modules import WaveNet
from math import log, pi

logabs = lambda x: tf.log(tf.abs(x))

class ActNorm:
    def __init__(self, in_channel, logdet=True, pretrained=False, scope='ActNorm'):
        with tf.variable_scope(scope):
            self._loc = tf.get_variable('loc', shape=[1, 1, in_channel])
            self._scale = tf.get_variable('scale', shape=[1, 1, in_channel])

            self._initialized = pretrained
            self._logdet = logdet

    def initialize(self, x):
        with tf.name_scope('initialize'):
            mean, std = tf.nn.moments(x, axes=[0, 1], keep_dims=True)
            mean = tf.stop_gradient(mean)
            std = tf.stop_gradient(std)

            self._loc.assign(-mean)
            self._scale.assign(1 / (std + 1e-6))

    def forward(self, x):
        with tf.name_scope('forward'):
            if not self._initialized:
                self.initialize(x)
                self.initialized = True

            log_abs = logabs(self._scale)

            logdet = tf.reduce_mean(log_abs)

            if self._logdet:
                return self._scale * (x + self._loc), logdet

            else:
                return self._scale * (x + self._loc)

    def reverse(self, output):
        with tf.name_scope('reverse'):
            return output / self._scale - self._loc

    def __call__(self, x):
        return self.forward(x)


class AffineCoupling:
    def __init__(self, in_channel, cin_channel, filter_size=256, num_layer=6, affine=True, causal=False, scope='AffineCoupling'):
        with tf.variable_scope(scope):
            self._in_channel = in_channel
            self._affine = affine
            self._net = WaveNet(in_channels=in_channel // 2, out_channels=in_channel if self._affine else in_channel // 2,
                            num_blocks=1, num_layers=num_layer, residual_channels=filter_size,
                            gate_channels=filter_size, skip_channels=filter_size,
                            kernel_size=3, cin_channels=cin_channel // 2, causal=causal)
                            

    def forward(self, x, c=None):
        with tf.name_scope('forward'):
            in_a, in_b = tf.split(x, axis=2, num_or_size_splits=2)
            in_a.set_shape([None, None, self._in_channel//2])
            in_b.set_shape([None, None, self._in_channel - (self._in_channel//2)])
            
            print(c)
            c_a, c_b = tf.split(c, axis=2, num_or_size_splits=2)
            c_a.set_shape([None, None, tf.shape(c)[2] // 2])
            c_b.set_shape([None, None, tf.shape(c)[2] - (tf.shape(c)[2] // 2)])

            if self._affine:
                log_s, t = tf.split(self._net(in_a, c_a), axis=2, num_or_size_splits=2)

                out_b = (in_b - t) * tf.exp(-log_s)
                logdet = tf.reduce_mean(-log_s) / 2
            else:
                net_out = self._net(in_a, c_a)
                out_b = in_b + net_out
                logdet = None

            return tf.concat(2, [in_a, out_b]), logdet

    def reverse(self, output, c=None):
        with tf.name_scope('reverse'):
            out_a, out_b = tf.split(output, axis=2, num_or_size_splits=2)
            c_a, c_b = tf.split(c, axis=2, num_or_size_splits=2)

            if self._affine:
                # log_s, t = self.net(out_a, c_a).chunk(2, 1)
                log_s, t = tf.split(self._net(out_a, c_a), axis=2, num_or_size_splits=2)
                in_b = out_b * tf.exp(log_s) + t
            else:
                net_out = self._net(out_a, c_a)
                in_b = out_b - net_out

            return tf.concat(2, [out_a, in_b])

    def __call__(self, x, c=None):
        return self.forward(x, c)

def change_order(x, c=None):
    x_a, x_b = tf.split(x, axis=2, num_or_size_splits=2)
    c_a, c_b = tf.split(c, axis=2, num_or_size_splits=2)

    return tf.concat(2, [x_b, x_a]), tf.concat(2, [c_b, c_a])

class Flow:
    def __init__(self, in_channel, cin_channel, filter_size, num_layer, affine=True, causal=False, pretrained=False, scope='Flow'):
        with tf.variable_scope(scope):
            self._actnorm = ActNorm(in_channel, pretrained=pretrained)
            self._coupling = AffineCoupling(in_channel, cin_channel, filter_size=filter_size,
                                       num_layer=num_layer, affine=affine, causal=causal)

    def forward(self, x, c=None):
        with tf.name_scope('forward'):
            out, logdet = self._actnorm(x)
            out, det = self._coupling(out, c)
            out, c = change_order(out, c)

            if det is not None:
                logdet = logdet + det

            return out, c, logdet

    def reverse(self, output, c=None):
        with tf.name_scope('reverse'):
            output, c = change_order(output, c)
            x = self._coupling.reverse(output, c)
            x = self._actnorm.reverse(x)
            return x, c

    def __call__(self, x, c=None):
        return self.forward(x, c)

class Block:
    def __init__(self, in_channel, cin_channel, n_flow, n_layer, affine=True, causal=False, pretrained=False, scope='Block'):
        with tf.variable_scope(scope):
            squeeze_dim = in_channel * 2
            squeeze_dim_c = cin_channel * 2

            self._flows = []
            for i in range(n_flow):
                self._flows.append(Flow(squeeze_dim, squeeze_dim_c, filter_size=256, num_layer=n_layer, affine=affine,
                                    causal=causal, pretrained=pretrained, scope='Flow_%d' % i))

    def forward(self, x, c):
        with tf.name_scope('forward'):
#             b_size, n_channel, T = x.size()
            # x.shape = [b_size, T, n_channel]
#             b_size, T, n_channel = tf.shape(x)
            shape = tf.shape(x)
            squeezed_x = tf.reshape(x, [shape[0], shape[1] // 2, 2, shape[2]])
            out = tf.reshape(squeezed_x, [shape[0], shape[1] // 2, 2 * shape[2]]) # может быть ошибка из-за изначально другого расположения осей


            squeezed_c = tf.reshape(c, [shape[0], shape[1] // 2, 2, -1])
            c = tf.reshape(squeezed_c, [shape[0], shape[1] // 2, -1]) # может быть ошибка из-за изначально другого расположения осей
            logdet = 0

            for flow in self._flows:
                out, c, det = flow(out, c)
                logdet = logdet + det
            return out, c, logdet

    def reverse(self, output, c):
        with tf.name_scope('reverse'):
            x = output

            for flow in self._flows[::-1]:
                x, c = flow.reverse(x, c)

            b_size, T, n_channel = tf.shape(x)
            
            unsqueezed_x = tf.reshape(x, [b_size, T, 2, n_channel // 2])
            unsqueezed_x = tf.reshape(unsqueezed_x, [b_size, T * 2, n_channel // 2])


            unsqueezed_c = tf.reshape(c, [b_size, T, 2, None])
            unsqueezed_c = tf.reshape(unsqueezed_c, [b_size, T * 2, None])


            return unsqueezed_x, unsqueezed_c

    def __call__(self, x, c):
        return self.forward(x, c)

class FloWaveNet:
    def __init__(self, in_channel, cin_channel, n_block, n_flow, n_layer, affine=True, causal=False, pretrained=False, scope='FloWaveNet'):
        with tf.variable_scope(scope):
            self._blocks = []
            self._n_block = n_block
            for i in range(self._n_block):
                self._blocks.append(Block(in_channel, cin_channel, n_flow, n_layer, affine=affine,
                                        causal=causal, pretrained=pretrained, scope='Block_%d' % i))
                in_channel *= 2
                cin_channel *= 2

            self._upsample_conv = []
            for s in [16, 16]:
                convt = tf.layers.Conv2DTranspose(filters=1, 
                                                  kernel_size=(2 * s, 3), 
                                                  padding='same', 
                                                  strides=(s, 1), 
                                                  activation=lambda x: tf.nn.leaky_relu(x, 0.4))
                self._upsample_conv.append(convt)                                  


    def forward(self, x, c):
        with tf.name_scope('forward'):
            logdet = 0
            out = x
            c = self.upsample(c)
            for block in self._blocks:
                out, c, logdet_new = block(out, c)
                logdet = logdet + logdet_new

            log_p = tf.reduce_mean(0.5 * (- log(2.0 * pi) - tf.square(out)))
            return log_p, logdet

    def reverse(self, z, c):
        with tf.name_scope('reverse'):
            c = tf.cond(tf.equal(tf.shape(z)[1], tf.shape(c)[1]), 
                        true_fn=lambda: c, 
                        false_fn=lambda: self.upsample(c))

            x = z
            for _ in range(self._n_block):
                b_size, T, _ = tf.shape(x)
                squeezed_x = tf.reshape(x, [b_size, T // 2, 2, None])
                x = tf.reshape(squeezed_x, [b_size, T // 2, None]) # может быть ошибка из-за изначально другого расположения осей

                squeezed_c = tf.reshape(c, [b_size, T // 2, 2, None])
                c = tf.reshape(squeezed_c, [b_size, T // 2, None]) # может быть ошибка из-за изначально другого расположения осей
               

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