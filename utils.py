import tensorflow as tf

def fp16_dtype_getter(getter, name, shape=None, dtype=None, trainable=True, regularizer=None, *args, **kwargs):
    storage_dtype = tf.float32 if dtype in [tf.float32, tf.float16] else dtype
    variable = getter(
        name,
        shape,
        dtype=storage_dtype,
        trainable=trainable,
        regularizer=(
            regularizer if
            (trainable and not any(l_name.lower() in name.lower()
                                    for l_name in ['batchnorm', 'batch_norm'])) else None
        ),
        *args,
        **kwargs
    )

    if dtype != tf.float32:
        cast_name = name + '/fp16_cast'

        try:
            cast_variable = tf.get_default_graph().get_tensor_by_name(cast_name + ':0')

        except KeyError:
            cast_variable = tf.cast(variable, dtype, name=cast_name)

        cast_variable._ref = variable._ref
        variable = cast_variable

    return variable


def average_gradients(tower_grads):
    with tf.name_scope('grad_avg'):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                if g is not None:
                    expanded_g = tf.expand_dims(g, 0)

                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)

            if len(grads) > 0:
                # Average over the 'tower' dimension.
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)

                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So .. we will just return the first tower's pointer to
                # the Variable.
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)
        return average_grads