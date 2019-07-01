import tensorflow as tf

from abstract_model import AbstractModel


class DistMultModel(AbstractModel):
    def define_emb(self, args, dat):
        '''Defines entity and relation embeddings of shape (dat.range_e, emb_dim) and
        (2 * dat.range_r, emb_dim)'''
        initializer = tf.contrib.layers.xavier_initializer()
        emb_e = tf.Variable(initializer(
            (dat.range_e, args.embedding_dim)), name='emb_e')
        emb_r = tf.Variable(initializer(
            (2 * dat.range_r, args.embedding_dim)), name='emb_r')
        return {'emb': emb_e}, {'emb': emb_r}

    def unnormalized_score(self, emb_in_e, emb_r, emb_all_e, args):
        x = emb_in_e['emb'] * emb_r['emb']
        # `x` has shape (minibatch_size, num_samples, embedding_dimension)

        logits = tf.matmul(tf.transpose(x, [1, 0, 2]),
                           tf.transpose(emb_all_e['emb'], [1, 0, 2]),
                           transpose_b=True)
        # `logits` has shape (num_samples, minibatch_size, range_e)
        return logits

    def single_log_prior(self, inverse_lambda, emb):
        # 3-norm prior.
        return tf.reduce_sum(tf.abs(emb['emb'])**3, axis=(1, 2)) / (-3.0 * inverse_lambda)

    def estimate_inverse_lambda(self, samples):
        samples_shape = samples['emb'].get_shape().as_list()
        total_dimensions = samples_shape[1] * samples_shape[2]
        return (1.0 / total_dimensions) * tf.reduce_sum(tf.abs(samples['emb'])**3, axis=(1, 2))
