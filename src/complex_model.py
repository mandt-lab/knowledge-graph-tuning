import tensorflow as tf

from abstract_model import AbstractModel


class ComplExModel(AbstractModel):
    def define_emb(self, args, dat):
        '''Defines complex variables of shape (dat.range_e, emb_dim) and
        (2 * dat.range_r, emb_dim)'''
        initializer = tf.contrib.layers.xavier_initializer()
        emb_e_re = tf.Variable(initializer(
            (dat.range_e, args.embedding_dim)), name='emb_e_re')
        emb_e_im = tf.Variable(initializer(
            (dat.range_e, args.embedding_dim)), name='emb_e_im')
        emb_r_re = tf.Variable(initializer(
            (2 * dat.range_r, args.embedding_dim)), name='emb_r_re')
        emb_r_im = tf.Variable(initializer(
            (2 * dat.range_r, args.embedding_dim)), name='emb_r_im')
        return {'re': emb_e_re, 'im': emb_e_im}, {'re': emb_r_re, 'im': emb_r_im}

    def unnormalized_score(self, emb_in_e, emb_r, emb_all_e, args):
        x = self._pointwise_complex_product(emb_in_e, emb_r)
        # `x` has shape (minibatch_size, num_samples, embedding_dimension)
        logits = self._re_complex_batch1_matmul_adjoint(x, emb_all_e)
        # `logits` has shape (num_samples, minibatch_size, range_e)
        return logits

    def _pointwise_complex_product(self, a, b):
        return {
            're': a['re'] * b['re'] - a['im'] * b['im'],
            'im': a['im'] * b['re'] + a['re'] * b['im']
        }

    def _re_complex_batch1_matmul_adjoint(self, a, b):
        '''Perform batch multiplication of complex matrices a and adjoint(b)

        Assumes that the batch dimension is axis 1. Returns only the real part of the result.'''
        a_re = tf.transpose(a['re'], [1, 0, 2])
        a_im = tf.transpose(a['im'], [1, 0, 2])
        b_re = tf.transpose(b['re'], [1, 0, 2])
        b_im = tf.transpose(b['im'], [1, 0, 2])
        return (tf.matmul(a_re, b_re, transpose_b=True) +
                tf.matmul(a_im, b_im, transpose_b=True))

    def single_log_prior(self, inverse_lambda, emb):
        # Nuclear 3-norm prior
        return (tf.reduce_sum((emb['re']**2 + emb['im']**2) ** (3.0/2.0), axis=(1, 2))
                / (-3.0 * inverse_lambda))

    def estimate_inverse_lambda(self, samples):
        samples_shape = samples['re'].get_shape().as_list()
        total_dimensions = 2 * samples_shape[1] * samples_shape[2]
        return (1.0 / total_dimensions) * tf.reduce_sum(
            (samples['re']**2 + samples['im']**2) ** (3.0/2.0), axis=(1, 2))
