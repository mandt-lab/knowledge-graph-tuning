import sys
import abc

import tensorflow as tf
import numpy as np

import optimizer


class AbstractModel(abc.ABC):
    '''Abstract base class for knowledge graph embedding models.

    To define a new model, derive from this class and implement the method
    `define_score()`.
    '''

    def __init__(self, args, dat, rng, log_file=sys.stdout):
        log_file.write('\n# Creating model.\n')
        log_file.flush()

        self._summaries = []
        with tf.variable_scope('means'):
            self.means_e, self.means_r = self.define_emb(args, dat)
        with tf.variable_scope('samples_e'):
            self.samples_e, self.expanded_means_e, self.log_std_e = self.create_all_samplers(
                self.means_e, args)
        with tf.variable_scope('samples_r'):
            self.samples_r, self.expanded_means_r, self.log_std_r = self.create_all_samplers(
                self.means_r, args)

        with tf.variable_scope('minibatch'):
            self.minibatch_htr = tf.placeholder(
                tf.int32, shape=(None, 3), name='minibatch_htr')
            minibatch_size = tf.shape(self.minibatch_htr)[0]
            minibatch_size_float = tf.cast(minibatch_size, tf.float32)

            self.idx_h = self.minibatch_htr[:, 0]
            self.idx_t = self.minibatch_htr[:, 1]
            idx_r_predict_t = self.minibatch_htr[:, 2]
            idx_r_predict_h = idx_r_predict_t + dat.range_r

            emb_h = {label: tf.gather(samples, self.idx_h)
                     for label, samples in self.samples_e.items()}
            emb_t = {label: tf.gather(samples, self.idx_t)
                     for label, samples in self.samples_e.items()}
            emb_r_predict_t = {label: tf.gather(samples, idx_r_predict_t)
                               for label, samples in self.samples_r.items()}
            emb_r_predict_h = {label: tf.gather(samples, idx_r_predict_h)
                               for label, samples in self.samples_r.items()}

            self.minibatch_mean_h = {
                label: tf.gather(means, self.idx_h)
                for label, means in self.expanded_means_e.items()}
            self.minibatch_mean_t = {
                label: tf.gather(means, self.idx_t)
                for label, means in self.expanded_means_e.items()}
            self.minibatch_mean_r_predict_t = {
                label: tf.gather(means, idx_r_predict_t)
                for label, means in self.expanded_means_r.items()}
            self.minibatch_mean_r_predict_h = {
                label: tf.gather(means, idx_r_predict_h)
                for label, means in self.expanded_means_r.items()}

            # Prefactor for normalization per training data point.
            normalizer = 1.0 / tf.cast(args.num_samples, tf.float32)

        with tf.variable_scope('log_likelihood'):
            with tf.variable_scope('tail_prediction'):
                self.scores_predict_t = self.unnormalized_score(
                    emb_h, emb_r_predict_t, self.samples_e, args)
                ll_predict_t = normalizer * self._log_likelihood(
                    self.scores_predict_t, self.idx_t, args)
            with tf.variable_scope('head_prediction'):
                self.scores_predict_h = self.unnormalized_score(
                    emb_t, emb_r_predict_h, self.samples_e, args)
                ll_predict_h = normalizer * self._log_likelihood(
                    self.scores_predict_h, self.idx_h, args)
            log_likelihood = ll_predict_t + ll_predict_h

        with tf.variable_scope('hyperparameters'):
            counts_e, sort_indices_e = self._get_counts(
                dat.dat['train'][:, :2].flatten(), dat.range_e, 'e')
            counts_r, sort_indices_r = self._get_counts(
                dat.dat['train'][:, 2], dat.range_r, 'r')
            self.inverse_lambda_e = self._define_inverse_lambda(
                args, counts_e, 'e')
            self.inverse_lambda_r = self._define_inverse_lambda(
                args, counts_r, 'r')

        inverse_counts_e = (1.0 / counts_e).astype(np.float32)
        inverse_counts_r = (1.0 / counts_r).astype(np.float32)
        self._lambda_sigma_summary(
            self.inverse_lambda_e, self.log_std_e, inverse_counts_e, sort_indices_e, 'e')
        self._lambda_sigma_summary(
            self.inverse_lambda_r, self.log_std_r, inverse_counts_r, sort_indices_r, 'r')

        with tf.variable_scope('log_prior'):
            # r-counts are the same for head and tail prediction, so gather them only once.
            minibatch_inverse_counts_r = tf.gather(
                inverse_counts_r, idx_r_predict_t)
            log_prior = normalizer * (
                tf.reduce_sum(
                    tf.gather(inverse_counts_e, self.idx_h) * self.single_log_prior(
                        tf.gather(self.inverse_lambda_e, self.idx_h), emb_h))
                + tf.reduce_sum(
                    tf.gather(inverse_counts_e, self.idx_t) * self.single_log_prior(
                        tf.gather(self.inverse_lambda_e, self.idx_t), emb_t))
                + tf.reduce_sum(
                    minibatch_inverse_counts_r * self.single_log_prior(
                        tf.gather(self.inverse_lambda_r, idx_r_predict_t), emb_r_predict_t))
                + tf.reduce_sum(
                    minibatch_inverse_counts_r * self.single_log_prior(
                        tf.gather(self.inverse_lambda_r, idx_r_predict_h), emb_r_predict_h)))

        if args.em:
            # Calculate entropy of entire variational distribution (independent of minibatch).
            # Normalize per training data point.
            with tf.variable_scope('entropy'):
                entropy = (minibatch_size_float / len(dat.dat['train'])) * tf.add_n(
                    [tf.reduce_sum(i) for i in
                     list(self.log_std_e.values()) + list(self.log_std_r.values())],
                    name='entropy')
            self.loss = -tf.add_n([log_prior, log_likelihood, entropy],
                                  name='elbo')
        else:
            self.loss = -tf.add_n([log_prior, log_likelihood],
                                  name='log_joint')

        with tf.variable_scope('loss_parts'):
            normalizer_per_embedding = (
                len(dat.dat['train']) /
                (args.embedding_dim * (dat.range_e + 2 * dat.range_r) * minibatch_size_float))
            normalizer_per_datapoint = 0.5 / minibatch_size_float
            if args.em:
                self._summaries.append(tf.summary.scalar('entropy_per_embedding_and_dimension',
                                                         normalizer_per_embedding * entropy))
            self._summaries.append(tf.summary.scalar('log_prior_per_embedding_and_dimension',
                                                     normalizer_per_embedding * log_prior))
            self._summaries.append(tf.summary.scalar('log_likelihood_per_datapoint',
                                                     normalizer_per_datapoint * log_likelihood))
            self._summaries.append(tf.summary.scalar('loss_per_datapoint',
                                                     normalizer_per_datapoint * self.loss))

        global_step, lr_base, lr_summary = optimizer.define_base_learning_rate(
            args)
        self._summaries.append(lr_summary)

        with tf.variable_scope('e_step'):
            opt_mean = optimizer.define_optimizer(args, args.lr0_mu * lr_base)
            variational_parameters_mean = tf.trainable_variables('means/')
            update_means = opt_mean.minimize(
                self.loss, global_step=global_step, var_list=variational_parameters_mean)
            log_file.write('# %d variational parameters for means\n' %
                           len(variational_parameters_mean))

            if args.em:
                opt_sigma = optimizer.define_optimizer(
                    args, args.lr0_sigma * lr_base)
                variational_parameters_sigma = (tf.trainable_variables('samples_e/') +
                                                tf.trainable_variables('samples_r/'))
                log_file.write('# %d variational parameters for standard deviations\n' %
                               len(variational_parameters_sigma))
                update_sigmas = opt_sigma.minimize(
                    self.loss, var_list=variational_parameters_sigma)
                self._e_step = tf.group(
                    update_means, update_sigmas, name='e_step')
            else:
                self._e_step = update_means

        if args.em:
            with tf.variable_scope('m_step'):
                lr_lambda = args.lr0_lambda * lr_base
                update_lambda_e = tf.assign(
                    self.inverse_lambda_e,
                    (1.0 - lr_lambda) * self.inverse_lambda_e
                    + lr_lambda * self.estimate_inverse_lambda(self.samples_e))
                update_lambda_r = tf.assign(
                    self.inverse_lambda_r,
                    (1.0 - lr_lambda) * self.inverse_lambda_r
                    + lr_lambda * self.estimate_inverse_lambda(self.samples_r))
                m_step = tf.group(
                    update_lambda_e, update_lambda_r, name='m_step')
                log_file.write('# 2 hyperparameters\n')
            self._em_step = tf.group(self._e_step, m_step, name='em_step')
        else:
            self._em_step = None

        self._summary_op = tf.summary.merge(self._summaries)

    def create_all_samplers(self, means, args):
        expanded_means = {}
        samples = {}
        log_std = {} if args.em else None

        for label, mean in means.items():
            expanded_mean = tf.expand_dims(mean, axis=1)
            expanded_means[label] = expanded_mean
            if args.em:
                log_std[label] = tf.Variable(
                    tf.fill(expanded_mean.get_shape(),
                            np.log(args.initial_std).astype(np.float32)),
                    dtype=tf.float32, name='%s_log_std' % label)

                std = tf.exp(log_std[label], name='%s_std' % label)
                self._summaries.append(
                    tf.summary.histogram('%s_std' % label, std))

                shape = expanded_mean.get_shape().as_list()
                shape[1] = args.num_samples
                samples[label] = tf.random_normal(
                    shape, mean=expanded_mean, stddev=std, name='%s_samples' % label)
            else:
                samples[label] = expanded_mean
        return samples, expanded_means, log_std

    def _get_counts(self, entries, range_max, label):
        counts = np.zeros(range_max, np.int32)
        for i in entries:
            counts[i] += 1
        counts = np.maximum(1, counts)  # Avoid division by zero.
        sort_indices = np.argsort(counts)
        counts = counts.astype(np.float32)
        if label == 'r':
            counts = np.concatenate((counts, counts))
            sort_indices = np.array([(i, i + range_max)
                                     for i in sort_indices]).flatten()
        return counts, sort_indices

    def _define_inverse_lambda(self, args, counts, label):
        if args.initial_reg_uniform:
            # Since frequencies add up to 1, the average frequency is `1 / len(frequencies)`.
            initial_lambda = args.initial_reg_strength * counts.sum() / len(counts)
            initializer = tf.fill(
                (len(counts),),
                (1.0 / initial_lambda).astype(np.float32))
        else:
            initializer = (1.0 / (args.initial_reg_strength *
                                  counts.astype(np.float32)))

        return tf.Variable(initializer, dtype=tf.float32,
                           name='inverse_lambda_%s' % label, trainable=args.em)

    def _lambda_sigma_summary(self, inverse_lambda, log_std, inverse_counts, sort_indices, label):
        lmbda = 1.0 / inverse_lambda
        sorted_lambda = tf.gather(lmbda, sort_indices)
        downscaled_sorted_lambda = tf.gather(
            inverse_counts * lmbda, sort_indices)

        self._summaries.append(tf.summary.histogram('lambda_%s' % label, sorted_lambda,
                                                    family='hyperparmeters'))
        self._summaries.append(tf.summary.histogram('downscaled_lambda_%s' % label,
                                                    downscaled_sorted_lambda,
                                                    family='hyperparmeters'))

        one_3rd = len(sort_indices) // 3
        with tf.variable_scope('avg_lambda_%s' % label):
            self._summaries.append(tf.summary.scalar(
                'a_low_frequency', tf.reduce_mean(sorted_lambda[:one_3rd])))
            self._summaries.append(tf.summary.scalar(
                'b_med_frequency', tf.reduce_mean(sorted_lambda[one_3rd: -one_3rd])))
            self._summaries.append(tf.summary.scalar(
                'c_high_frequency', tf.reduce_mean(sorted_lambda[-one_3rd:])))

        with tf.variable_scope('avg_downscaled_lambda_%s' % label):
            self._summaries.append(tf.summary.scalar(
                'a_low_frequency', tf.reduce_mean(downscaled_sorted_lambda[:one_3rd])))
            self._summaries.append(tf.summary.scalar(
                'b_med_frequency', tf.reduce_mean(downscaled_sorted_lambda[one_3rd: -one_3rd])))
            self._summaries.append(tf.summary.scalar(
                'c_high_frequency', tf.reduce_mean(downscaled_sorted_lambda[-one_3rd:])))

        if log_std is not None:
            sorted_log_std = tf.add_n([tf.gather(l, sort_indices)
                                       for l in log_std.values()])
            with tf.variable_scope('avg_log_std_%s' % label):
                normalizer = 1.0 / len(log_std)
                self._summaries.append(tf.summary.scalar(
                    'a_low_frequency', normalizer * tf.reduce_mean(sorted_log_std[:one_3rd])))
                self._summaries.append(tf.summary.scalar(
                    'b_med_frequency', normalizer * tf.reduce_mean(sorted_log_std[one_3rd: -one_3rd])))
                self._summaries.append(tf.summary.scalar(
                    'c_high_frequency', normalizer * tf.reduce_mean(sorted_log_std[-one_3rd:])))

    @abc.abstractmethod
    def define_emb(self, args, dat):
        '''Define tensorflow Variables for the latent embedding vectors.

        Arguments:
        args -- Namespace containing command line arguments.
        dat -- A `Dataset`. Mainly useful for its properties `range_e` and `range_r`.

        Returns:
        A pair of dicts. The first dict maps labels (strings) to `tf.Variable`s for the
        entity embeddings.
        '''
        pass

    @abc.abstractmethod
    def unnormalized_score(self, emb_in_e, emb_r, emb_all_e, args):
        '''Define a tensorflow op that calculates the prediction scores (logits).

        This is also sometimes called `logits`.

        Arguments:
        emb_in_e -- Embedding vectors of the input entities, i.e., the entities on which
            we condition. A dict that maps labels (strings) to tensors of shape
            `(minibatch_size, num_samples, embedding_dimensions...)`.
        emb_r -- Embedding vectors of the relations. If reciprocal relations are used
            then the caller should pass in different embedding vectors for head or tail
            prediction. A dict that maps labels (strings) to tensors of shape
            `(minibatch_size, num_samples, embedding_dimensions...)`.
        emb_all_e -- Embedding vectors of all entities. A dict that maps labels (strings)
            to tensors of shape `(range_e, num_samples, embedding_dimensions...)`.
        args -- Namespace holding command line arguments.

        Returns:
        A tensor of shape `(num_samples, minibatch_size, range_e)` that holds the
        unnormalized represents the negative log likelihood of the data.
        Should *not* be normalized to the batch size or sample size.
        '''
        pass

    def _log_likelihood(self, scores, idx_target, args):
        labels = tf.expand_dims(idx_target, 0)
        if args.em and args.num_samples != 1:
            # Broadcast explicitly since Tensorflow's cross entropy function does not do it.
            labels = tf.tile(labels, [args.num_samples, 1])
            # `labels` has shape (num_samples, minibatch_size)

        # The documentation for `tf.nn.sparse_softmax_cross_entropy_with_logits` is a bit unclear
        # about signs and normalization. It turns out that the function does the following,
        # assuming that `labels.shape = (m, n)` and `logits.shape = (m, n, o)`:
        #   tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        #   = - [[logits[i, j, labels[i, j]] for j in range(n)] for i in range(m)]
        #     + log(sum(exp(logits), axis=2))
        negative_scores = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=scores)
        # `negative_scores` has shape (num_samples, minibatch_size)
        return -tf.reduce_sum(negative_scores)

    @abc.abstractmethod
    def single_log_prior(self, inverse_lambda, emb):
        '''
        Calculate the log prior of a single embedding vector.

        Arguments:
        inverse_lambda -- The inverse of the regularizer strengths. Tensor of shape
            `(minibatch_size,)`.
        emb -- A single embedding vector. A dict with the same keys as the ones
            returned by `define_emb`. Each tensor has shape
            `(minibatch_size, num_samples, embedding_dimensions...)`.

        Returns:
        A tensor of size `(minibatch_size)` containing the weighted log priors for all samples.
        Should *not* be normalized by the number of samples.
        '''
        pass

    @abc.abstractmethod
    def estimate_inverse_lambda(self, emb):
        pass

    @property
    def e_step(self):
        '''Tensorflow op for a gradient step with fixed hyperparameters.'''
        return self._e_step

    @property
    def em_step(self):
        '''Tensorflow op for a gradient step in model and hyperaparametre space.'''
        return self._em_step

    @property
    def summary_op(self):
        '''A tensorflow op that evaluates some summary statistics for Tensorboard.

        Run this op in a session and use a `tf.summary.FileWriter` to write the result
        to a file. Visualize the summaries with Tensorboard.'''
        return self._summary_op


def add_cli_args(parser):
    '''Add command line arguments for all knowledge graph embedding models.

    Arguments:
    parser -- An `argparse.ArgumentParser`.

    Returns:
    The command line argument group that was just added to the parser.
    '''
    model_args = parser.add_argument_group(
        'Generic model parameters')
    model_args.add_argument('--model', required=True, choices=['ComplEx', 'DistMult'], help='''
        Choose the knowledge graph embedding model.''')
    model_args.add_argument('-k', '--embedding_dim', metavar='N', type=int, default=100, help='''
        Set the embedding dimension.''')
    model_args.add_argument('--em', action='store_true', help='''
        Turn on variational expectation maximization, i.e., optimize over hyperparmeters.''')
    model_args.add_argument('-S', '--num_samples', metavar='FLOAT', type=int, default=1, help='''
        Set the number of samples from the variational distribution that is used to estimate
        the ELBO; only used if `--em` is used.''')
    model_args.add_argument('--initial_std', metavar='FLOAT', type=float, default=1.0, help='''
        Initial standard deviation of the variational distribution; only used if `--em` is
        used.''')
    model_args.add_argument('--initial_reg_strength', metavar='FLOAT', type=float,
                            default=1.0, help='''
        Set the (initial) regularizer strengths $\\lambda$. To make this flag more portable across
        data sets, the provided value will still be scaled with a frequency: if
        `--initial_reg_uniform` is set, then the scaling factor is the average frequency of
        entities or relations. If `--initial_reg_uniform` is not set, then the scaling factor is
        the individual frequency of each entity or relation. Not that, if `--em` is used, then
        `--initial_reg_strength` only affects the initialization of the regularizer strengths as
        the EM algorithm will optimize over the regularizer strengths. If `--em` is not used, then
        the regularizer strengths controlled by this flag are held constant throughout
        training.''')
    model_args.add_argument('--initial_reg_uniform', action='store_true', help='''
        Set the (initial) regularizer strengths $\\lambda$ to a uniform value: use the value
        provided with `--initial_reg_strength`, scaled only by the average frequency of entities
        or relations. Default is to scale the (initial) regularizer strengths by the frequency of
        each individual entity or relation. See also `--initial_reg_strength`.''')
    model_args.add_argument('--lr0_mu', metavar='FLOAT', type=float, default=0.02, help='''
        Set the initial prefactor for the (possibly adaptive) learning rate for the means. See
        `--lr_exponent`.''')
    model_args.add_argument('--lr0_sigma', metavar='FLOAT', type=float, default=0.1, help='''
        Set the initial prefactor for the (possibly adaptive) learning rate for the (log) standard
        deviations (only used if `--em` is set). See `--lr_exponent`.''')
    model_args.add_argument('--lr0_lambda', metavar='FLOAT', type=float, default=0.5, help='''
        Set the initial prefactor for the learning rate for the hyperparameters. Must not be larger
        than 1. See `--lr_exponent`.''')
