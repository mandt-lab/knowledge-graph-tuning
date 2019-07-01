import sys
from collections import defaultdict

import numpy as np
import tensorflow as tf


class Evaluator:
    def __init__(self, model, dat, args, log_file=sys.stdout, hits_at=[1, 3, 10]):
        log_file.write('\n# Creating evaluation harness.\n')
        log_file.flush()

        if args.eval_dat == 'both':
            self._eval_sets = ['valid', 'test']
        elif args.eval_dat == 'all':
            self._eval_sets = ['valid', 'test', 'train']
        elif args.eval_dat == 'none':
            self._eval_sets = []
        else:
            self._eval_sets = [args.eval_dat]

        if args.eval_mode == 'both':
            modes = ['raw', 'filtered']
        else:
            modes = [args.eval_mode]
        self._report_raw_metrics = 'raw' in modes
        self._report_filtered_metrics = 'filtered' in modes

        self._hits_at = np.expand_dims(np.array(hits_at), 1)  # Shape (n, 1).
        self._minibatch_placeholder = model.minibatch_htr
        self._dat = dat
        if self._report_filtered_metrics:
            self._build_filters()

        # Pick a minibatch size for which we know it will fit into the GPU.
        self._minibatch_size = args.minibatch_size * args.num_samples

        self._raw_ranks_t, self._all_scores_t, self._scores_t = self._define_ranks(
            model, model.minibatch_mean_h, model.minibatch_mean_r_predict_t,
            model.expanded_means_e, model.idx_t, dat.range_e, args)
        self._raw_ranks_h, self._all_scores_h, self._scores_h = self._define_ranks(
            model, model.minibatch_mean_t, model.minibatch_mean_r_predict_h,
            model.expanded_means_e, model.idx_h, dat.range_e, args)
        # `all_scores` have shape (minibatch_size, range_e).
        # `scores` have shape (minibatch_size).

        log_file.write('progress_columns = [\n    "training epoch",')
        metric_labels = (['mrr', 'mrr_balanced'] +
                         ['hits_at_%d' % i for i in hits_at])
        self._weights = {}
        self._placeholders_and_summaries = {}
        self._early_stopping_id = None
        for dat_label in self._eval_sets:
            # Calculate weights for balanced MRR.
            counts = np.zeros(self._dat.range_e, np.int32)
            for i in self._dat.dat[dat_label][:, :2].flatten():
                counts[i] += 1
            counts = np.maximum(1, counts)  # Avoid division by zero.
            self._weights[dat_label] = ((2.0 * len(self._dat.dat[dat_label]) / self._dat.range_e)
                                        / counts.astype(np.float32))

            # Create summary ops and placeholders to feed into them.
            placeholders = []
            summary_ops = []
            for mode in modes:
                log_file.write('\n   ')
                for metric in metric_labels:
                    if dat_label == 'valid' and mode=='filtered' and metric =='mrr':
                        self._early_stopping_id = len(summary_ops)
                    log_file.write(' "%s %s %s",' % (dat_label, mode, metric))
                    ph = tf.placeholder(tf.float32, shape=(),
                                        name='ph_%s_%s_%s' % (dat_label, mode, metric))
                    placeholders.append(ph)
                    summary_ops.append(
                        tf.summary.scalar('%s_%s' % (mode, metric), ph,
                                          family='eval_%s' % dat_label))
            self._placeholders_and_summaries[dat_label] = (
                placeholders, tf.summary.merge(summary_ops))

        log_file.write('\n]\n')
        log_file.flush()

    def _define_ranks(self, model, emb_in_e, emb_r, emb_all_e, idx_predict, range_e, args):
        all_scores = tf.squeeze(model.unnormalized_score(emb_in_e, emb_r, emb_all_e, args),
                                axis=0)
        scores = tf.batch_gather(all_scores, tf.expand_dims(idx_predict, 1))
        raw_ranks = range_e - tf.reduce_sum(tf.cast(all_scores < scores, tf.int32),
                                            axis=1)
        # This defines 1-based ranks effectively as `tf.reduce_sum(all_scores >= scores, axis=1)`,
        # except that it cannot be tricked into producing good scores producing `NaN`s.
        return raw_ranks, all_scores, tf.squeeze(scores, axis=1)

    def run(self, session, summary_writer, step, epoch, log_file=sys.stdout):
        log_file.write('    (%d,' % epoch)
        log_file.flush()
        valid_mrr = None
        for dat_label in self._eval_sets:
            phs, summary_op = self._placeholders_and_summaries[dat_label]
            results = self._evaluate_dat(session, dat_label)
            summaries = session.run(summary_op,
                                    feed_dict={ph: val for ph, val in zip(phs, results)})
            summary_writer.add_summary(summaries, global_step=step)
            for i, result in enumerate(results):
                if i % (2 + len(self._hits_at)) == 0:
                    log_file.write('\n    ')
                log_file.write(' %g,' % result)
            log_file.flush()
            if dat_label == 'valid' and self._early_stopping_id is not None:
                valid_mrr = results[self._early_stopping_id]
        log_file.write('),\n')
        log_file.flush()
        return valid_mrr

    def _evaluate_dat(self, session, dat_label):
        weights = self._weights[dat_label]
        ops = [self._raw_ranks_t, self._raw_ranks_h,
               self._all_scores_t, self._all_scores_h,
               self._scores_t, self._scores_h]
        if self._report_raw_metrics:
            raw_metrics = np.zeros((2 + len(self._hits_at),), dtype=np.float32)
        if self._report_filtered_metrics:
            filtered_metrics = np.zeros(
                (2 + len(self._hits_at),), dtype=np.float32)

        for minibatch in self._dat.iterate_in_minibatches(dat_label, self._minibatch_size):
            raw_ranks_t, raw_ranks_h, all_scores_t, all_scores_h, scores_t, scores_h = session.run(
                ops, feed_dict={self._minibatch_placeholder: minibatch})

            if self._report_raw_metrics:
                raw_metrics += self._get_minibatch_metrics(
                    raw_ranks_t, minibatch[:, 1], weights)
                raw_metrics += self._get_minibatch_metrics(
                    raw_ranks_h, minibatch[:, 0], weights)

            if self._report_filtered_metrics:
                filtered_ranks_t = self._filter(raw_ranks_t, scores_t, all_scores_t,
                                                minibatch[:, [0, 2]], self._filter_predict_t)
                filtered_ranks_h = self._filter(raw_ranks_h, scores_h, all_scores_h,
                                                minibatch[:, [1, 2]], self._filter_predict_h)
                filtered_metrics += self._get_minibatch_metrics(
                    filtered_ranks_t, minibatch[:, 1], weights)
                filtered_metrics += self._get_minibatch_metrics(
                    filtered_ranks_h, minibatch[:, 0], weights)

        ret = []
        if self._report_raw_metrics:
            ret += list(raw_metrics)
        if self._report_filtered_metrics:
            ret += list(filtered_metrics)
        return np.array(ret, dtype=np.float32) / (2 * len(self._dat.dat[dat_label]))

    def _get_minibatch_metrics(self, ranks, target_indices, weights):
        inverse_ranks = 1.0 / ranks.astype(np.float32)
        target_weights = weights[target_indices]
        sum_rr = np.sum(inverse_ranks)
        balanced_sum_rr = np.dot(target_weights, inverse_ranks)
        hits_at = np.sum(np.expand_dims(ranks, 0) <= self._hits_at, axis=1)
        return np.array([sum_rr, balanced_sum_rr] + list(hits_at), dtype=np.float32)

    def _build_filters(self):
        filter_predict_t = defaultdict(set)
        filter_predict_h = defaultdict(set)
        for subset in self._dat.dat.values():
            for h, t, r in subset:
                filter_predict_t[(h, r)].add(t)
                filter_predict_h[(t, r)].add(h)

        # Turns sets into numpy arrays for more efficient lookup
        self._filter_predict_t = {key: np.array(list(value))
                                  for key, value in filter_predict_t.items()}
        self._filter_predict_h = {key: np.array(list(value))
                                  for key, value in filter_predict_h.items()}

    def _filter(self, raw_ranks, scores, all_scores, inputs, filter_dict):
        return np.array([raw_rank - np.sum(all_s[filter_dict[(e_in, r)]] > score)
                         for raw_rank, score, all_s, (e_in, r)
                         in zip(raw_ranks, scores, all_scores, inputs)])


def add_cli_args(parser):
    '''Add command line arguments to control frequency and type of model evaluations.

    Arguments:
    parser -- An `argparse.ArgumentParser`.

    Returns:
    The command line argument group that was just added to the parser.
    '''
    eval_args = parser.add_argument_group(
        'Evaluation parameters')
    eval_args.add_argument('--eval_dat', choices=['valid', 'test', 'both', 'train', 'all', 'none'],
                           default='valid', help='''
        Choose whether to evaluate on the validation set (`--eval_dat valid`), the test set
        (`--eval_dat test`), or both (`--eval_dat both`). For debugging purposes, we also provide
        the choices `--eval_dat train`, `--eval_dat all` (which means evaluating on the train,
        validation, and test set), and `--eval_dat none`.''')
    eval_args.add_argument('--eval_mode', choices=['raw', 'filtered', 'both'], default='both',
                           help='''
        Choose which type of predicted ranks to use for evaluation: `raw` to use unfiltered ranks,
        `filtered` to use filtered ranks according to Bordes et al., NIPS 2013, or `both` (default)
        to report evaluation metrics based on both filtered and filtered ranks.''')
