import os
import sys

import numpy as np


class Dataset:
    def __init__(self, dir_path, log_file=sys.stdout, subsets=['train', 'valid', 'test']):
        log_file.write(
            '# Loading data set from directory `%s`.\n' % dir_path)
        log_file.flush()

        self.dat = {}
        for label in subsets:
            self.dat[label] = self._load_file(
                os.path.join(dir_path, '%s2id.txt' % label))
            log_file.write('%s_points = %d\n' % (label, len(self.dat[label])))
            log_file.flush()

        self.range_e = max(dat[:, :2].max() for dat in self.dat.values()) + 1
        self.range_r = max(dat[:, 2].max() for dat in self.dat.values()) + 1
        log_file.write('range_e = %d  # number of entities\n' % self.range_e)
        log_file.write('range_r = %d  # number of relations (before adding reciprocal relations)\n'
                       % self.range_r)

    def _load_file(self, path):
        with open(path, 'r') as f:
            count = int(f.readline())
            ret = np.empty((count, 3), dtype=np.int32)
            for i in range(count):
                ret[i, :] = [int(val) for val in f.readline().split(' ')]
            assert len(f.read(1)) == 0

        return ret

    def iterate_in_minibatches(self, subset_label, minibatch_size, rng=None):
        '''Note: if rng is provided, the returned iterator mutates the data set.'''
        dat = self.dat[subset_label]
        if rng is not None:
            rng.shuffle(dat)

        # Iterate over all full sized minibatches
        for start in range(0, len(dat) - minibatch_size + 1, minibatch_size):
            yield dat[start: start + minibatch_size]

        # If in deterministic mode, also return a final smaller minibatch if the data set size is
        # not a multiple of the minibatch size (not necessary in shuffle mode).
        if rng is None and start + minibatch_size < len(dat):
            yield dat[start + minibatch_size:]
