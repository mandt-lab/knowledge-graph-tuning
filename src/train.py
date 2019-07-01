'''Module for boilerplate code around the training loop.

Defines an abstract base class `Model` for Bayesian word embedding models, a
function `train()` that runs the training loop, and a function `add_cli_args()`
that adds command line arguments to control the training loop (e.g., the number
of training steps and the log frequency).
'''

import pickle
from time import time
import abc
import traceback
import datetime
import socket
import subprocess
import os
import sys
import pprint
import argparse
import re

import numpy as np
import tensorflow as tf

from dataset import Dataset
import abstract_model
import optimizer
import distmult_model
import complex_model
import evaluate


def train(arg_parser):
    '''Create an instance of model with the command line arguments and train it.

    Arguments:
    arg_parser -- An `argparse.ArgumentParser`.
    '''

    args = arg_parser.parse_args()
    if not args.em:
        args.num_samples = 1
    if args.model == 'DistMult':
        Model = distmult_model.DistMultModel
    elif args.model == 'ComplEx':
        Model = complex_model.ComplExModel

    # Get random seed from system if the user did not specify a random seed.
    if args.rng_seed is None:
        args.rng_seed = int.from_bytes(os.urandom(4), byteorder='little')
    rng = np.random.RandomState(seed=args.rng_seed)
    tf.set_random_seed(rng.randint(2**31))

    # Create the output directory.
    try:
        os.mkdir(args.output)
    except OSError:
        if not args.force:
            sys.stderr.write(
                'ERROR: Cannot create output directory %s\n' % args.output)
            sys.stderr.write(
                'HINT: Does the directory already exist? To prevent accidental data loss this\n'
                '      script, by default, does not write to an existing output directory.\n'
                '      Specify a non-existing output directory or use the `--force`.\n')
            exit(1)
    else:
        print('Writing output into directory `%s`.' % args.output)

    try:
        with open(os.path.join(args.output, 'log'), 'w') as log_file:
            # We write log files in the form of python scripts. This way, log files are both human
            # readable and very easy to parse by different python scripts. We begin log files with
            # a shebang (`#!/usr/bin/python`) so that text editors turn on syntax highlighting.
            log_file.write('#!/usr/bin/python\n')
            log_file.write('\n')

            # Log information about the executing environment to make experiments reproducible.
            log_file.write('program = "%s"\n' % arg_parser.prog)
            log_file.write(
                'args = {\n %s\n}\n\n' % pprint.pformat(vars(args), indent=4)[1:-1])
            log_file.write('git_revision = "%s"\n' % subprocess.check_output(
                ['git', 'rev-parse', 'HEAD']).decode('utf-8').strip())
            log_file.write('host_name = "%s"\n' % socket.gethostname())
            log_file.write('start_time = "%s"\n' %
                           str(datetime.datetime.now()))
            log_file.write('\n')

            dat = Dataset(args.input, log_file=log_file)
            model = Model(args, dat, rng, log_file=log_file)
            session = tf.Session()
            session.run(tf.initializers.global_variables())
            if args.initialize_from is not None:
                load_checkpoint(model, session, args.initialize_from,
                                log_file=log_file)

            evaluator = evaluate.Evaluator(model, dat, args, log_file=log_file)

            training_loop(args, model, session, dat, rng, evaluator,
                          log_file=log_file)

            log_file.write('\n')
            log_file.write('end_time = "%s"\n' %
                           str(datetime.datetime.now()))
    except:
        with open(os.path.join(args.output, 'err'), 'w') as err_file:
            traceback.print_exc(file=err_file)
        exit(2)


def training_loop(args, model, session, dat, rng, evaluator, log_file=sys.stdout):
    log_file.write('\n# Starting training loop.\n')

    step, = session.run(
        [var for var in tf.global_variables() if var.name == 'training_step:0'])
    initial_summaries = args.initial_summaries + step
    log_file.write('pretrained_steps = %d\n' % step)
    log_file.write('\n')
    log_file.write('progress = [\n')
    log_file.flush()

    summary_writer = tf.summary.FileWriter(args.output, session.graph)
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=2)

    # Implement early stopping using the filtered MRR on the validation set. We don't actually stop
    # training when the MRR starts going down, but we keep a checkpoint from peak MRR.
    max_mrr_so_far = 0.0

    opt_step = model.e_step
    for epoch in range(args.epochs):
        if args.em and epoch == args.initial_e_epochs:
            log_file.write('# Starting hyperparameter optimization.\n')
            log_file.flush()
            opt_step = model.em_step

        if epoch % args.epochs_per_eval == 0:
            valid_mrr = evaluator.run(session, summary_writer, step, epoch,
                                      log_file=log_file)
            if valid_mrr is not None and valid_mrr > max_mrr_so_far:
                max_mrr_so_far = valid_mrr
                save_checkpoint(args.output, session, step, saver, log_file)

        for minibatch in dat.iterate_in_minibatches('train', args.minibatch_size, rng):
            step += 1
            if step % args.steps_per_summary == 0 or step <= initial_summaries:
                _, summary = session.run([opt_step, model.summary_op],
                                         feed_dict={model.minibatch_htr: minibatch})
                summary_writer.add_summary(summary, global_step=step)
            else:
                session.run(opt_step, feed_dict={
                            model.minibatch_htr: minibatch})

    save_checkpoint(args.output, session, step, saver, log_file)

    evaluator.run(session, summary_writer, step, args.epochs,
                  log_file=log_file)
    log_file.write(']\n')


def save_checkpoint(directory, session, step, saver, log_file=sys.stdout):
    '''Save the current state of the model to a tensorflow checkpoint.

    Arguments:
    directory -- Output directory. Must exist. Any files with clashing names
        in the output directory will be overwritten.
    session -- A `tf.Session` that contains the state.
    step -- Integer number of concluded training steps.
    saver -- A `tf.train.Saver`.
    log_file -- File handle to the log file.
    '''

    start_time = time()
    log_file.write('# Saving checkpoint after step %d... ' % step)
    log_file.flush()

    saver.save(session, os.path.join(directory, 'checkpoint'),
               global_step=step)

    log_file.write('done. (%.2g seconds)\n' % (time() - start_time))
    log_file.flush()


def load_checkpoint(model, session, path, log_file=sys.stdout):
    log_file.write('# Loading model from checkpoint %s\n' % path)
    log_file.flush()

    reader = tf.train.NewCheckpointReader(path)
    checkpoint_variables = set(reader.get_variable_to_shape_map().keys())

    # Map "scope/var_name:0" to "scope/var_name".
    trimmer = re.compile(r'(.+):\d+$')

    def trim(s):
        return trimmer.match(s).group(1)

    model_variables = set(trim(var.name) for var in tf.global_variables())
    try:
        model_variables.remove('training_step')
    except:
        pass

    restored_variables = sorted(list(
        model_variables.intersection(checkpoint_variables)))
    ignored_in_checkpoint = sorted(list(
        checkpoint_variables - model_variables))
    not_in_checkpoint = sorted(list(
        model_variables - checkpoint_variables))

    log_file.write('restored_variables = [\n %s\n]\n\n'
                   % pprint.pformat(restored_variables, indent=4)[1:-1])
    log_file.write('not_found_in_checkpoint = [\n %s\n]\n\n'
                   % pprint.pformat(not_in_checkpoint, indent=4)[1:-1])
    log_file.write('found_in_checkpoint_but_not_restored = [\n %s\n]\n\n'
                   % pprint.pformat(ignored_in_checkpoint, indent=4)[1:-1])
    log_file.flush()

    loader = tf.train.Saver(
        var_list=[var for var in tf.global_variables() if trim(var.name) in restored_variables])
    loader.restore(session, path)

    log_file.write('# Done loading model from checkpoint.\n')
    log_file.flush()


def add_cli_args(parser):
    '''Add generic command line arguments.

    This function defines command line arguments that are required for all
    models in this project.

    Arguments:
    parser -- An `argparse.ArgumentParser`.

    Returns:
    A tuple of two command line argument groups that were added to the parser.
    '''

    positional_args = parser.add_argument_group(
        'Required positional arguments')
    positional_args.add_argument('input', metavar='IN_PATH', help='''
        Path to a directory containing the training, validation, and test data sets.''')
    positional_args.add_argument('output', metavar='OUT_PATH', help='''
        Path to the output directory. Must not already exist (unless --force is used).''')

    train_args = parser.add_argument_group(
        'Parameters of the training environment')
    train_args.add_argument('-f', '--force', action='store_true', help='''
        Allow writing into existing output directory, possibly overwriting existing files.''')
    train_args.add_argument('-E', '--epochs', metavar='N', type=int, default=500, help='''
        Set the number of training epochs.''')
    train_args.add_argument('--initial_e_epochs', metavar='N', type=int, default=50, help='''
        Set the number of initial epochs in which the hyperparameters are kept constant (i.e., only
        the "E-step" but not the "M-step" is performed during these initial epochs). Only used if
        `--em` is set.''')
    train_args.add_argument('-B', '--minibatch_size', metavar='N', type=int, default=100, help='''
        Set the minibatch size.''')
    train_args.add_argument('--rng_seed', metavar='N', type=int, help='''
        Set the seed of the pseudo random number generator. If not provided, a
        seed will automatically be generated from a system random source. In order to
        make experiments reproducible, the seed is always written to the output file,
        along with the git commit hash and all command line arguments.''')
    train_args.add_argument('--steps_per_summary', metavar='N', type=int, default=100, help='''
        Set the number of training steps to run between generating a Tensorboard summary.''')
    train_args.add_argument('--initial_summaries', metavar='N', type=int, default=100, help='''
        Set the number of initial training steps for which a Tensorboard summary will be generated
        after every step.''')
    train_args.add_argument('--epochs_per_eval', metavar='N', type=int, default=1, help='''
        Set the number of training epochs to run between model evaluations.''')
    train_args.add_argument('--initialize_from', metavar='PATH', help='''
        Provide a path to a tensorflow checkpoint file from which to load initial model parameters,
        initial hyperparameters, and the number of already concluded training steps. The provided
        PATH will likely have the form `directory/checkpoint-NNN` where `NNN` is the step number.
        Note that any internal state of optimizers, such as momentum or other accumulators, is not
        restored. Values stored in the checkpoint file take precedence over any initializations
        provided by command line arguments. This operation can be used both to initialize EM with a
        point estimated model, as well as to initialize point estimation with the means and
        hyperparametersfrom EM. The operation restores the intersection between parameters in the
        checkpoint and parmeters of the new model, and reports the names of the restored parameters
        to the log file.''')
    return positional_args, train_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
        Train a probabilistic knowledge graph embedding model.''')
    add_cli_args(parser)
    abstract_model.add_cli_args(parser)
    optimizer.add_cli_args(parser)
    evaluate.add_cli_args(parser)

    train(parser)
