# Code for Paper "Augmenting and Tuning Knowledge Graph Embeddings"

This repository contains the code for the paper

- R. Bamler, F. Salehi, and S. Mandt,<br/>
  [*Augmenting and Tuning Knowledge Graph Embeddings*](paper.pdf),<br/>
  UAI 2019



## Quickstart

Set up and activate a python environment with Tensorflow. `cd` into the
directory of this `README.md` file and run the following command:

```
CUDA_VISIBLE_DEVICES=0 nohup python3 src/train.py dat/FB15K out/fb15k-map \
    --model ComplEx \
    -k 2000 \
    -B 100 \
    --initial_reg_strength 0.02 \
    --optimizer adagrad \
    --lr0_mu 0.02 \
    --eval_dat both \
    --epochs_per_eval 5 \
    --steps_per_summary 1000 \
    --lr_exponent 0.5 \
    --lr_offset 120786 &
    # (120786 steps corresponds to 25 epochs)
```

This starts a background process that fits a point estimation of the ComplEx
model to the FB15K data set using proportionally scaled regularizer strengths.

You can track the progress and predictive performances by reading the log file,
or graphically in tensorboard:

```
tensorboard --logdir out
```

It shows evaluation metrics as scalars under the tabs "eval_valid" (for results
on the validation set). By default, the script trains for 500 epochs, but this
is rarely needed for convergence. The script automatically imitates early
stopping by keeping a checkpoint of the training epoch with the highest MRR on
the validation set around.


## Variational EM

For variational EM, use the following command line options:

* `--em`: Activate variational EM.
* `--lr0_sigma`: (Initial) learning rate for the log standard deviations (we
  used the default value).
* `--lr0_lambda`: (Initial) learning rate for the hyperparameters (we used the
  default value).
* `--initialize_from`: Path to a checkpoint file with a pretrained model.

For a full list of available command line arguments, use `--help`.


## Implementing Your Own Models

To implement your own model, derive from the class `AbstractModel`. Use the
DistMult or ComplEx models, defined in the file `src/distmult_model.py` and
`src/complex_model.py` as a template. Overwrite the functions `define_emb` to
define the model parameters for entity and relation embeddings, and the function
`unnormalized_score` to implement the score `f`.



## Licences

Please see the file `LICENSES.md` in the `dat` directory for licenses of the
packaged data sets.
