import pathlib
from argparse import ArgumentParser

# imports and set up Jukebox's multi-GPU parallelization
import jukebox
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.make_models import MODELS, make_prior, make_vqvae
from jukebox.utils.dist_utils import setup_dist_from_mpi
from tqdm import tqdm

rank, local_rank, device = setup_dist_from_mpi()
device = "cuda"
model = "5b"  # or "1b_lyrics" or "5b_lyrics"

# Set up VQ-VAE model:
hps = Hyperparams()
hps.sr = 44100
hps.n_samples = 3 if model == "5b_lyrics" else 8
hps.name = "samples"
chunk_size = 16 if model == "5b_lyrics" else 32
max_batch_size = 3 if model == "5b_lyrics" else 16
hps.sample_length = 1048576 if model=="5b_lyrics" else 786432 
hps.levels = 3
hps.hop_fraction = [0.5, 0.5, 0.125]
vqvae, *priors = MODELS[model]
vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length=hps.sample_length)), device)

# Set up language model
hparams = setup_hparams(priors[-1], dict())
hparams["prior_depth"] = 1
top_prior = make_prior(hparams, vqvae, device)

"""
In [2]: vqvae
Out[2]: 
VQVAE(
  (encoders): ModuleList(
    (0): Encoder(
      (level_blocks): ModuleList(
        (0): EncoderConvBlock(
          (model): Sequential(
            (0): Sequential(
              (0): Conv1d(1, 64, kernel_size=(4,), stride=(2,), padding=(1,))
              (1): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (4): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(81,), dilation=(81,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (5): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(243,), dilation=(243,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (6): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(729,), dilation=(729,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (7): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(2187,), dilation=(2187,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
            )
            (1): Sequential(
              (0): Conv1d(64, 64, kernel_size=(4,), stride=(2,), padding=(1,))
              (1): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (4): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(81,), dilation=(81,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (5): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(243,), dilation=(243,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (6): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(729,), dilation=(729,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (7): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(2187,), dilation=(2187,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
            )
            (2): Sequential(
              (0): Conv1d(64, 64, kernel_size=(4,), stride=(2,), padding=(1,))
              (1): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (4): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(81,), dilation=(81,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (5): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(243,), dilation=(243,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (6): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(729,), dilation=(729,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (7): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(2187,), dilation=(2187,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
            )
            (3): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
          )
        )
      )
    )
    (1): Encoder(
      (level_blocks): ModuleList(
        (0): EncoderConvBlock(
          (model): Sequential(
            (0): Sequential(
              (0): Conv1d(1, 32, kernel_size=(4,), stride=(2,), padding=(1,))
              (1): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
            )
            (1): Sequential(
              (0): Conv1d(32, 32, kernel_size=(4,), stride=(2,), padding=(1,))
              (1): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
            )
            (2): Sequential(
              (0): Conv1d(32, 32, kernel_size=(4,), stride=(2,), padding=(1,))
              (1): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
            )
            (3): Conv1d(32, 64, kernel_size=(3,), stride=(1,), padding=(1,))
          )
        )
        (1): EncoderConvBlock(
          (model): Sequential(
            (0): Sequential(
              (0): Conv1d(64, 32, kernel_size=(4,), stride=(2,), padding=(1,))
              (1): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
            )
            (1): Sequential(
              (0): Conv1d(32, 32, kernel_size=(4,), stride=(2,), padding=(1,))
              (1): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
            )
            (2): Conv1d(32, 64, kernel_size=(3,), stride=(1,), padding=(1,))
          )
        )
      )
    )
    (2): Encoder(
      (level_blocks): ModuleList(
        (0): EncoderConvBlock(
          (model): Sequential(
            (0): Sequential(
              (0): Conv1d(1, 32, kernel_size=(4,), stride=(2,), padding=(1,))
              (1): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
            )
            (1): Sequential(
              (0): Conv1d(32, 32, kernel_size=(4,), stride=(2,), padding=(1,))
              (1): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
            )
            (2): Sequential(
              (0): Conv1d(32, 32, kernel_size=(4,), stride=(2,), padding=(1,))
              (1): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
            )
            (3): Conv1d(32, 64, kernel_size=(3,), stride=(1,), padding=(1,))
          )
        )
        (1): EncoderConvBlock(
          (model): Sequential(
            (0): Sequential(
              (0): Conv1d(64, 32, kernel_size=(4,), stride=(2,), padding=(1,))
              (1): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
            )
            (1): Sequential(
              (0): Conv1d(32, 32, kernel_size=(4,), stride=(2,), padding=(1,))
              (1): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
            )
            (2): Conv1d(32, 64, kernel_size=(3,), stride=(1,), padding=(1,))
          )
        )
        (2): EncoderConvBlock(
          (model): Sequential(
            (0): Sequential(
              (0): Conv1d(64, 32, kernel_size=(4,), stride=(2,), padding=(1,))
              (1): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
            )
            (1): Sequential(
              (0): Conv1d(32, 32, kernel_size=(4,), stride=(2,), padding=(1,))
              (1): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
            )
            (2): Conv1d(32, 64, kernel_size=(3,), stride=(1,), padding=(1,))
          )
        )
      )
    )
  )
  (decoders): ModuleList(
    (0): Decoder(
      (level_blocks): ModuleList(
        (0): DecoderConvBock(
          (model): Sequential(
            (0): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
            (1): Sequential(
              (0): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(2187,), dilation=(2187,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(729,), dilation=(729,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(243,), dilation=(243,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(81,), dilation=(81,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (4): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (5): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (6): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (7): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
              (1): ConvTranspose1d(64, 64, kernel_size=(4,), stride=(2,), padding=(1,))
            )
            (2): Sequential(
              (0): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(2187,), dilation=(2187,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(729,), dilation=(729,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(243,), dilation=(243,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(81,), dilation=(81,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (4): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (5): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (6): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (7): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
              (1): ConvTranspose1d(64, 64, kernel_size=(4,), stride=(2,), padding=(1,))
            )
            (3): Sequential(
              (0): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(2187,), dilation=(2187,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(729,), dilation=(729,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(243,), dilation=(243,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(81,), dilation=(81,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (4): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (5): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (6): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (7): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
              (1): ConvTranspose1d(64, 64, kernel_size=(4,), stride=(2,), padding=(1,))
            )
          )
        )
      )
      (out): Conv1d(64, 1, kernel_size=(3,), stride=(1,), padding=(1,))
    )
    (1): Decoder(
      (level_blocks): ModuleList(
        (0): DecoderConvBock(
          (model): Sequential(
            (0): Conv1d(64, 32, kernel_size=(3,), stride=(1,), padding=(1,))
            (1): Sequential(
              (0): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
              (1): ConvTranspose1d(32, 32, kernel_size=(4,), stride=(2,), padding=(1,))
            )
            (2): Sequential(
              (0): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
              (1): ConvTranspose1d(32, 32, kernel_size=(4,), stride=(2,), padding=(1,))
            )
            (3): Sequential(
              (0): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
              (1): ConvTranspose1d(32, 64, kernel_size=(4,), stride=(2,), padding=(1,))
            )
          )
        )
        (1): DecoderConvBock(
          (model): Sequential(
            (0): Conv1d(64, 32, kernel_size=(3,), stride=(1,), padding=(1,))
            (1): Sequential(
              (0): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
              (1): ConvTranspose1d(32, 32, kernel_size=(4,), stride=(2,), padding=(1,))
            )
            (2): Sequential(
              (0): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
              (1): ConvTranspose1d(32, 64, kernel_size=(4,), stride=(2,), padding=(1,))
            )
          )
        )
      )
      (out): Conv1d(64, 1, kernel_size=(3,), stride=(1,), padding=(1,))
    )
    (2): Decoder(
      (level_blocks): ModuleList(
        (0): DecoderConvBock(
          (model): Sequential(
            (0): Conv1d(64, 32, kernel_size=(3,), stride=(1,), padding=(1,))
            (1): Sequential(
              (0): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
              (1): ConvTranspose1d(32, 32, kernel_size=(4,), stride=(2,), padding=(1,))
            )
            (2): Sequential(
              (0): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
              (1): ConvTranspose1d(32, 32, kernel_size=(4,), stride=(2,), padding=(1,))
            )
            (3): Sequential(
              (0): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
              (1): ConvTranspose1d(32, 64, kernel_size=(4,), stride=(2,), padding=(1,))
            )
          )
        )
        (1): DecoderConvBock(
          (model): Sequential(
            (0): Conv1d(64, 32, kernel_size=(3,), stride=(1,), padding=(1,))
            (1): Sequential(
              (0): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
              (1): ConvTranspose1d(32, 32, kernel_size=(4,), stride=(2,), padding=(1,))
            )
            (2): Sequential(
              (0): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
              (1): ConvTranspose1d(32, 64, kernel_size=(4,), stride=(2,), padding=(1,))
            )
          )
        )
        (2): DecoderConvBock(
          (model): Sequential(
            (0): Conv1d(64, 32, kernel_size=(3,), stride=(1,), padding=(1,))
            (1): Sequential(
              (0): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
              (1): ConvTranspose1d(32, 32, kernel_size=(4,), stride=(2,), padding=(1,))
            )
            (2): Sequential(
              (0): Resnet1D(
                (model): Sequential(
                  (0): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(27,), dilation=(27,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (1): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(9,), dilation=(9,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (2): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                  (3): ResConv1DBlock(
                    (model): Sequential(
                      (0): ReLU()
                      (1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
                      (2): ReLU()
                      (3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
                    )
                  )
                )
              )
              (1): ConvTranspose1d(32, 64, kernel_size=(4,), stride=(2,), padding=(1,))
            )
          )
        )
      )
      (out): Conv1d(64, 1, kernel_size=(3,), stride=(1,), padding=(1,))
    )
  )
  (bottleneck): Bottleneck(
    (level_blocks): ModuleList(
      (0): BottleneckBlock()
      (1): BottleneckBlock()
      (2): BottleneckBlock()
    )
  )
)

"""


import librosa as lr
import numpy as np
import torch
import scipy
JUKEBOX_SAMPLE_RATE = 44100 # Hz
T = 8192 # tokens (also CONTEXT LENGTH)
audio, _ = lr.load('input/sine1khz.wav', sr=JUKEBOX_SAMPLE_RATE)
audio = audio[: JUKEBOX_SAMPLE_RATE * 25] # crop only the first 25 seconds
zs = vqvae.encode(torch.cuda.FloatTensor(audio[np.newaxis, :, np.newaxis]))


#------------------------
# Set up language model
#hparams = setup_hparams(priors[-1], dict())
#hparams["prior_depth"] = 100
#top_prior = make_prior(hparams, vqvae, device)

