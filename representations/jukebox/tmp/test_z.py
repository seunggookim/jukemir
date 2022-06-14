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
hps = Hyperparams()
hps.sr = 44100
hps.n_samples = 3 if model == "5b_lyrics" else 8
hps.name = "samples"
chunk_size = 16 if model == "5b_lyrics" else 32
max_batch_size = 3 if model == "5b_lyrics" else 16
hps.levels = 3
hps.hop_fraction = [0.5, 0.5, 0.125]
vqvae, *priors = MODELS[model]
vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length=1048576)), device)

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

