import librosa as lr
import numpy as np
import torch
import scipy

JUKEBOX_SAMPLE_RATE = 44100 # Hz
T = 8192 # tokens (also CONTEXT LENGTH)

def load_audio_from_file(fpath):
    audio, _ = lr.load(fpath, sr=JUKEBOX_SAMPLE_RATE)
    if audio.ndim == 1:
        audio = audio[np.newaxis]
    audio = audio.mean(axis=0)

    # normalize audio
    norm_factor = np.abs(audio).max()
    if norm_factor > 0:
        audio /= norm_factor

    return audio.flatten()


def get_z(audio, vqvae):
    # don't compute unnecessary discrete encodings
    audio = audio[: JUKEBOX_SAMPLE_RATE * 25] # crop only the first 25 seconds
    zs = vqvae.encode(torch.cuda.FloatTensor(audio[np.newaxis, :, np.newaxis]))
    z = zs[-1].flatten()[np.newaxis, :] # top-level encoding
    if z.shape[-1] < 8192:
        raise ValueError("Audio file is not long enough")

    return z


def get_cond(hps, top_prior):
    sample_length_in_seconds = 62
    hps.sample_length = (
        int(sample_length_in_seconds * hps.sr) // top_prior.raw_to_tokens
    ) * top_prior.raw_to_tokens

    # NOTE: the 'lyrics' parameter is required, which is why it is included,
    # but it doesn't actually change anything about the `x_cond`, `y_cond`,
    # nor the `prime` variables
    metas = [
        dict(
            artist="unknown",
            genre="unknown",
            total_length=hps.sample_length,
            offset=0,
            lyrics="""lyrics go here!!!""",
        ),
    ] * hps.n_samples

    labels = [None, None, top_prior.labeller.get_batch_labels(metas, "cuda")]
    x_cond, y_cond, prime = top_prior.get_cond(None, top_prior.get_y(labels[-1], 0))
    x_cond = x_cond[0, :T][np.newaxis, ...]
    y_cond = y_cond[0][np.newaxis, ...]

    return x_cond, y_cond


def get_final_activations(z, x_cond, y_cond, top_prior):

    x = z[:, :T]

    # make sure that we get the activations
    top_prior.prior.only_encode = True

    # encoder_kv and fp16 are set to the defaults, but explicitly so
    out = top_prior.prior.forward(
        x, x_cond=x_cond, y_cond=y_cond, encoder_kv=None, fp16=False
    )

    return out


def get_acts_from_file(fpath, hps, vqvae, top_prior, meanpool=True):
    audio = load_audio_from_file(fpath)

    # run vq-vae on the audio
    z = get_z(audio, vqvae)

    # get conditioning info
    x_cond, y_cond = get_cond(hps, top_prior)

    # get the activations from the LM
    acts = get_final_activations(z, x_cond, y_cond, top_prior)

    # postprocessing
    acts = acts.squeeze().type(torch.float32)

    if meanpool:
        acts = acts.mean(dim=0)

    acts = np.array(acts.cpu())

    return acts, z


if __name__ == "__main__":
    import pathlib
    from argparse import ArgumentParser

    # imports and set up Jukebox's multi-GPU parallelization
    import jukebox
    from jukebox.hparams import Hyperparams, setup_hparams
    from jukebox.make_models import MODELS, make_prior, make_vqvae
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    from tqdm import tqdm

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--batch_idx", type=int)
    parser.add_argument("--representations", type=str)
    #parser.add_argument("--depth", type=int)

    parser.set_defaults(
        batch_size=None,
        batch_idx=None,
        representations="mean_pool,max_pool",
    )
    args = parser.parse_args()

    input_dir = pathlib.Path("input")
    output_dir = pathlib.Path("output")
    output_dir.mkdir(exist_ok=True)
    input_paths = sorted(list(input_dir.iterdir()))
    if args.batch_size is not None and args.batch_idx is not None:
        batch_starts = list(range(0, len(input_paths), args.batch_size))
        if args.batch_idx >= len(batch_starts):
            raise ValueError("Invalid batch index")
        batch_start = batch_starts[args.batch_idx]
        input_paths = input_paths[batch_start : batch_start + args.batch_size]

    #prior_depth = args.depth
    #if args.representations is not "mean_pool,max_pool":
    #    meanpool = False
    #else:
    #    meanpool = True

    #--------------------------------
    #TODO: FIX the ArgumentParser
    # for now:
    #prior_depth = 36
    #layers_upto = 72
    layers_upto = 72
    meanpool = False
    #--------------------------------
    loaded = False
    # Set up MPI
    rank, local_rank, device = setup_dist_from_mpi()
    
    for input_path in tqdm(input_paths):
        # Check if output already exists
        for prior_depth in range(1,layers_upto+1): # because range[a,b)
            output_path = pathlib.Path(output_dir, '%s_depth%02i.mat' %(input_path.stem, prior_depth))
            outz_path = pathlib.Path(output_dir, '%s_z02.mat' %(input_path.stem))

            if not pathlib.os.path.isfile(output_path):

                # Set up VQVAE
                model = "5b"  # or "1b_lyrics" or "5b_lyrics"
                hps = Hyperparams()
                hps.sr = 44100
                hps.n_samples = 3 if model == "5b_lyrics" else 8
                hps.name = "samples"
                chunk_size = 16 if model == "5b_lyrics" else 32
                max_batch_size = 3 if model == "5b_lyrics" else 16
                hps.sample_length = 1048576 if "5b" in model else 786432
                hps.levels = 3
                hps.hop_fraction = [0.5, 0.5, 0.125]
                vqvae, *priors = MODELS[model]
                vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length=hps.sample_length)), device)

                # Set up language model
                hparams = setup_hparams(priors[-1], dict())
                hparams["prior_depth"] = prior_depth
                top_prior = make_prior(hparams, vqvae, device)

                loaded = True

                # Decode, resample, convert to mono, and normalize audio
                with torch.no_grad():
                    acts, top_z = get_acts_from_file(
                        input_path, hps, vqvae, top_prior, meanpool=meanpool
                    ) # nparray

                # Save activation in a .MAT format
                hop = vqvae.hop_lengths[-1]
                times = np.arange(hop, (8192+1)*hop, hop) / JUKEBOX_SAMPLE_RATE
                scipy.io.savemat(output_path, dict(Acts=acts, Times=times))
                
                if not pathlib.os.path.isfile(outz_path):
                    audio = load_audio_from_file(input_path)[: JUKEBOX_SAMPLE_RATE * 25] # crop only the first 25 seconds
                    zs = vqvae.encode(torch.cuda.FloatTensor(audio[np.newaxis, :, np.newaxis])) # list with tensors
                    #Z = [this.cpu().numpy() for this in zs]
                    #scipy.io.savemat(outz_path, dict(Z0=Z[0], Z1=Z[1], Z2=Z[2]))
                    for z, hop, i in zip(zs, vqvae.hop_lengths, range(0,3)):
                        times = np.arange(hop, JUKEBOX_SAMPLE_RATE * 25, hop) / JUKEBOX_SAMPLE_RATE
                        outz_path = pathlib.Path(output_dir, '%s_z%02i.mat' %(input_path.stem, i))
                        scipy.io.savemat(outz_path, dict(Z=z.cpu().numpy(), Times=times))

                
                # Clear the models for the next loop
                del top_prior, vqvae      # removing references
                torch.cuda.empty_cache()  # removing garbage


