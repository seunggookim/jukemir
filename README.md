# Jukebox for MIR transfer learning

This repository contains code for our paper [_Codified audio language modeling learns useful representations for music information retrieval_]() (Castellon et al. 2021), which demonstrates that [OpenAI's Jukebox](https://openai.com/blog/jukebox/) provides rich representations for MIR transfer learning.

This README is divided into two standalone sections. The [first section](#simple-example) is optimized for simplicity and provides an end-to-end example of genre detection using representations from Jukebox. The [second section](#reproducing-results) is optimized for reproducibility and provides step-by-step instructions for reproducing the results from our paper.

## Simple example of using Jukebox for transfer learning

## Reproducing results from our paper

This section provides step-by-step instructions for reproducing all results from our paper. All code is executed within [pre-built Docker containers](https://hub.docker.com/orgs/jukemir/repositories) to increase reproducibility.

### Setting up Docker and cache

If you do not already have Docker on your machine, please follow [these instructions](https://docs.docker.com/get-docker/) to install it.

To initialize the default cache directory, **run `mkdir ~/.jukemir`**. If you would prefer to use a different directory, set the appropriate environment variable: `export JUKEMIR_CACHE_DIR=<your desired directory>`.

To launch our pre-built Docker container in the background, **navigate to the [`reproduce`](reproduce/) folder and run `./0_docker.sh`**. You may kill and remove this container at any time by typing `docker kill jukemir`.

### Download pre-computed representations (skipping steps 1-3)

Unless you are probing a new dataset or comparing a new representation on an existing dataset, you can **skip steps 1-3 by downloading the pre-computed representations via `./123_precomputed.sh`**. If you need to selectively download particular pre-computed representations, see [`jukemir/assets/precomputed.json`](jukemir/assets/precomputed.json) for relevant URLs.

### (Step 1) Downloading the datasets

Once Docker is running, **run [`./1_download.sh`](reproduce/1_download.sh)** to download all of the raw dataset assets from their respective sources. Note that this downloads about 8GB of files and may take quite a long time depending on the speed of your network connection.

The resultant files will be downloaded to the `datasets` subdirectory of your cache directory (`~/.jukemir/cache/datasets` by default).

### (Step 2) Processing the datasets

Once the raw assets have been downloaded, **run [2_process.sh](reproduce/)** to process them into a standard format (parses all metadata and decodes MP3s to 16-bit PCM WAV files). Note that this script will also take several hours to run and will produce about 50GB of WAV files.

The resultant files will be saved to the `processed` subdirectory of your cache directory (`~/.jukemir/cache/processed` by default).

### (Step 3) Extracting representations

Next, we need to extract representations for each WAV file in the processed datasets. Note that this process is resource-intensive (especially for certain representations). Alternatively, you can [download pre-computed representations](#download-precomputed).

Each representation from our paper (`chroma`, `mfcc`, `choi`, `musicnn`, `clmr`, `jukebox`) has been packaged into a pre-built Docker container with a common API. The basic "type signature" of each container takes a folder of WAV files as input and returns a folder of Numpy arrays containing the corresponding representations. For example, if you have a folder in your current working directory called `mywavs/`, you can extract representations from Jukebox via the following command, which will create a folder of Numpy arrays called `mywavs_jukebox/`:

```sh
docker run \
	-it \
	--rm \
	-v $(pwd)/mywavs:/input \
	-v $(pwd)/mywavs_jukebox:/output \
	jukemir/representations_jukebox
```

Note that each container also takes two optional arguments as input, `--batch_size` and `--batch_idx`, which can be used to compute representations for a subset (batch) of the input WAV file directory. This is useful for parallelizing computation across several workers.

Because `choi` uses batchnorm during inference, its representations will differ if `--batch_size` is changed from its default value of `256` (not recommended). All other representations are invariant to batch size. Note that `musicnn`, `clmr`, and `jukebox` should be run on a machine with a GPU.

A Python script at [`scripts/3_extract.py`](scripts/3_extract.py) will generate all of the Docker commands needed to re-extract all representations for all datasets (see [`scripts/3_extract.sh`](scripts/3_extract.sh) for output). We highly recommend executing these commands in parallel in your own computing environment, as running them one at a time will take a long time.

### (Step 4) Running probing experiments

Individual probing experiments are defined using a `jukemir.probe.ProbeExperimentConfig` and executed via `jukemir.probe.execute_probe_experiment`. For example, assuming you've followed the previous steps, you can train a linear probe on `gtzan_ff` using features from `jukebox` via the following script:

```py
from jukemir.probe import ProbeExperimentConfig, execute_probe_experiment

cfg = ProbeExperimentConfig(dataset="gtzan_ff", representation="jukebox")
execute_probe_experiment(cfg)
```

To generate config files for the grid searches described in our paper, **run `4_grid_config.sh`**. The resultant files will be saved to the `probes` subdirectory of your cache directory (`~/.jukemir/cache/probes` by default).

### Aggregating grid search results and evaluating test performance

## Citation

If you use this codebase in your work, please consider citing our paper:

```
@article{castellon2021calm,
  title={Codified audio language modeling learns useful representations for music information retrieval},
  author={Rodrigo Castellon and Chris Donahue and Percy Liang},
  journal={arXiv:2107.},
  year={2021},
}
```
