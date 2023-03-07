# snip-dedup

SNIP is a very compact index (25GB) that has found roughly half a billion duplicates on the LAION-2B-en dataset. You may download the de-duplicated dataset below.

SNIP de-duplicated L2B on a standard home computer, taking just several days. We believe the community will benefit from such a dataset, in light of recent research showing the copyright and privacy risks associated with training generative models on highly duplicated datasets, as well as SNIP for a de-duplication, compression and retrieval tool.

## Install

```sh
pip install --upgrade snip-dedup
```

## Usage

```sh
# List available commands
snip --help
snip download --help

# Download and deduplicate the 10 first shards of the dataset
snip download --start 0 --end 10
```

Then, you may download (deduplicated) laion2b images with the awesome [img2dataset](https://github.com/rom1504/img2dataset).

You may check the fidelity of the duplicates by randomly sampling labeled duplicates, and using SNIP to detect its dup. You may do that with retrieve_dup_urls_demo.py (note you will need the original metadata files for this)

## Roadmap

You can also do with SNIP (coming soon...)
- [ ] Train SNIP Indices on your features
- [ ] Download full or sharded SNIP indices for various CLIP networks
- [ ] Do semantic search with extremely compact indices (25 GB or less) on billions of images
- [ ] Compress your features with SNIP descriptors
- [ ] Read our research paper

## About

** DISCLAIMER ** 
Use at your own risk. Help for better de-duiplication (higher acc, higher recall) is very much appreciated. Taking raw CLIP features as the ground truth for exact duplicates, we get nearly 81% precision (and likely much higher for near duplicates, see below).

We release this index for public use and exploration of the LAION-2B-en dataset (more indices coming soon). Soon we will release tools to train your own SNIP indices as well as our scientific paper discussing the method in more detail.

You may find the following necessary files here:

[Binary array of De-duplicated Images](https://drive.google.com/file/d/1RYDylZKaPyaVs5YNwIrGqHU2BewdFwxY/view?usp=sharing)

[SNIP index](https://drive.google.com/file/d/1RYDylZKaPyaVs5YNwIrGqHU2BewdFwxY/view?usp=sharing)

[SNIP descriptor](https://drive.google.com/file/d/1QTA9yWevwPMhvMW8P5mAIBDy42xUpr-m/view?usp=share_link)

Other:

[cumulative sizes of features (for indexing sharded files)](https://drive.google.com/file/d/1OdVt5rjYw55XfMhsQSdqcVOP7lG2qj4W/view?usp=sharing)

## Finding images overfit by Stable Diffusion

By analyzing the most duplicated images, we have found several more images verbatim copied by Stable Diffusion, posing a copyright problem:

![sylvester stallone](https://github.com/ryanwebster90/snip-dedup/blob/main/sylvester_overfit.jpeg)
![hopped up logo](https://github.com/ryanwebster90/snip-dedup/blob/main/overfit_2.jpeg)


## Note on False positives
We noticed many images labled as dup by SNIP but not by raw feats are in fact newar duplicates, for example:

![Chess1](https://en.chessok.net/uploads/posts/2017-09/1506718434_knight-on-the-left-1.nc3.jpg)
![Chess2](https://m.media-amazon.com/images/I/51jNRpWUCjL.jpg)

you may check a list of (randomly sampled) detected duplicate pairs [here](https://docs.google.com/spreadsheets/d/1Eq46U3MbTXzNoLCvnHLcw64X3bWE3ZE8zMJVQU9_gCg/edit?usp=sharing)


## Semantic Search

SNIP can also be used for semantic search. At just 25GB, it still can return the same k-NN's compared to exhaustive search roughly a third of the time, over 2.15B database vectors. 
