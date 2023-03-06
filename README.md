# snip-dedup

SNIP is a very compact index (25GB) that has found roughly half a billion duplicates on the LAION-2B-en dataset (or roughly 1/4 the images), taking just several days on a standard computer with 32GB of RAM. We believe the community will benefit from such a dataset, in light of recent research showing the copyright and privacy risks associated with training generative models on highly duplicated datasets.


** DISCLAIMER ** 
Due to the scale of the dataset, there is still a chance our analysis is false. Use at your own risk. Help for better de-duiplication (higher acc, higher recall) is very much appreciated. Taking raw CLIP features as the ground truth for exact duplicates, we get nearly 81% precision (and likely much higher for near duplicates, see below).

We release this index for public use and exploration of the LAION-2B-en dataset (more indices coming soon). Soon we will release tools to train your own SNIP indices as well as our scientific paper discussing the method in more detail.

You may find the following necessary files here:

[Binary array of De-duplicated Images](https://drive.google.com/file/d/1RYDylZKaPyaVs5YNwIrGqHU2BewdFwxY/view?usp=sharing)

[SNIP index](https://drive.google.com/file/d/1RYDylZKaPyaVs5YNwIrGqHU2BewdFwxY/view?usp=sharing)

[SNIP descriptor](https://drive.google.com/file/d/1QTA9yWevwPMhvMW8P5mAIBDy42xUpr-m/view?usp=share_link)

Other:

[cumulative sizes of features (for indexing sharded files)](https://drive.google.com/file/d/1OdVt5rjYw55XfMhsQSdqcVOP7lG2qj4W/view?usp=sharing)


## Setup
Required packages are torch, faiss, numpy and fire.

**
We assume you have all the metadata for laion-2b-en vith14 and some of the features for verification. You may download the parquet files [here](https://huggingface.co/datasets/laion/laion2B-en). After, you may use the file "is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy" above to remove all non-duplicates. The index within this array corresponds to the index within the parquet files.

You may then create a set with duplicates removed with

python dedup_l2b.py /path/to/metadata /output/path

which will save the de-dup'd metadata to /output/path. You may also check the fidelity of the duplicates by randomly sampling labeled duplicates, and using SNIP to detect its dup. You may do that with retrieve_dup_urls_demo.py.

## Finding images overfit by Stable Diffusion

We have found several more images verbatim copied by Stable Diffusion:




![Chess1](https://en.chessok.net/uploads/posts/2017-09/1506718434_knight-on-the-left-1.nc3.jpg)
![Chess2](https://m.media-amazon.com/images/I/51jNRpWUCjL.jpg)

## Semantic Search

SNIP can also be used for semantic search. At just 25GB, it still can return the same k-NN's compared to exhaustive search roughly a third of the time, over 2.15B database vectors. 














