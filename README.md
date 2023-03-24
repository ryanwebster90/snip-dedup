# snip-dedup

[![PyPI - Version](https://img.shields.io/pypi/v/snip-dedup.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/snip-dedup/)
[![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff)
[![format - Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license - MIT](https://img.shields.io/badge/license-MIT-9400d3.svg)](https://spdx.org/licenses/)
[![license - MIT](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nKccWcCz566qDg3AohTV-zjBn7u_INnG?usp=sharing)

## This repo is a WIP

This repo is a WIP, but the main functionalities will be:

- [x] Download de-duplicated versions of LAION-2B-en (Better versions coming soon...)
- [ ] Download small indices (25-40GB) for retrieval / dataset creation / de-duplciation
- [ ] Compress features using pretrained SNIP networks (for ViT-H-14, ViT-L14, ViT-B-32)
- [x] Read our research paper
- [ ] Train SNIP on your CLIP features
- [ ] Run a de-duplication of your dataset using our de-dup code

SNIP is a technique to compress CLIP features. It is competitive with previous works for large scale retrieval of deep features, and has some nice properties for multi-modal features. Read more about it [here](https://arxiv.org/abs/2303.12733). 

We used SNIP to perform several de-duplications of LAION-2B-en. Our latest de-duplication found roughly 700M duplicates (we define total duplicates as total samples - duplicate groups). SNIP performs well at high compression ratios and can run at very high q/s with low memory.

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

See the colab [![license - MIT](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nKccWcCz566qDg3AohTV-zjBn7u_INnG?usp=sharing) for a demo on search.

## What is a Duplicate?

In our first iteration, we merely marked duplicates pairwise, and remove one sample from a duplicate pair (the above code downloads a binary array, for samples to remove). In our latest run, we recorded the entire adjacency matrix of duplication. For instance, suppose SNIP has labeled feature $k$ as a duplicate with feature $j$. Then $A[k,j] = A[j,k] = 1$ in the adjacency matrix. We're currently having trouble computing the full connected components of this matrix, see [this issue](https://github.com/ryanwebster90/snip-dedup/issues/7#issue-1639736690). 

If you allow connected components with only one node, Then to compute the number of "unique" samples, you simply take one from each duplicate set, say $|\mathcal{C}|$ sets, with $N$ nodes is $D := N - |\mathcal{C}|$ duplicates.

### Approximate CCs of Duplicates

Currently, we have an approximation of the CC of the duplicates, as follows. During the de-duplication, we label nodes as follows. Suppose we are at node $n$, the pseudo code for one step of labeling is as follows
```python
labels = np.arange(0,N)
...
d,i = index.search(feats[n,:],k)
dups = get_dups(d,i) #Use adaptive threshhold on ADC (see paper)
label[dups] = resolve_labels_one_step(dups)
```
Where `N` is number of nodes (2B for L2B). Here `resolve_labels_one_step` will simply re-write any node that is unlabeled to be the current node $n$. This can be thought of as a tree. We then connect nodes with common ancestors with a fixed point
```python
while True:
      label = label[label]
```

We have

## Misc files (old)

We release this index for public use and exploration of the LAION-2B-en dataset.

You may find the following necessary files here:

[Binary array of De-duplicated Images](https://drive.google.com/file/d/1RYDylZKaPyaVs5YNwIrGqHU2BewdFwxY/view?usp=sharing)

[SNIP index](https://drive.google.com/file/d/1RYDylZKaPyaVs5YNwIrGqHU2BewdFwxY/view?usp=sharing)

[SNIP descriptor](https://drive.google.com/file/d/1QTA9yWevwPMhvMW8P5mAIBDy42xUpr-m/view?usp=share_link)

Other:

[cumulative sizes of features (for indexing sharded files)](https://drive.google.com/file/d/1OdVt5rjYw55XfMhsQSdqcVOP7lG2qj4W/view?usp=sharing)

## Finding images overfit by Stable Diffusion

By analyzing the most duplicated images, we have found several more images verbatim copied by Stable Diffusion, posing a copyright problem:

![sylvester_overfit](https://user-images.githubusercontent.com/2905865/225423740-e0befaba-cb74-44bf-9a64-f5dd9cbd4c33.jpeg)
![hopped up logo](https://user-images.githubusercontent.com/2905865/225423836-7c64428b-6782-4452-8d29-1628dc192c6c.jpeg)


## Note on False positives
We noticed many images labled as dup by SNIP but not by raw feats are in fact newar duplicates, for example:

![Chess1](https://en.chessok.net/uploads/posts/2017-09/1506718434_knight-on-the-left-1.nc3.jpg)
![Chess2](https://m.media-amazon.com/images/I/51jNRpWUCjL.jpg)

you may check a list of (randomly sampled) detected duplicate pairs [here](https://docs.google.com/spreadsheets/d/1Eq46U3MbTXzNoLCvnHLcw64X3bWE3ZE8zMJVQU9_gCg/edit?usp=sharing)


## Semantic Search

SNIP can also be used for semantic search. At just 25GB, it still can return the same k-NN's compared to exhaustive search roughly a third of the time, over 2.15B database vectors. 

## Contribute

Contributions are welcome.
Usually, the best way is first to open an issue to discuss things.

This python project uses the [`hatch`][hatch] project manager.
Dependencies are specified inside the `pyproject.toml` file, and build configs inside the `hatch.toml` file.
As such you can enter the isolated development environment with `hatch shell` from inside the repository.

The code should be documented following the [Numpy docstring standard][docstring].

To avoid silly mistakes, the code is checked with [pyright][pyright].
To ensure a consistent styling, all python code is formatted with [black][black] and we use the [ruff][ruff] linter.
Remark that these can usually get installed in your editor, such as VS Code, to view the checks directly in the code.
Once you have installed them (suggested via [pipx][pipx]), you can check that the code is consistent with:

```sh
hatch run check           # check for mistakes via static analysis with pyright
black --check snip_dedup/ # check formatting of all python files
ruff check snip_dedup/    # check linting rules
```

STILL TODO:

- [ ] add docs / tutorial
- [ ] add tests
- [ ] check max file size on CI to prevent pushing data
- [ ] auto publish github action. example at https://github.com/ofek/hatch-showcase/blob/master/.github/workflows/build.yml

[hatch]: https://github.com/pypa/hatch
[pyright]: https://github.com/microsoft/pyright
[black]: https://github.com/psf/black
[ruff]: https://github.com/charliermarsh/ruff
[pipx]: https://github.com/pypa/pipx
[docstring]: https://numpydoc.readthedocs.io/en/latest/format.html

## Citation
```
@misc{webster2023deduplication,
      title={On the De-duplication of LAION-2B}, 
      author={Ryan Webster and Julien Rabin and Loic Simon and Frederic Jurie},
      year={2023},
      eprint={2303.12733},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

