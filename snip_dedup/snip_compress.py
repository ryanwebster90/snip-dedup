"""snip compress"""
import os
import os.path
import sys
import fire
import numpy as np
import torch


def _validate_shards(shards):
    shards_bounds = shards.split(":")
    try:
        shards_bounds = [int(shard) for shard in shards_bounds]
    except Exception:
        sys.exit(
            f'The shards pattern "{shards}" is not valid. It should be a valid range such as "0:" or "14:42"'
        )
    if len(shards_bounds) == 0:
        sys.exit("The --shards argument cannot be empty")
    elif len(shards_bounds) == 1:
        sys.exit(
            "Single value is not accepted for --shards as it's ambiguous between wanting only that exact shard or that number of shards starting from 0."
        )
    elif len(shards_bounds) > 2:
        sys.exit(
            "Ranges with more than 2 parts, such as 0:2:14 are not valid for --shard. Please limit yourself with simple ranges such as 0:14"
        )
    start_shard, end_shard = shards_bounds
    if start_shard < 0 or end_shard < 0:
        sys.exit("Only positive integers are allowed for --shards, such as 0:14")
    if end_shard <= start_shard:
        sys.exit(
            'The --shards argument must be of the shape "s:e" with s < e, such as "0:1" or "14:42". The "e" bound is excluded.'
        )
    return start_shard, end_shard


# compute features over chunks
@torch.no_grad()
def compute_feats_for_chunk(net, chunk, batch_size=256):
    feats = []
    for b in range(0, chunk.shape[0], batch_size):
        end_ind = min(b + batch_size, chunk.shape[0])
        batch = chunk[b:end_ind, :]
        feats += [net(torch.from_numpy(batch).float().cuda()).cpu().numpy()]
    feats = np.concatenate(feats, axis=0)
    return feats


def snip_compress(
    snip_model_path,
    shards="0:2",
    clip_feats="clip_feats/{shard:04d}.npy",
    snip_feats_out="snip_feats/{shard:04d}.npy",
):
    """Compress frozen clip features with SNIP

    Parameters
    ----------
    feats_folder : str
        folder with .npy clip features
    model_folder : str
        folder with snip models
    snip_model : str, optional
        Snip model artifact which compresses feats (see available models at fraisdufour/snip-dedup on hf)
    snip_index : str, optional
        snip index built on top of compressed feats
    download_models : bool, optional
        download the model and index
    """
    # Check that the SNIP model path passed as argument is valid
    if not os.path.isfile(snip_model_path):
        sys.exit(
            f'The SNIP model file "{snip_model_path}" does not exist or is not readable.'
        )

    # Check that the shards argument is correct
    start_shard, end_shard = _validate_shards(shards)

    # Check that the CLIP features exist
    for shard in range(start_shard, end_shard):
        clip_shard_path = clip_feats.format(shard=shard)
        if not os.path.isfile(clip_shard_path):
            sys.exit(
                f'The CLIP file for shard {shard} does not exist: "{clip_shard_path}"'
            )

    # Create directory for the computed SNIP shards
    try:
        first_snip_shard_path = snip_feats_out.format(shard=start_shard)
        snip_out_parent_dir = os.path.dirname(first_snip_shard_path)
        os.makedirs(snip_out_parent_dir, exist_ok=True)
    except Exception:
        sys.exit(
            f'Something is wrong with the output file paths specified for --snip_feats_out "{snip_feats_out}"'
        )

    # Load SNIP net
    net = torch.load(snip_model_path).eval().cuda()

    # Compute SNIP features for all shards
    print(f"Start computing SNIP features for shards {start_shard} to {end_shard} (excluded)")
    for shard in range(start_shard, end_shard):
        print(f"  Computing SNIP features for shard {shard} ...")
        # this is normally the bottleneck
        clip_shard_path = clip_feats.format(shard=shard)
        clip_shard = np.load(clip_shard_path)
        # compute SNIP features
        snip_feats = compute_feats_for_chunk(net, clip_shard)
        snip_shard_path = snip_feats_out.format(shard=shard)
        np.save(snip_shard_path, snip_feats)


if __name__ == "__main__":
    fire.Fire(snip_compress)
