"""snip compress"""
import os
import os.path
import sys
import fire
import numpy as np
import torch

from . import _cli_helper


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
    snip_model_path="snip_models/snip_vitl14_128_deep.pth",
    shards="0:2",
    clip_feats="clip_feats/{shard:04d}.npy",
    snip_feats_out="snip_feats/{shard:04d}.npy",
):
    """Compress frozen CLIP features with SNIP

    Parameters
    ----------
    snip_model_path : str
        Path to the SNIP model file to use, might be something like: snip_models/snip_vitl14_128_deep.pth
    shards : str
        Shards to compress, using a slice notation, such as 0:2 or 14:42
    clip_feats : str
        Pattern referencing the path to the CLIP features shards.
        You are expected to use the "shard" variable with formatting options, such as "{shard:03d}.npy" which will be replaced by "001.npy" when shard==1.
    snip_feats_out : str
        Pattern referencing the path to the SNIP compressed features shards that will be computed.
        You are expected to use the "shard" variable with formatting options, such as "{shard:03d}.npy" which will be replaced by "001.npy" when shard==1.
    """
    # Check that the SNIP model path passed as argument is valid
    if not os.path.isfile(snip_model_path):
        sys.exit(
            f'The SNIP model file "{snip_model_path}" does not exist or is not readable.'
        )

    # Check that the shards argument is correct
    start_shard, end_shard = _cli_helper.validate_shards(shards)

    # Check that the CLIP features exist
    _cli_helper.validate_shard_format(clip_feats)
    for shard in range(start_shard, end_shard):
        clip_shard_path = clip_feats.format(shard=shard)
        if not os.path.isfile(clip_shard_path):
            sys.exit(
                f'The CLIP file for shard {shard} does not exist: "{clip_shard_path}"'
            )

    # Create directory for the computed SNIP shards
    _cli_helper.validate_shard_format(snip_feats_out)
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
    print(
        f"Start computing SNIP features for shards {start_shard} to {end_shard} (excluded)"
    )
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
