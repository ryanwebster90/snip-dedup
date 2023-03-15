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
    parts="0:2",
    clip_feats="clip_feats/{part:04d}.npy",
    snip_feats_out="snip_feats/{part:04d}.npy",
):
    """Compress frozen CLIP features with SNIP

    Parameters
    ----------
    snip_model_path : str
        Path to the SNIP model file to use, might be something like: snip_models/snip_vitl14_128_deep.pth
    parts : str
        Parts to compress, using a slice notation, such as 0:2 or 14:42
    clip_feats : str
        Pattern referencing the path to the CLIP features parts.
        You are expected to use the "part" variable with formatting options, such as "{part:03d}.npy" which will be replaced by "001.npy" when part==1.
    snip_feats_out : str
        Pattern referencing the path to the SNIP compressed features parts that will be computed.
        You are expected to use the "part" variable with formatting options, such as "{part:03d}.npy" which will be replaced by "001.npy" when part==1.
    """
    # Check that the SNIP model path passed as argument is valid
    if not os.path.isfile(snip_model_path):
        sys.exit(
            f'The SNIP model file "{snip_model_path}" does not exist or is not readable.'
        )

    # Check that the parts argument is correct
    start_part, end_part = _cli_helper.validate_parts(parts)

    # Check that the CLIP features exist
    _cli_helper.validate_part_format(clip_feats)
    for part in range(start_part, end_part):
        clip_part_path = clip_feats.format(part=part)
        if not os.path.isfile(clip_part_path):
            sys.exit(
                f'The CLIP file for part {part} does not exist: "{clip_part_path}"'
            )

    # Create directory for the computed SNIP parts
    _cli_helper.validate_part_format(snip_feats_out)
    try:
        first_snip_part_path = snip_feats_out.format(part=start_part)
        snip_out_parent_dir = os.path.dirname(first_snip_part_path)
        os.makedirs(snip_out_parent_dir, exist_ok=True)
    except Exception:
        sys.exit(
            f'Something is wrong with the output file paths specified for --snip_feats_out "{snip_feats_out}"'
        )

    # Load SNIP net
    net = torch.load(snip_model_path).eval().cuda()

    # Compute SNIP features for all parts
    print(
        f"Start computing SNIP features for parts {start_part} to {end_part} (excluded)"
    )
    for part in range(start_part, end_part):
        print(f"  Computing SNIP features for part {part} ...")
        # this is normally the bottleneck
        clip_part_path = clip_feats.format(part=part)
        clip_part = np.load(clip_part_path)
        # compute SNIP features
        snip_feats = compute_feats_for_chunk(net, clip_part)
        snip_part_path = snip_feats_out.format(part=part)
        np.save(snip_part_path, snip_feats)


if __name__ == "__main__":
    fire.Fire(snip_compress)
