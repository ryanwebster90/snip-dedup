import sys
import string


def validate_shards(shards):
    if type(shards) is int:
        sys.exit(
            "Single value is not accepted for --shards as it's ambiguous between wanting only that exact shard or that number of shards starting from 0. Please use a range instead like 0:2"
        )
    shards_bounds = shards.split(":")
    try:
        shards_bounds = [int(shard) for shard in shards_bounds]
    except Exception:
        sys.exit(
            f'The shards pattern "{shards}" is not valid. It should be a valid range such as "0:2" or "14:42"'
        )
    if len(shards_bounds) == 0:
        sys.exit("The --shards argument cannot be empty")
    elif len(shards_bounds) == 1:
        sys.exit(
            "Single value is not accepted for --shards as it's ambiguous between wanting only that exact shard or that number of shards starting from 0. Please use a range instead like 0:2"
        )
    elif len(shards_bounds) > 2:
        sys.exit(
            "Ranges with more than 2 parts, such as 0:2:14 are not valid for --shards. Please limit yourself with simple ranges such as 0:14"
        )
    start_shard, end_shard = shards_bounds
    if start_shard < 0 or end_shard < 0:
        sys.exit("Only positive integers are allowed for --shards, such as 0:14")
    if end_shard <= start_shard:
        sys.exit(
            'The --shards argument must be of the shape "s:e" with s < e, such as "0:1" or "14:42". The "e" bound is excluded.'
        )
    return start_shard, end_shard


def validate_shard_format(pattern):
    format_variables = [
        tup[1] for tup in string.Formatter().parse(pattern) if tup[1] is not None
    ]
    if len(format_variables) != 1 or format_variables[0] != "shard":
        sys.exit(
            f'Your pattern "{pattern}" is not valid as it should contain the "shard" variable such as "{{shard:04d}}.npy".'
        )
