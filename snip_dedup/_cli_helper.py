import sys
import string


def validate_parts(parts):
    if type(parts) is int:
        sys.exit(
            "Single value is not accepted for --parts as it's ambiguous between wanting only that exact part or that number of parts starting from 0. Please use a range instead like 0:2"
        )
    parts_bounds = parts.split(":")
    try:
        parts_bounds = [int(part) for part in parts_bounds]
    except Exception:
        sys.exit(
            f'The parts pattern "{parts}" is not valid. It should be a valid range such as "0:2" or "14:42"'
        )
    if len(parts_bounds) == 0:
        sys.exit("The --parts argument cannot be empty")
    elif len(parts_bounds) == 1:
        sys.exit(
            "Single value is not accepted for --parts as it's ambiguous between wanting only that exact part or that number of parts starting from 0. Please use a range instead like 0:2"
        )
    elif len(parts_bounds) > 2:
        sys.exit(
            "Ranges with more than 2 parts, such as 0:2:14 are not valid for --parts. Please limit yourself with simple ranges such as 0:14"
        )
    start_part, end_part = parts_bounds
    if start_part < 0 or end_part < 0:
        sys.exit("Only positive integers are allowed for --parts, such as 0:14")
    if end_part <= start_part:
        sys.exit(
            'The --parts argument must be of the shape "s:e" with s < e, such as "0:1" or "14:42". The "e" bound is excluded.'
        )
    return start_part, end_part


def validate_part_format(pattern):
    format_variables = [
        tup[1] for tup in string.Formatter().parse(pattern) if tup[1] is not None
    ]
    if len(format_variables) != 1 or format_variables[0] != "part":
        sys.exit(
            f'Your pattern "{pattern}" is not valid as it should contain the "part" variable such as "{{part:04d}}.npy".'
        )
