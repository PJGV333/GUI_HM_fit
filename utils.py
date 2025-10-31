import math


def parse_range(range_str):
    """Parse a concentration range expressed as "start,end,count"."""
    if range_str is None:
        raise ValueError("Range string cannot be None")

    parts = [part.strip() for part in str(range_str).split(',')]
    if len(parts) != 3 or any(part == '' for part in parts):
        raise ValueError("Range must contain exactly three comma-separated values")

    try:
        start = float(parts[0])
        end = float(parts[1])
        samples = float(parts[2])
    except ValueError as exc:
        raise ValueError("Range values must be numeric") from exc

    if not all(map(math.isfinite, (start, end, samples))):
        raise ValueError("Range values must be finite numbers")

    if samples <= 0:
        raise ValueError("Number of samples must be positive")

    return start, end, samples
