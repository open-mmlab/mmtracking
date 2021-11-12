try:
    from vot.region import calculate_overlap
except ImportError:
    calculate_overlap = None


def calculate_region_overlap(*args, **kwargs):
    if calculate_overlap is None:
        raise ImportError(
            'Please run'
            'pip install vot-toolkit@git+https://github.com/votchallenge/vot-toolkit-python@0c61b3'  # noqa: E501
            'to manually install vot-toolkit')
    return calculate_overlap(*args, **kwargs)
