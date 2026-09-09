# data/__init__.py
from .download import download_nsl_kdd, download_unsw_nb15
from .preprocess import preprocess_nsl_kdd, preprocess_unsw_nb15, load_processed
from .partition import get_partitions
