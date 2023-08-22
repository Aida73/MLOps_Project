import datetime

"""Settings"""
from pathlib import Path

# Home directory
# HOME_DIR = Path.cwd().parent
HOME_DIR = Path.cwd()

# data
DATA_DIR = Path(HOME_DIR, "data")
DATA_DIR_INPUT = Path(DATA_DIR, "input")
DATA_DIR_OUTPUT = Path(DATA_DIR, "output")

# models
MODEL_DIR = Path(HOME_DIR, "models")
# add on prefix the execution date (YYYYMMDD_{MODEL_NAME})
execution_date = datetime.datetime.now().strftime('%Y%m%d')
MODEL_NAME = f"{execution_date}_model_stanford_breed_dogs.h5"

# reports: graphs, html, ...
REPORT_DIR = Path(HOME_DIR, "reports")

# Source de code
SRC_DIR = Path(HOME_DIR, "src")

# Notebook
NOTEBOOK_DIR = Path(HOME_DIR, "notebooks")

TIMEZONE = "UTC"


SEED = 43
