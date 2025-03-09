"""
Default configuration settings for the data preparation module.
This module contains constants and default values used by data_prep.py.
"""

# File naming conventions for data preparation outputs
TRANSFORMED_DATA_FILENAME = "transformed_data.json"
VISION_LM_FILENAME = "vision_lm_training.json"
SHAREGPT_FILENAME = "sharegpt_training.json"

# JSON formatting
JSON_INDENT = 4

# Data transformation thresholds
ELEMENT_SCORE_THRESHOLD = (
    2  # Elements with sum of scores >= this value will be scored as 1, otherwise 0
)

# Command line argument defaults for data_prep.py
DEFAULT_ARGS = {
    "verbose": False,
    "require_images": False,
    "frames_dir": None,
    "output": None,
}

# Image path verification settings
IMAGE_PATH_VERIFICATION = {
    "max_invalid_paths_to_show": 5,  # Number of invalid paths to display in logs
}
