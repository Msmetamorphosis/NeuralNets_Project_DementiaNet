from pathlib import Path

# base dataset location
BASE_DIR = Path("/content/drive/My Drive/NeuralNets_Project_DementiaNet/data/")

# audio folders
DEMENTIA_DIR = BASE_DIR / "dementia"
NODEMENTIA_DIR = BASE_DIR / "nodementia"

# metadata
DEMENTIA_META = BASE_DIR / "dementia.csv"
NODEMENTIA_META = BASE_DIR / "nodementia.csv"

# cleaned dataset output
OUTPUT_DIR = BASE_DIR / "clean_dataset"

# dataset manifest files
TRAIN_MANIFEST = OUTPUT_DIR / "train_dm.csv"
VALID_MANIFEST = OUTPUT_DIR / "valid_dm.csv"
TEST_MANIFEST = OUTPUT_DIR / "test_dm.csv"

# experiment settings
SEED = 42
TEST_SPLIT_PERCENT = 0.15
