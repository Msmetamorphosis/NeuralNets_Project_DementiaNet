from pathlib import Path

# base dataset location
BASE_DIR = Path("/content/drive/My Drive/NeuralNets_Project_DementiaNet/data/")

# audio folders
DEMENTIA_DIR = BASE_DIR / "audio" / "dementia"
NODEMENTIA_DIR = BASE_DIR / "audio" / "nodementia"

# metadata
DEMENTIA_META = BASE_DIR / "metadata" / "dementia.csv"
NODEMENTIA_META = BASE_DIR / "metadata" / "nodementia.csv"

# cleaned dataset output
OUTPUT_DIR = BASE_DIR / "clean_dataset"

# dataset manifest files
TRAIN_MANIFEST = OUTPUT_DIR / "manifests" / "train_dm.csv"
VALID_MANIFEST = OUTPUT_DIR / "manifests" / "valid_dm.csv"
TEST_MANIFEST = OUTPUT_DIR / "manifests" / "test_dm.csv"

# experiment settings
SEED = 42
TEST_SPLIT_PERCENT = 0.15
