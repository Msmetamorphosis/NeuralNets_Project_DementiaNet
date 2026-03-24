from pathlib import Path
import sys

# figure out project root depending on where this is running
if "google.colab" in sys.modules:
    PROJECT_ROOT = Path("/content/drive/MyDrive/NeuralNets_Project_DementiaNet")
else:
    PROJECT_ROOT = Path(__file__).resolve().parent

# base data location
BASE_DIR = PROJECT_ROOT / "data"

# audio folders
DEMENTIA_DIR = BASE_DIR / "audio" / "dementia"
NODEMENTIA_DIR = BASE_DIR / "audio" / "nodementia"

# metadata
DEMENTIA_META = BASE_DIR / "metadata" / "dementia.csv"
NODEMENTIA_META = BASE_DIR / "metadata" / "nodementia.csv"

# output
OUTPUT_DIR = BASE_DIR / "clean_dataset"

# manifests
TRAIN_MANIFEST = OUTPUT_DIR / "manifests" / "train_dm.csv"
VALID_MANIFEST = OUTPUT_DIR / "manifests" / "valid_dm.csv"
TEST_MANIFEST = OUTPUT_DIR / "manifests" / "test_dm.csv"

# reproducibility
RANDOM_SEED = 42
SEED = RANDOM_SEED

# splits (used in speaker-based sampling, not naive splitting)
TEST_SPLIT_PERCENT = 0.15
VALID_SPLIT_PERCENT = 0.15

# clip settings
TRAIN_STRIDE = 20
EVAL_STRIDE = 30