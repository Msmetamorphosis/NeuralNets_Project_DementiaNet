from pathlib import Path
import sys

# detect if running in colab
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')

# dynamically find project root
PROJECT_ROOT = Path.cwd()

while PROJECT_ROOT.name != "NeuralNets_Project_DementiaNet":
    PROJECT_ROOT = PROJECT_ROOT.parent

# base paths
BASE_DIR = PROJECT_ROOT / "data"

# metadata
DEMENTIA_META = BASE_DIR / "metadata" / "dementia.csv"
NODEMENTIA_META = BASE_DIR / "metadata" / "nodementia.csv"

# outputs
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