from pathlib import Path

PROJECT_ROOT = Path("/content/NeuralNets_Project_DementiaNet")
BASE_DIR = Path("/content/drive/MyDrive/NeuralNets_Project_DementiaNet/data")

AUDIO_DIR = BASE_DIR / "audio"
DEMENTIA_DIR = AUDIO_DIR / "dementia"
NODEMENTIA_DIR = AUDIO_DIR / "nodementia"

DEMENTIA_META = BASE_DIR / "metadata" / "dementia.csv"
NODEMENTIA_META = BASE_DIR / "metadata" / "nodementia.csv"

OUTPUT_DIR = BASE_DIR / "clean_dataset"

TRAIN_MANIFEST = OUTPUT_DIR / "train_dm.csv"
VALID_MANIFEST = OUTPUT_DIR / "valid_dm.csv"
TEST_MANIFEST = OUTPUT_DIR / "test_dm.csv"

RANDOM_SEED = 42
SEED = RANDOM_SEED

TEST_SPLIT_PERCENT = 0.15
VALID_SPLIT_PERCENT = 0.15

# Fixed-length clips for manifests (see 01_finalize_validated_dataset.ipynb)
CLIP_SECONDS = 30

TRAIN_STRIDE = 20
EVAL_STRIDE = 30