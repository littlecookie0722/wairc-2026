from pathlib import Path

NUM_CLASSES = 9
RANDOM_SEED = 2026
VAL_RATIO = 0.2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = PROJECT_ROOT / "data_and_code" / "ai_radio_2026_qualifying_release" / "train"
TEST_ROOT = PROJECT_ROOT / "data_and_code_patch-1" / "test_public_v1.1"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SUBMISSION_PATH = OUTPUT_DIR / "submissions" / "submission_spectrogram_kfold.txt"
CACHE_DIR = OUTPUT_DIR / "cache"
