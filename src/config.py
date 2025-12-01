from pathlib import Path

# Get the directory of this file (src/), then go up one level to project root
# .resolve() ensures we get the absolute path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Define your static paths here
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
SRC_DIR = PROJECT_ROOT / "src"
CERT_PATH = PROJECT_ROOT / "cert.pem"
KEY_PATH = PROJECT_ROOT / "key.pem"