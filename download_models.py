#!/usr/bin/env python3
import os
import sys

# Re-use the original tool main function to parse args the same way
from scope.server.download_models import main as download_models

from models_config import MODELS_DIR_ENV_VAR, get_models_dir

os.environ[MODELS_DIR_ENV_VAR] = str(get_models_dir())
sys.exit(download_models())

