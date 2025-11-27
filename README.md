# Test Project for Scope Library

This project tests that `daydream-scope` can be imported and used as a library, using only the **core** parts (not server).

## Setup

1. **Download models** (required before running the pipeline):
   ```bash
   uv run python download_models.py --pipeline longlive
   ```

2. **Run the pipeline**:
   ```bash
   uv run python test_pipeline.py
   ```
