import os
import json
from typing import Optional, List, Dict, Any


def save_checkpoint(results: List[str], checkpoint_file: str) -> None:
    """Save current progress to a checkpoint file

    Args:
        results: List of answer strings
        checkpoint_file: Path to the checkpoint file
    """
    checkpoint_data = {"answers": results, "last_processed_idx": len(results) - 1}
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f)


def load_checkpoint(checkpoint_file: str) -> Optional[Dict[str, Any]]:
    """Load progress from checkpoint file if it exists

    Args:
        checkpoint_file: Path to the checkpoint file

    Returns:
        Optional dictionary containing checkpoint data or None if file doesn't exist
    """
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return None
