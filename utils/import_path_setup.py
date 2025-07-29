import sys
from pathlib import Path

def setup_paths(anchor_filename="app.py"):
    """
    Traverse up from current file until the anchor file is found (e.g., app.py),
    then add its parent directory (project root) and its relevant submodules to sys.path
    """
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent

    # Find project root by looking for anchor file
    for parent in current_file.parents:
        if (parent / anchor_filename).exists():
            project_root = parent
            break
    else:
        raise RuntimeError(f"Could not find project root containing '{anchor_filename}'")

    # Append project root and its shared module subdirs
    candidate_paths = [
        project_root,
        project_root / "model",
        project_root / "utils",
        project_root / "database",
        project_root / "tests",
        project_root / "model" / "lightgbm_202504"
    ]

    for path in candidate_paths:
        if str(path) not in sys.path:
            sys.path.append(str(path))

    # Optionally, remove the current file's directory to avoid import shadowing
    try:
        sys.path.remove(str(current_dir))
    except ValueError:
        pass
