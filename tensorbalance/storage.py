"""Write complete files into a new output directory without overwriting runs."""
from pathlib import Path
import shutil
import tempfile


def write_directory(destination, writer):
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Output path already exists: {destination}. Choose another --out directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".tensor-balance-", dir=destination.parent))
    reserved = False
    moved = []
    try:
        writer(stage)
        destination.mkdir()  # exclusive reservation, including concurrent jobs
        reserved = True
        for source in stage.iterdir():
            target = destination / source.name
            source.rename(target)
            moved.append(target)
    except BaseException:
        # Remove only files published by this call, never another writer's files.
        for target in reversed(moved):
            try:
                target.unlink()
            except OSError:
                pass
        if reserved:
            try:
                destination.rmdir()
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return destination
