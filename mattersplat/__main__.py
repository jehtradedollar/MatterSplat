"""mattersplat <command> ...

    run      E57 -> trained splat (+ compact .splat, sweep list), optionally open the viewer
    view     serve the web viewer for a splat
    inspect  summarise an E57, dump its images, preview cloud and poses (step 1)
    colmap   export an E57 as a COLMAP text model and verify it (step 2)
    train    Spirula import + training only (step 3)
    tune     one-flag A/B training runs on held-out sweeps
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {"run": "run", "view": "view", "inspect": "inspect", "colmap": "to_colmap", "train": "train",
            "tune": "tune"}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    module = importlib.import_module(f"mattersplat.{COMMANDS[argv[0]]}")
    return module.main(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
