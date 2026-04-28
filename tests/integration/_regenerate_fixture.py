"""Regenerate the regression fixture at tests/fixtures/apple_in_the_tray_baseline/.

Not invoked by the test suite itself. Run manually when a deliberate,
reviewed behaviour change has been made to the Gaussian tracker:

    python tests/integration/_regenerate_fixture.py

The script:
  1. Runs the Gaussian tracker on apple_in_the_tray frames 280..339.
  2. Strips each per-frame JSON dump to the "canonical subset"
     (the observable tracker behaviour; see test_gaussian_pipeline_regression.py).
  3. Writes the trimmed JSONs to tests/fixtures/apple_in_the_tray_baseline/.

After running, review `git diff` carefully to confirm the changes match
the intended behaviour shift before committing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
import sys

# Reuse the extractor and comparator constants from the test module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_gaussian_pipeline_regression import (      # noqa: E402
    FIXTURE_DIR, TRACKER_CMD, REPO_ROOT, _canonical,
)


def main() -> int:
    scratch = REPO_ROOT / "tests" / "visualization_pipeline" \
                        / "apple_in_the_tray" / "_regression_gen"
    if scratch.exists():
        shutil.rmtree(scratch)

    cmd = TRACKER_CMD + [
        "--out-subdir", "_regression_gen",
        "--state-subdir", "_regression_gen",
    ]
    print(f"[regen] running: {' '.join(cmd)}")
    import os as _os
    env = {
        **_os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    res = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if res.returncode != 0:
        print(f"[regen] tracker failed rc={res.returncode}", file=sys.stderr)
        return res.returncode

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    # Clear any stale fixture files first.
    for old in FIXTURE_DIR.glob("frame_*.json"):
        old.unlink()

    frames = sorted(scratch.glob("frame_*.json"))
    if not frames:
        print(f"[regen] no frames produced under {scratch}", file=sys.stderr)
        return 1

    for src in frames:
        dump = json.loads(src.read_text())
        canon = _canonical(dump)
        dst = FIXTURE_DIR / src.name
        dst.write_text(json.dumps(canon, indent=2, default=str))

    print(f"[regen] wrote {len(frames)} fixtures to {FIXTURE_DIR}")
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
