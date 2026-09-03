"""E2E-G1-R1：语义正确槽位存在表示的修正版配对Gate。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from 统一模型代码 import e2e_g1 as g1  # noqa: E402


SOURCE_RUN = PROJECT_ROOT / "outputs_e2e" / "unified" / "e2e_g1" / "20260901_145307"
SNAPSHOT_ROOT = SOURCE_RUN / "reference_snapshot"
SNAPSHOT_MANIFEST = SNAPSHOT_ROOT / "snapshot_manifest.json"


def snapshot_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(SNAPSHOT_MANIFEST.read_text(encoding="utf-8"))
    g1.require(payload["status"] == "PASS", "G1冻结快照未通过")
    files = [row["snapshot"] for row in payload["files"]]
    artifacts = []
    for row in payload["artifacts"]:
        current = dict(row["snapshot"])
        current["name"] = row["expected"]["name"]
        artifacts.append(current)
    return files, artifacts


FILES, ARTIFACTS = snapshot_rows()


def verify_snapshot_artifacts(names: list[str]) -> list[dict[str, Any]]:
    indexed = {row["name"]: row for row in ARTIFACTS}
    rows = []
    for name in names:
        g1.require(name in indexed, f"冻结快照缺少checkpoint: {name}")
        expected = indexed[name]
        current = g1.file_identity(Path(expected["path"]))
        g1.require(
            current["size_bytes"] == expected["size_bytes"]
            and current["sha256"] == expected["sha256"],
            f"冻结checkpoint身份变化: {name}",
        )
        rows.append({"name": name, **current})
    return rows


g1.GATE_NAME = "E2E-G1-R1"
g1.OUTPUT_SUBDIR = "e2e_g1_r1"
g1.CONFIG_PATH = PACKAGE_ROOT / "configs" / "e2e_g1_r1.json"
g1.EXTRA_CODE_PATHS = [Path(__file__).resolve()]
g1.SOURCE_G1_MANIFEST = SOURCE_RUN / "manifest.json"
g1.USING_LOCAL_SNAPSHOT = True
g1.REFERENCE_OUTPUT_ROOT = SNAPSHOT_ROOT
g1.COARSE_TRAIN = Path(FILES[0]["path"])
g1.COARSE_VAL_SELECT = Path(FILES[1]["path"])
g1.COARSE_VAL_COMPARE = Path(FILES[2]["path"])
g1.RAW_VALIDATION = Path(FILES[3]["path"])
g1.RAW_TRAIN_PARTS = (
    (0, 4096, Path(FILES[4]["path"])),
    (4096, 8192, Path(FILES[5]["path"])),
    (8192, 16384, Path(FILES[6]["path"])),
)
g1.verify_artifacts = verify_snapshot_artifacts


if __name__ == "__main__":
    g1.main()
