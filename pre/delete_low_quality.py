"""Delete (or back up) low-quality radiology description files.

Reads the invalid-files list produced by filter_quality.py and removes
those files from the text directory.

Configuration (environment variables / .env)
--------------------------------------------
INVALID_FILES_REPORT   Path to the invalid-files list (one filename per line,
                       first line may be a header starting with "Invalid").
TEXT_FILTER_DIR        Directory containing the .txt files to clean.
                       Falls back to TEXT_MEDGEMMA_DIR if not set.
LOW_QUALITY_BACKUP_DIR (optional) Directory to back up deleted files.
                       If not set, files are deleted without backup.

Usage
-----
    # First run filter_quality.py to produce invalid_files.txt, then:
    python -m stroke_multimodal_prognosis.pre.delete_low_quality
    # or
    python stroke_multimodal_prognosis/pre/delete_low_quality.py
"""

import os
import shutil
import sys
from pathlib import Path

# ── Load .env if available ────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parents[2] / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass


def delete_invalid(
    invalid_list_path: str,
    target_folder: str,
    backup_folder: str | None = None,
) -> dict:
    """Delete files listed in *invalid_list_path* from *target_folder*.

    Returns a summary dict with keys: total, deleted, backed_up, not_found, errors.
    """
    try:
        lines = Path(invalid_list_path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        print(f"[delete_low_quality] ERROR: Report not found: {invalid_list_path}")
        return {}

    # Skip header line if present
    filenames = [
        ln.strip()
        for ln in lines
        if ln.strip() and not ln.startswith("Invalid")
    ]

    if not filenames:
        print("[delete_low_quality] No filenames found in the report.")
        return {}

    if backup_folder:
        os.makedirs(backup_folder, exist_ok=True)

    results = {"total": len(filenames), "deleted": 0, "backed_up": 0,
               "not_found": [], "errors": []}

    for fname in filenames:
        fpath = os.path.join(target_folder, fname)
        if not os.path.exists(fpath):
            results["not_found"].append(fname)
            continue
        try:
            if backup_folder:
                shutil.copy2(fpath, os.path.join(backup_folder, fname))
                results["backed_up"] += 1
            os.remove(fpath)
            results["deleted"] += 1
            print(f"  Deleted: {fname}")
        except Exception as exc:
            results["errors"].append(f"{fname}: {exc}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _target  = os.environ.get("TEXT_FILTER_DIR", os.environ.get("TEXT_MEDGEMMA_DIR", ""))
    _backup  = os.environ.get("LOW_QUALITY_BACKUP_DIR", None)

    # Default report location: <_target>/invalid_files.txt
    _default = os.path.join(_target, "invalid_files.txt") if _target else ""
    _report  = os.environ.get("INVALID_FILES_REPORT", _default)

    if not _target or not _report:
        print(
            "[delete_low_quality] ERROR: Set TEXT_FILTER_DIR and INVALID_FILES_REPORT. "
            "See .env.example for details."
        )
        sys.exit(1)

    print(f"[delete_low_quality] Report : {_report}")
    print(f"[delete_low_quality] Target : {_target}")
    print(f"[delete_low_quality] Backup : {_backup or '(none)'}")

    summary = delete_invalid(_report, _target, _backup)
    if summary:
        print(
            f"\n[delete_low_quality] Summary:\n"
            f"  Total in report : {summary['total']}\n"
            f"  Deleted         : {summary['deleted']}\n"
            f"  Backed up       : {summary['backed_up']}\n"
            f"  Not found       : {len(summary['not_found'])}\n"
            f"  Errors          : {len(summary['errors'])}"
        )
        if summary["not_found"]:
            print("  Not found:", summary["not_found"])
        if summary["errors"]:
            print("  Errors:", summary["errors"])
