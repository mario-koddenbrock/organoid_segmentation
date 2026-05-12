"""
Rename image files in Organoids_for_autosegmentation so each file name
starts with the date stamp of its parent session folder.

Example:
  20231108_Organoids_P021N/cropped_isotropic_images/
    P021N_40xSil_..._A_001_cropped_isotropic.tif
  →
    20231108_P021N_40xSil_..._A_001_cropped_isotropic.tif

Usage (dry-run, prints what would change):
    python rename_autoseg_files.py /scratch/koddenbrock/Organoids_for_autosegmentation

Apply renames:
    python rename_autoseg_files.py /scratch/koddenbrock/Organoids_for_autosegmentation --apply
"""

import argparse
import os
import sys


def collect_renames(root: str) -> list[tuple[str, str]]:
    """Return list of (old_path, new_path) for every file that needs renaming."""
    renames = []

    for session_dir in sorted(os.listdir(root)):
        session_path = os.path.join(root, session_dir)
        if not os.path.isdir(session_path):
            continue

        # Extract date stamp from first 8 characters of folder name (YYYYMMDD)
        if len(session_dir) < 8 or not session_dir[:8].isdigit():
            continue
        date_prefix = session_dir[:8]

        images_dir = os.path.join(session_path, "cropped_isotropic_images")
        if not os.path.isdir(images_dir):
            continue

        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith((".tif", ".tiff")):
                continue
            if fname.startswith(date_prefix):
                continue  # already renamed

            old_path = os.path.join(images_dir, fname)
            new_path = os.path.join(images_dir, f"{date_prefix}_{fname}")
            renames.append((old_path, new_path))

    return renames


def main():
    parser = argparse.ArgumentParser(description="Prepend date stamp to autoseg image filenames")
    parser.add_argument("root", help="Path to Organoids_for_autosegmentation directory")
    parser.add_argument("--apply", action="store_true", help="Actually rename (default: dry-run)")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"Directory not found: {args.root}", file=sys.stderr)
        sys.exit(1)

    renames = collect_renames(args.root)

    if not renames:
        print("Nothing to rename — all files already have date prefixes.")
        return

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] {len(renames)} files to rename:\n")

    for old, new in renames:
        rel_old = os.path.relpath(old, args.root)
        rel_new = os.path.relpath(new, args.root)
        print(f"  {rel_old}")
        print(f"  → {rel_new}\n")

        if args.apply:
            os.rename(old, new)

    if args.apply:
        print(f"Done. {len(renames)} files renamed.")
    else:
        print(f"Dry-run complete. Run with --apply to execute.")


if __name__ == "__main__":
    main()
