"""Audit TIFF structure and physical calibration of a reviewed AKPS directory."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import tifffile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    files = sorted(args.root.glob("*.tif"))
    lines = collections.Counter()
    axis_shapes = collections.Counter()
    spacings = collections.Counter()
    missing_calibration = []

    for path in files:
        match = re.match(r"\d{8}_(NCO|AKPS|AKP|AK|A)_", path.name)
        lines[match.group(1) if match else "UNKNOWN"] += 1
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            axis_shapes[(series.axes, tuple(series.shape))] += 1
            metadata = tif.imagej_metadata or {}
            tags = tif.pages[0].tags
            try:
                z_um = float(metadata["spacing"])
                y_num, y_den = tags["YResolution"].value
                x_num, x_den = tags["XResolution"].value
                spacing = (
                    round(z_um, 9),
                    round(y_den / y_num, 9),
                    round(x_den / x_num, 9),
                    str(metadata.get("unit", "")),
                )
                spacings[spacing] += 1
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                missing_calibration.append({"file": path.name, "error": str(error)})

    report = {
        "root": str(args.root),
        "n_tiffs": len(files),
        "line_counts": dict(sorted(lines.items())),
        "spacing_counts": [
            {"z_um": key[0], "y_um": key[1], "x_um": key[2], "unit": key[3], "count": count}
            for key, count in sorted(spacings.items())
        ],
        "axis_shape_counts": [
            {"axes": key[0], "shape": key[1], "count": count}
            for key, count in axis_shapes.most_common()
        ],
        "missing_calibration": missing_calibration,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
