from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze video to count animal species")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to ./outputs/animal_counter/<video_stem>",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path or name")
    parser.add_argument("--device", default=None, help="Device: cpu or cuda index like 0")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold")
    parser.add_argument(
        "--no-annotated",
        action="store_true",
        help="Do not write annotated video output",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display live annotated frames (ESC to stop)",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Do not restrict to animal classes; count all detected classes",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Lazy import to allow --help without heavy deps installed
    from .video_analyzer import AnimalVideoAnalyzer  # noqa: WPS433 (local import)

    analyzer = AnimalVideoAnalyzer(
        model_path_or_name=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        restrict_to_animals=not args.all_classes,
    )

    result = analyzer.analyze(
        video_path=Path(args.video),
        output_dir=Path(args.output) if args.output else None,
        save_annotated=not args.no_annotated,
        display=args.display,
    )

    print("Species counts:")
    for species, count in result.species_counts.items():
        print(f"- {species}: {count}")
    print(f"Total: {result.total_count}")
    print(f"JSON: {result.counts_json_path}")
    print(f"CSV: {result.counts_csv_path}")
    if result.annotated_video_path:
        print(f"Annotated video: {result.annotated_video_path}")


if __name__ == "__main__":
    main()

