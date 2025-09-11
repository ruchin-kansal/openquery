from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Ultralytics is required. Please install dependencies: pip install -r requirements.txt"
    ) from exc


# COCO animal classes commonly available in stock YOLO models
DEFAULT_ANIMAL_CLASS_NAMES: Set[str] = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}


@dataclass
class AnalysisResult:
    species_counts: Dict[str, int]
    total_count: int
    output_dir: Path
    counts_json_path: Path
    counts_csv_path: Path
    annotated_video_path: Optional[Path]


class AnimalVideoAnalyzer:
    def __init__(
        self,
        model_path_or_name: str = "yolov8n.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.50,
        restrict_to_animals: bool = True,
    ) -> None:
        self.model = YOLO(model_path_or_name)
        self.device = device
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.restrict_to_animals = restrict_to_animals

    def _prepare_output_dir(self, video_path: Path, output_dir: Optional[Path]) -> Path:
        if output_dir is not None:
            out_dir = Path(output_dir)
        else:
            out_dir = Path.cwd() / "outputs" / "animal_counter" / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _init_video_writer(
        self, video_path: Path, output_dir: Path, annotated_basename: Optional[str]
    ) -> Tuple[Optional[cv2.VideoWriter], Optional[Path], float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for metadata: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if annotated_basename is None:
            return None, None, fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        annotated_path = output_dir / annotated_basename
        writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {annotated_path}")
        return writer, annotated_path, fps

    def analyze(
        self,
        video_path: str | os.PathLike[str],
        output_dir: Optional[str | os.PathLike[str]] = None,
        save_annotated: bool = True,
        display: bool = False,
    ) -> AnalysisResult:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        out_dir = self._prepare_output_dir(video_path, Path(output_dir) if output_dir else None)

        annotated_basename = f"{video_path.stem}_annotated.mp4" if save_annotated else None
        writer, annotated_path, _ = self._init_video_writer(video_path, out_dir, annotated_basename)

        # Track and stream results frame by frame
        results_generator = self.model.track(
            source=str(video_path),
            stream=True,
            tracker="bytetrack.yaml",
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        # Sets of unique track IDs per species
        species_to_track_ids: Dict[str, Set[int]] = {}

        # Determine allowed classes if restricting to animals
        model_class_names: Dict[int, str] = self.model.names if hasattr(self.model, "names") else {}
        allowed_species: Optional[Set[str]] = None
        if self.restrict_to_animals and model_class_names:
            allowed_species = {n for n in DEFAULT_ANIMAL_CLASS_NAMES if n in set(model_class_names.values())}

        try:
            for result in results_generator:
                if result.boxes is None or result.boxes.data is None:
                    frame_img = result.orig_img if hasattr(result, "orig_img") else None
                    if save_annotated and frame_img is not None and writer is not None:
                        writer.write(frame_img)
                    if display and frame_img is not None:
                        cv2.imshow("Animal Species Counter", frame_img)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                    continue

                boxes = result.boxes
                ids = boxes.id
                classes = boxes.cls

                # Annotated frame from Ultralytics util
                plotted_frame = result.plot() if save_annotated or display else None

                if ids is not None and classes is not None:
                    id_array = ids.cpu().numpy().astype(int).ravel()
                    cls_array = classes.cpu().numpy().astype(int).ravel()
                    for track_id, cls_idx in zip(id_array, cls_array):
                        class_name = model_class_names.get(int(cls_idx), str(cls_idx))
                        if allowed_species is not None and class_name not in allowed_species:
                            continue
                        if class_name not in species_to_track_ids:
                            species_to_track_ids[class_name] = set()
                        species_to_track_ids[class_name].add(int(track_id))

                if save_annotated and writer is not None:
                    frame_to_write = plotted_frame if plotted_frame is not None else result.orig_img
                    if frame_to_write is not None:
                        writer.write(frame_to_write)

                if display:
                    frame_to_show = plotted_frame if plotted_frame is not None else result.orig_img
                    if frame_to_show is not None:
                        cv2.imshow("Animal Species Counter", frame_to_show)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
        finally:
            if writer is not None:
                writer.release()
            if display:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

        species_counts: Dict[str, int] = {
            species: len(track_ids) for species, track_ids in sorted(species_to_track_ids.items())
        }
        total_count = int(sum(species_counts.values()))

        # Persist results
        counts_json_path = out_dir / "counts.json"
        counts_csv_path = out_dir / "counts.csv"

        with counts_json_path.open("w", encoding="utf-8") as f:
            json.dump({"species_counts": species_counts, "total_count": total_count}, f, indent=2)

        with counts_csv_path.open("w", encoding="utf-8") as f:
            f.write("species,count\n")
            for species, count in species_counts.items():
                f.write(f"{species},{count}\n")

        return AnalysisResult(
            species_counts=species_counts,
            total_count=total_count,
            output_dir=out_dir,
            counts_json_path=counts_json_path,
            counts_csv_path=counts_csv_path,
            annotated_video_path=Path(annotated_path) if annotated_path is not None else None,
        )

