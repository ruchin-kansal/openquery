## Animal Species Counter (Video)

Analyze a video to detect animals, classify them into species, and output per-species counts. Uses a YOLO tracker to avoid double-counting across frames and can optionally save an annotated video.

### Features
- Detects animals in video using a YOLO model
- Tracks objects across frames to count unique individuals
- Outputs per-species counts as JSON and CSV
- Optionally saves an annotated video with boxes, labels, and track IDs

### Requirements
- Python 3.9+
- CPU or GPU. CPU works but will be slower

### Quickstart
1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Run the analyzer:
```bash
python -m animal_counter --video /path/to/video.mp4 --output /path/to/output_dir
```

Common flags:
- `--model yolov8n.pt` (default) or a custom Ultralytics YOLO model path
- `--conf 0.25` confidence threshold
- `--iou 0.5` IoU threshold
- `--device cpu` or CUDA device like `0`
- `--no-annotated` to skip saving annotated video

Outputs in the chosen directory:
- `counts.json`: species counts as JSON
- `counts.csv`: species counts as CSV
- `VIDEO_NAME_annotated.mp4`: annotated video (optional)

### Notes
- The default model (`yolov8n.pt`) is trained on COCO and includes several animal species (e.g., dog, cat, bird, horse, sheep, cow, elephant, bear, zebra, giraffe). To count additional species, use or fine-tune a model that includes those classes.
- If you have a custom model with many species, pass it via `--model`.

# openquery