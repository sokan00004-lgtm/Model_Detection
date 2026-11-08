"""Export YOLO26n-seg to ONNX for the Flutter dashboard.

    pip install -U ultralytics onnx onnxslim
    python export_model.py

The exported model is end-to-end (NMS-free), so it outputs:
    output0: [1, 300, 38]       x1, y1, x2, y2, score, class, 32 mask coeffs
    output1: [1, 32, 160, 160]  mask prototypes
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")  # downloads weights on first run
onnx_path = model.export(format="onnx", imgsz=640, opset=17, simplify=True)

dest = Path("web/models")
dest.mkdir(parents=True, exist_ok=True)
shutil.copy(onnx_path, dest / "yolo26n-seg.onnx")
print(f"Copied {onnx_path} -> {dest / 'yolo26n-seg.onnx'}")
