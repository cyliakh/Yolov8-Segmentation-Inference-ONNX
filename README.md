# YOLOv8 Segmentation ONNX Tools

This repository provides tools to **export**, **convert**, and **run inference** on YOLOv8 segmentation models using **ONNX Runtime** — optimized for both **CPU** and **GPU (TensorRT)** execution.

---

## 📦 Overview

### `export_model.py`
- Exports a trained YOLOv8 model (`.pt`) to ONNX format.
- Use this to prepare your model for deployment on ONNX Runtime or TensorRT.

---

### `convert_fp16.py`
- Converts your exported ONNX model into two optimized versions:
  - **FP16 weights** – standard half-precision model.
  - **FP16 (IO)** – both model weights and inputs/outputs in FP16.
- The FP16-IO version can **double inference speed** on compatible GPUs by reducing memory bandwidth and transfer overhead.

---

### `test_image.py`
- Runs inference on a single image using ONNX Runtime.
- Generates and saves bounding boxes on the output image.
- Masks are not handled yet (to be added later).

#### 🧩 Multi-Class Handling
By default, inference extracts boxes for **one class** (e.g. `class 0`).
To visualize all classes, replace this snippet with the loop below and customize it:

```python
preds = preds[0]  # remove batch dim -> [40,33600]
boxes_xywh = preds[:4].T  # [N,4] (x,y,w,h)
boxes_xyxy = xywh2xyxy(boxes_xywh)

# loop over each class (4 classes example)
for cls_idx in range(4):
    scores = preds[4 + cls_idx].T       # each class score
    mask = scores > CONF_THRESHOLD
    cls_boxes = boxes_xyxy[mask]
    cls_scores = scores[mask]
    cls_ids = np.full(len(cls_boxes), cls_idx)

    keep = nms(cls_boxes, cls_scores, IOU_THRESHOLD)
    cls_boxes = cls_boxes[keep]
    cls_scores = cls_scores[keep]
    cls_ids = cls_ids[keep]

