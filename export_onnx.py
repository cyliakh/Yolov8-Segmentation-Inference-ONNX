from ultralytics import YOLO

model = YOLO("yolov8seg.pt")

model.export(format="onnx")
