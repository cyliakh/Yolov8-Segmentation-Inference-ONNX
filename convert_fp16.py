import onnx
from onnxconverter_common import float16

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "yolov8seg.onnx"
FP16_MODEL_PATH = "yolov8seg-fp16.onnx"
FP16_IO_MODEL_PATH = "yolov8seg-fp16-io.onnx"

# ===============================
# LOAD ORIGINAL MODEL
# ===============================
print(f"🔹 Loading model: {MODEL_PATH}")
model = onnx.load(MODEL_PATH)

# ===============================
# CONVERT TO FP16
# ===============================
print("⚙️  Converting to FP16...")
model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
model_fp16_io = float16.convert_float_to_float16(model, keep_io_types=False)

# keep_io_types=True ensures inputs/outputs stay float32 (for compatibility)

# ===============================
# SAVE FP16 MODEL
# ===============================
onnx.save(model_fp16, FP16_MODEL_PATH)
onnx.save(model_fp16_io, FP16_IO_MODEL_PATH)
print(f"✅ Saved FP16 model: {FP16_MODEL_PATH}")
print(f"✅ Saved FP16 model with IO in FP16: {FP16_IO_MODEL_PATH}")
