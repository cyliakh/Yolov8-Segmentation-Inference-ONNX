import cv2
import numpy as np
import onnxruntime as ort


#this code only draws the bounding boxes from yolov8 segmentation model, the masks will be dealt with later


# =========================
# CONFIG
# =========================
MODEL_PATH = "yolov8seg.onnx"#MODEL_PATH = "/home/cylia/Fire/Yolov8_seg/firev8_x_seg_1280.onnx"

IMAGE_PATH = "image.jpg"
INPUT_SIZE = 1280
CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.45

# =========================
# UTILS
# =========================
def letterbox(im, new_shape=(1280, 1280), color=(114, 114, 114)):
    h0, w0 = im.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (dw, dh)

def preprocess(image_path):
    im = cv2.imread(image_path)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_resized, r, (dw, dh) = letterbox(im_rgb, (INPUT_SIZE, INPUT_SIZE))
    im_resized = im_resized.astype(np.float32) / 255.0
    im_resized = np.transpose(im_resized, (2, 0, 1))  # HWC to CHW
    im_resized = np.expand_dims(im_resized, 0)  # add batch
    return im_resized, r, dw, dh

def xywh2xyxy(x):
    # x = [N,4] xywh -> xyxy
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2]/2
    y[:, 1] = x[:, 1] - x[:, 3]/2
    y[:, 2] = x[:, 0] + x[:, 2]/2
    y[:, 3] = x[:, 1] + x[:, 3]/2
    return y

def nms(boxes, scores, iou_threshold):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return keep

def compute_iou(box, boxes):
    # box = [4], boxes = [N,4]
    x1 = np.maximum(box[0], boxes[:,0])
    y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2])
    y2 = np.minimum(box[3], boxes[:,3])
    inter_area = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    box_area = (box[2]-box[0])*(box[3]-box[1])
    boxes_area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    union = box_area + boxes_area - inter_area
    return inter_area / union

# =========================
# INFERENCE
# =========================
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

img, r, dw, dh = preprocess(IMAGE_PATH)


import time

# warmup (optional)
_ = session.run(None, {input_name: img})

# timed inference
t1 = time.time()
preds = session.run(None, {input_name: img})[0]  # [1,40,33600]
t2 = time.time()
# ensure float32 type
if preds.dtype != np.float32:
    preds = preds.astype(np.float32)
    #this is used in case of model both weights and IO in fp16

# compute inference time and FPS
inf_time = t2 - t1
fps = 1 / inf_time if inf_time > 0 else 0

print(f"Inference time: {inf_time*1000:.2f} ms, FPS: {fps:.2f}")

# =========================
# POSTPROCESS
# =========================
preds = preds[0]  # remove batch dim -> [40,33600]
boxes_xywh = preds[:4].T  # [N,4] (x,y,w,h)
scores = preds[4].T       # [N]  #this is class 0, increment to get other classes, add a for loop if you want to draw all of them
class_ids = np.argmax(preds[4:8], axis=0)  # ignore masks
boxes_xyxy = xywh2xyxy(boxes_xywh)

# NMS
mask = scores > CONF_THRESHOLD
boxes_xyxy = boxes_xyxy[mask]
scores = scores[mask]
class_ids = class_ids[mask]

keep = nms(boxes_xyxy, scores, IOU_THRESHOLD)
boxes_xyxy = boxes_xyxy[keep]
scores = scores[keep]
class_ids = class_ids[keep]

# rescale boxes to original image
boxes_xyxy[:, [0,2]] -= dw
boxes_xyxy[:, [1,3]] -= dh
boxes_xyxy /= r
boxes_xyxy = boxes_xyxy.round().astype(int)

# =========================
# PRINT RESULTS
# =========================
for box, score, cls in zip(boxes_xyxy, scores, class_ids):
    print(f"Class {cls}, Score {score:.2f}, Box {box}")


# =========================
# DRAW AND SAVE IMAGE
# =========================
img_orig = cv2.imread(IMAGE_PATH)

for box, score, cls in zip(boxes_xyxy, scores, class_ids):
    x1, y1, x2, y2 = box
    cv2.rectangle(img_orig, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img_orig, f"{cls} {score:.2f}", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

OUTPUT_PATH = "result.jpg"
cv2.imwrite(OUTPUT_PATH, img_orig)
print(f"Saved result image to {OUTPUT_PATH}")
