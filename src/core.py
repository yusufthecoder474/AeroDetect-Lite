"""Shared image transforms and YOLOv8 decoding; no torch dependency."""
from pathlib import Path
import csv
import cv2
import numpy as np

FORMATS = {'.jpg', '.jpeg', '.png'}
# COCO -> VisDrone. Person is deliberately excluded (pedestrian/people split).
COMPATIBLE = {1: 3, 2: 4, 5: 9, 7: 6}
TARGETS = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

def load_image(path, max_bytes=50_000_000):
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(f'Image not found: {path}')
    if path.suffix.lower() not in FORMATS: raise ValueError('Only JPEG and PNG supported')
    if not 0 < path.stat().st_size <= max_bytes: raise ValueError('Empty or oversized image')
    data = path.read_bytes()
    if not (data.startswith(b'\xff\xd8\xff') or data.startswith(b'\x89PNG\r\n\x1a\n')):
        raise ValueError('Invalid image signature')
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None: raise ValueError(f'Cannot decode image: {path}')
    return image

def discover(directory):
    directory = Path(directory)
    if not directory.is_dir(): raise FileNotFoundError(f'Dataset directory not found: {directory}')
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in FORMATS)
    if not paths: raise ValueError('No supported images found')
    return paths

def annotations(path):
    """Retain all eight VisDrone fields, including ignored regions and score=0."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        fields = line.rstrip(',').split(',')
        if len(fields) != 8: raise ValueError('Expected eight VisDrone fields')
        row = [int(v.strip()) for v in fields]
        x,y,w,h,score,cls,trunc,occ = row
        if min(x,y,w,h) < 0 or w == 0 or h == 0 or cls not in range(12) or score not in (0,1) or trunc not in range(3) or occ not in range(3):
            raise ValueError('Invalid VisDrone annotation')
        rows.append(row)
    return np.array(rows, dtype=np.int64).reshape(-1,8)

def preprocess(image, size=640):
    if size <= 0 or size % 32: raise ValueError('Input size must be a positive multiple of 32')
    h,w = image.shape[:2]
    ratio = min(size/h, size/w)
    nw,nh = round(w*ratio),round(h*ratio)
    left,top = (size-nw)//2,(size-nh)//2
    canvas = np.full((size,size,3),114,np.uint8)
    canvas[top:top+nh,left:left+nw] = cv2.resize(image,(nw,nh))
    tensor = np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
    return tensor,(ratio,left,top,h,w)

def validate_input(tensor, size):
    if tensor.shape != (1,3,size,size) or tensor.dtype != np.float32 or not np.isfinite(tensor).all() or tensor.min()<0 or tensor.max()>1:
        raise ValueError('Expected finite float32 RGB NCHW [1,3,size,size] in [0,1]')

def restore(boxes, transform):
    r,left,top,h,w = transform
    boxes = np.asarray(boxes,dtype=np.float32).copy()
    boxes[:,[0,2]] = np.clip((boxes[:,[0,2]]-left)/r,0,w)
    boxes[:,[1,3]] = np.clip((boxes[:,[1,3]]-top)/r,0,h)
    return boxes

def iou(box, boxes):
    intersection = np.maximum(0,np.minimum(box[2:],boxes[:,2:])-np.maximum(box[:2],boxes[:,:2])).prod(1)
    area = np.maximum(0,box[2:]-box[:2]).prod()
    areas = np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(1)
    return intersection/np.maximum(area+areas-intersection,1e-9)

def nms(detections, threshold=.45, limit=300):
    order = np.argsort(-detections[:,4],kind='stable'); keep=[]
    while len(order) and len(keep)<limit:
        idx = order[0]; keep.append(idx); rest = order[1:]
        suppress = (detections[rest,5]==detections[idx,5]) & (iou(detections[idx,:4],detections[rest,:4])>threshold)
        order=rest[~suppress]
    return detections[keep]

def decode(raw, transform, confidence=.25, nms_iou=.45):
    raw=np.asarray(raw)
    if raw.ndim!=3 or raw.shape[0]!=1 or raw.shape[1]!=84 or raw.shape[2]==0 or not np.isfinite(raw).all():
        raise ValueError('Expected YOLOv8 COCO raw output [1,84,N]')
    if not 0<=confidence<=1 or not 0<=nms_iou<=1: raise ValueError('Threshold must be in [0,1]')
    pred=raw[0].T; classes=pred[:,4:].argmax(1); scores=pred[:,4:].max(1)
    if np.any(pred[:,2:4]<0) or np.any(pred[:,4:]<0) or np.any(pred[:,4:]>1): raise ValueError('Invalid prediction values')
    selected=(scores>=confidence)&np.isin(classes,list(TARGETS))
    pred,scores,classes=pred[selected],scores[selected],classes[selected]
    boxes=np.concatenate((pred[:,:2]-pred[:,2:4]/2,pred[:,:2]+pred[:,2:4]/2),axis=1)
    result=nms(np.column_stack((boxes,scores,classes)),nms_iou)
    result[:,:4]=restore(result[:,:4],transform)
    return result[(result[:,2]>result[:,0])&(result[:,3]>result[:,1])]

def draw(image, detections):
    image=image.copy()
    for x1,y1,x2,y2,score,cls in detections:
        cv2.rectangle(image,(round(x1),round(y1)),(round(x2),round(y2)),(30,220,80),2)
        cv2.putText(image,f'{TARGETS[int(cls)]} {score:.2f}',(round(x1),max(15,round(y1)-4)),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,220,80),1)
    return image

def metrics(milliseconds):
    values=np.asarray(milliseconds,float)
    if not values.size or not np.isfinite(values).all() or np.any(values<=0): raise ValueError('Timings must be finite and positive')
    return dict(mean_ms=float(values.mean()),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),images_per_second=float(1000/values.mean()))

def write_csv(path, rows):
    if not rows: raise ValueError('No rows to write')
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
