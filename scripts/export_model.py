"""Official Ultralytics export; record graph, provenance and checkpoint hash."""
import argparse, hashlib, json
from pathlib import Path
import onnx
from ultralytics import YOLO
p=argparse.ArgumentParser(); p.add_argument('--weights',default='yolov8n.pt'); p.add_argument('--size',type=int,default=640); a=p.parse_args()
model=YOLO(a.weights)
output=model.export(format='onnx',imgsz=a.size,batch=1,dynamic=False,simplify=False,opset=17,half=False,nms=False,device='cpu')
graph=onnx.load(output); onnx.checker.check_model(graph,full_check=True)
record={'checkpoint':a.weights,'sha256':hashlib.sha256(Path(a.weights).read_bytes()).hexdigest(),'onnx_sha256':hashlib.sha256(Path(output).read_bytes()).hexdigest(),'license':'AGPL-3.0 / Ultralytics enterprise terms','classes':model.names,'input':str(graph.graph.input),'output':str(graph.graph.output),'operators':sorted({n.op_type for n in graph.graph.node}),'opset':17,'validation':'PASSED'}
Path('outputs').mkdir(exist_ok=True); Path('outputs/export.json').write_text(json.dumps(record,indent=2))
print(output)
