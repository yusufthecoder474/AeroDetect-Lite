from pathlib import Path
import numpy as np
from .core import validate_input

class TorchBackend:
    def __init__(self, path, size=640, threads=1):
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(threads)
        self.torch=torch; self.size=size
        self.model=YOLO(str(path)).model.cpu().float().eval()
        self.model.fuse(verbose=False)
    def run(self, tensor):
        validate_input(tensor,self.size)
        with self.torch.inference_mode():
            result=self.model(self.torch.from_numpy(tensor))
        if isinstance(result,(tuple,list)): result=result[0]
        return result.detach().cpu().numpy()

class OnnxBackend:
    def __init__(self,path,size=640,threads=1):
        import onnxruntime as ort
        if not Path(path).is_file(): raise FileNotFoundError(path)
        settings=ort.SessionOptions(); settings.intra_op_num_threads=threads; settings.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path),sess_options=settings,providers=['CPUExecutionProvider'])
        inputs=self.session.get_inputs()
        if len(inputs)!=1 or inputs[0].shape!=[1,3,size,size] or inputs[0].type!='tensor(float)':
            raise ValueError('Only static batch-one float32 YOLOv8 export is supported')
        outputs=self.session.get_outputs()
        if len(outputs)!=1 or len(outputs[0].shape)!=3 or outputs[0].shape[:2]!=[1,84]:
            raise ValueError('Expected one raw COCO YOLOv8 output; export with nms=False')
        self.name=inputs[0].name; self.size=size
    def run(self,tensor):
        validate_input(tensor,self.size)
        return self.session.run(None,{self.name:tensor})[0]
