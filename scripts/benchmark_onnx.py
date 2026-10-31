"""Standalone CPU benchmark without importing torch. Keep original benchmark intact."""
import argparse,csv,hashlib,importlib.metadata,json,platform,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from src.core import load_image,preprocess,decode,draw,metrics,write_csv
from src.backends import OnnxBackend

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True,type=Path);p.add_argument('--manifest',default='config/subset.json',type=Path)
    p.add_argument('--onnx',default='yolov8n.onnx',type=Path);p.add_argument('--output',default='outputs/current',type=Path)
    p.add_argument('--repeat',default=5,type=int);p.add_argument('--warmup',default=5,type=int);p.add_argument('--threads',default=1,type=int)
    a=p.parse_args()
    if min(a.repeat,a.warmup,a.threads)<1:p.error('repeat, warmup and threads must be positive')
    a.output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    paths=[]
    for item in json.loads(a.manifest.read_text())['images']:
        path=a.dataset/'images'/item['filename']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']: raise ValueError(f'Image hash mismatch: {path}')
        paths.append(path)
    if not paths:raise ValueError('Empty manifest')
    backend=OnnxBackend(a.onnx,640,a.threads)
    def full(path,save):
        image=load_image(path);tensor,transform=preprocess(image,640)
        pred=decode(backend.run(tensor),transform)
        if not cv2.imwrite(str(save),draw(image,pred)):raise OSError('Cannot save output')
        return pred
    rows=[];raw=[];failures=[]
    for scope in ['model_call','end_to_end']:
        tensor,_=preprocess(load_image(paths[0]),640)
        for _ in range(a.warmup):
            if scope=='model_call':backend.run(tensor)
            else:full(paths[0],a.output/'warmup.jpg')
        timings=[]
        for repetition in range(a.repeat):
            for path in paths:
                try:
                    if scope=='model_call':tensor,_=preprocess(load_image(path),640)
                    started=time.perf_counter_ns()
                    if scope=='model_call':backend.run(tensor)
                    else:full(path,a.output/f'onnx_{path.name}')
                    ms=(time.perf_counter_ns()-started)/1e6;timings.append(ms)
                    raw.append(dict(scope=scope,repetition=repetition,filename=path.name,ms=ms))
                except (ValueError,OSError) as error:failures.append(dict(scope=scope,filename=path.name,error=str(error)))
        if timings:rows.append(dict(backend='onnx_cpu',scope=scope,**metrics(timings),samples=len(timings),images=len(paths),repeat=a.repeat,warmup=a.warmup,threads=a.threads,size=640,batch=1,model_bytes=a.onnx.stat().st_size,failures=sum(f['scope']==scope for f in failures)))
    (a.output/'failures.json').write_text(json.dumps(failures,indent=2))
    if rows:write_csv(a.output/'RESULTS.csv',rows)
    if raw:write_csv(a.output/'timings.csv',raw)
    import psutil
    env=dict(os=platform.platform(),python=sys.version,cpu=platform.processor(),cpus=psutil.cpu_count(),ram_bytes=psutil.virtual_memory().total,provider=backend.session.get_providers(),model_sha256=hashlib.sha256(a.onnx.read_bytes()).hexdigest(),packages={k:importlib.metadata.version(k) for k in ['numpy','onnxruntime','onnx','opencv-python-headless']},scope='model_call includes validation and array handling; end_to_end includes decode/read, transforms, inference, NMS, drawing and cached JPEG writes; model load excluded')
    (a.output/'environment.json').write_text(json.dumps(env,indent=2));print(json.dumps(rows,indent=2))
    if failures:raise SystemExit('Benchmark had image-processing failures; see failures.json')
if __name__=='__main__':main()
