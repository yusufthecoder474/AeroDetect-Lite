"""CPU inference, comparable benchmarks and backend consistency checks."""
import argparse, hashlib, json, platform, sys, time, importlib.metadata
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from src.core import load_image,preprocess,decode,draw,metrics,write_csv,iou
from src.backends import TorchBackend,OnnxBackend

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['infer','benchmark','compare'])
    p.add_argument('--dataset',required=True); p.add_argument('--manifest',default='config/subset.json')
    p.add_argument('--backend',choices=['torch','onnx'],default='onnx')
    p.add_argument('--weights',default='yolov8n.pt'); p.add_argument('--onnx',default='yolov8n.onnx')
    p.add_argument('--size',type=int,default=640); p.add_argument('--threads',type=int,default=1)
    p.add_argument('--warmup',type=int,default=5); p.add_argument('--repeat',type=int,default=5)
    p.add_argument('--confidence',type=float,default=.25); p.add_argument('--iou',type=float,default=.45)
    p.add_argument('--output',default='outputs'); a=p.parse_args()
    if min(a.repeat,a.warmup,a.threads)<1: p.error('Repeat, warmup and threads must be positive')
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); cv2.setNumThreads(1)
    manifest=json.loads(Path(a.manifest).read_text()); paths=[]
    for item in manifest['images']:
        path=Path(a.dataset)/'images'/item['filename']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']: raise ValueError(f'Image changed: {path}')
        paths.append(path)
    if not paths: raise ValueError('Empty manifest')
    backends={}
    names=['torch','onnx'] if a.mode in ('compare','benchmark') else [a.backend]
    for name in names: backends[name]=(TorchBackend(a.weights,a.size,a.threads) if name=='torch' else OnnxBackend(a.onnx,a.size,a.threads))
    def pipeline(backend,path,save=None):
        image=load_image(path); tensor,transform=preprocess(image,a.size)
        prediction=decode(backend.run(tensor),transform,a.confidence,a.iou)
        rendered=draw(image,prediction)
        if save is not None and not cv2.imwrite(str(save),rendered): raise OSError('Image save failed')
        return prediction
    if a.mode=='infer':
        records=[]
        for path in paths:
            start=time.perf_counter(); detections=pipeline(backends[a.backend],path,out/f'{a.backend}_{path.name}')
            records.append({'filename':path.name,'detections':detections.tolist(),'end_to_end_ms':(time.perf_counter()-start)*1000})
        (out/f'{a.backend}_predictions.json').write_text(json.dumps(records,indent=2)); print(f'Saved {len(records)} images'); return
    if a.mode=='compare':
        records=[]
        for path in paths:
            tensor,transform=preprocess(load_image(path),a.size)
            raw1=backends['torch'].run(tensor); raw2=backends['onnx'].run(tensor)
            det1=decode(raw1,transform,a.confidence,a.iou); det2=decode(raw2,transform,a.confidence,a.iou)
            remaining=set(range(len(det2))); matches=[]
            for box in det1:
                candidates=[j for j in remaining if det2[j,5]==box[5]]
                if not candidates: continue
                overlaps=iou(box[:4],det2[candidates,:4]); j=candidates[int(np.argmax(overlaps))]
                if max(overlaps)>=.99 and abs(box[4]-det2[j,4])<=.001:
                    remaining.remove(j); matches.append(float(max(overlaps)))
            records.append({'filename':path.name,'raw_max_abs_error':float(np.max(np.abs(raw1-raw2))),'raw_allclose':bool(np.allclose(raw1,raw2,atol=.01,rtol=.001)),'torch_count':len(det1),'onnx_count':len(det2),'matched':len(matches),'passed':len(matches)==len(det1)==len(det2)})
        (out/'consistency.json').write_text(json.dumps(records,indent=2)); print(json.dumps(records,indent=2))
        if not all(r['passed'] and r['raw_allclose'] for r in records):
            raise SystemExit('Backend consistency check FAILED; see consistency.json')
        return
    rows=[]; raw=[]; failures=[]
    for name,backend in backends.items():
        tensor,_=preprocess(load_image(paths[0]),a.size)
        for _ in range(a.warmup): backend.run(tensor)
        for scope in ['model_only','end_to_end']:
            timings=[]
            for repetition in range(a.repeat):
                for path in paths:
                    try:
                        if scope=='model_only': tensor,_=preprocess(load_image(path),a.size)
                        start=time.perf_counter()
                        if scope=='model_only': backend.run(tensor)
                        else: pipeline(backend,path,out/f'benchmark_{name}.jpg')
                        elapsed=(time.perf_counter()-start)*1000; timings.append(elapsed)
                        raw.append(dict(backend=name,scope=scope,repetition=repetition,filename=path.name,ms=elapsed))
                    except (ValueError,OSError) as error: failures.append(dict(backend=name,scope=scope,filename=path.name,error=str(error)))
            if timings:
                rows.append(dict(backend=name,scope=scope,**metrics(timings),samples=len(timings),images=len(paths),repetitions=a.repeat,warmup=a.warmup,input_size=a.size,batch=1,threads=a.threads,model_bytes=Path(a.weights if name=='torch' else a.onnx).stat().st_size,failures=sum(f['backend']==name and f['scope']==scope for f in failures)))
    write_csv(out/'RESULTS.csv',rows); write_csv(out/'timings.csv',raw)
    (out/'failures.json').write_text(json.dumps(failures,indent=2))
    import psutil
    environment=dict(os=platform.platform(),python=sys.version,cpu=platform.processor(),logical_cpus=psutil.cpu_count(),ram_bytes=psutil.virtual_memory().total,packages={k:importlib.metadata.version(k) for k in ['torch','ultralytics','onnx','onnxruntime','numpy','opencv-python']},settings=vars(a))
    (out/'environment.json').write_text(json.dumps(environment,indent=2)); print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
