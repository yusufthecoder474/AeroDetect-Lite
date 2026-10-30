"""Freeze a seeded selection; preserve annotations, dimensions and image hashes."""
import sys, argparse, random, json, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import discover,load_image,annotations
p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--count',type=int,default=10); p.add_argument('--seed',type=int,default=42); p.add_argument('--output',default='config/subset.json'); a=p.parse_args()
root=Path(a.dataset); paths=discover(root/'images')
if not 1<=a.count<=len(paths): p.error('Count outside dataset size')
selected=sorted(random.Random(a.seed).sample(paths,a.count)); records=[]
for image in selected:
    label=root/'annotations'/f'{image.stem}.txt'; ann=annotations(label); im=load_image(image)
    records.append({'filename':image.name,'width':im.shape[1],'height':im.shape[0],'sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'annotation_sha256':hashlib.sha256(label.read_bytes()).hexdigest(),'annotations':ann.tolist()})
Path(a.output).write_text(json.dumps({'dataset':'VisDrone2019-DET-val','seed':a.seed,'images':records},indent=2))
