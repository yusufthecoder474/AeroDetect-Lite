"""Fetch versioned public assets; preserve only fixed-manifest VisDrone images locally."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

MODEL_URL='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt'
DATA_URL='https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip'

def fetch(url, path):
    if path.is_file(): return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.part')
    try:
        with urllib.request.urlopen(url,timeout=120) as source, temporary.open('wb') as target:
            shutil.copyfileobj(source,target)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); a=p.parse_args()
    root=a.root.resolve()
    reference=json.loads((root/'outputs/export.json').read_text())
    fetch(MODEL_URL,root/'yolov8n.pt')
    if hashlib.sha256((root/'yolov8n.pt').read_bytes()).hexdigest()!=reference['sha256']:
        raise ValueError('Checkpoint SHA-256 differs from the verified original; refusing to export')
    manifest=json.loads((root/'config/subset.json').read_text())
    archive=root/'data/VisDrone2019-DET-val.zip'
    dataset=root/'data/VisDrone2019-DET-val'
    expected=json.loads((root/'outputs/provenance.json').read_text())['archive_sha256']
    fetch(DATA_URL,archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:
        raise ValueError('Validation archive SHA-256 mismatch')
    with zipfile.ZipFile(archive) as z:
        for record in manifest['images']:
            name=record['filename']
            if Path(name).name!=name: raise ValueError('Manifest filename must be a basename')
            for folder,filename,key in [('images',name,'sha256'),('annotations',Path(name).stem+'.txt','annotation_sha256')]:
                data=z.read(f'VisDrone2019-DET-val/{folder}/{filename}')
                if hashlib.sha256(data).hexdigest()!=record[key]: raise ValueError(f'Hash mismatch: {filename}')
                dest=dataset/folder/filename;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    print(f'Prepared {len(manifest["images"])} fixed original validation images in {dataset}')
if __name__=='__main__': main()
