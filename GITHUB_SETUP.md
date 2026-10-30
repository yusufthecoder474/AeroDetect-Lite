# GitHub upload and Actions — AeroDetect-Lite

This repository evaluates COCO-pretrained YOLOv8n deployment via PyTorch and ONNX Runtime on VisDrone imagery. The included GitHub Actions workflow runs **core Python unit tests only**; it does not download pretrained weights, the VisDrone dataset or a GPU, and does not reproduce the benchmark or calculate mAP.

## Windows 11 with Python 3.12

1. Extract into `C:\Projects\AeroDetect-Lite` (ensure `requirements.txt` exists directly there).
2. Install Python 3.12 and check `py -3.12 --version` in PowerShell.
3. From PowerShell:

```powershell
cd C:\Projects\AeroDetect-Lite
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -v
```

**Important:** The complete PyTorch environment is large. If the exact pinned packages cannot be installed on Windows, retain the error and inspect version compatibility; do not substitute different versions while claiming identical results.

4. For dataset inference: download VisDrone2019-DET-val using the official repository link in README and extract it into `data/VisDrone2019-DET-val` (contains `images` and `annotations`). Download the attributed `yolov8n.pt` link in README into this project's root.
5. Then run:

```powershell
.\.venv\Scripts\python.exe scripts/export_model.py --weights yolov8n.pt --size 640
.\.venv\Scripts\python.exe scripts/run.py compare --dataset data/VisDrone2019-DET-val
.\.venv\Scripts\python.exe scripts/run.py benchmark --dataset data/VisDrone2019-DET-val --warmup 5 --repeat 5
```

6. Examine `outputs/RESULTS.csv`, `outputs/consistency.json`, and annotated images. Numbers will differ by hardware. mAP/TensorRT remain unverified until genuinely evaluated.

## GitHub

1. Create a new repo `AeroDetect-Lite` on GitHub, Public for recruiter visibility. Skip auto-generated README/license/gitignore.
2. Choose **Add file → Upload files**, and drag **the contents** of the extracted `AeroDetect-Lite` folder, including `.github`. This archive has fewer than 100 files. Do not upload the model weights (`*.pt`, `*.onnx`), `data/`, or `.venv/`.
3. Commit to `main`; open **Actions**. A workflow named `AeroDetect-Lite — Python core CI` should start automatically.
4. Open its run and verify test output. The workflow runs fast tests without the model or dataset; it is not a PyTorch vs ONNX benchmark.
5. Optionally use **Run workflow** to trigger CI again.

Dataset and model licensing notes are in README. Follow repository attribution terms before publishing derivatives.
