import importlib.util,json,shutil
import torch
from pathlib import Path
status={'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,'nvidia_smi':shutil.which('nvidia-smi'),'tensorrt_installed':importlib.util.find_spec('tensorrt') is not None,'status':'NOT EXECUTED','reason':'CPU pipeline only; no compatible GPU verified'}
Path('outputs/gpu_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))
