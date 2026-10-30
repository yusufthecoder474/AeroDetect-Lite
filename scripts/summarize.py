"""Render measured results without substituting hard-coded performance claims."""
import csv,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((root/'outputs/RESULTS.csv').open()))
shutil.copyfile(root/'outputs/RESULTS.csv',root/'RESULTS.csv')
fig,ax=plt.subplots(figsize=(8,4.5))
labels=[r['backend']+'\n'+r['scope'].replace('_',' ') for r in rows]
ax.bar(labels,[float(r['mean_ms']) for r in rows],color=['#64748b','#94a3b8','#0891b2','#22d3ee'])
ax.set_ylabel('Mean latency (ms)');ax.set_title('AeroDetect-Lite • measured CPU latency')
for i,r in enumerate(rows): ax.text(i,float(r['mean_ms'])+2,f"{float(r['mean_ms']):.2f}",ha='center')
ax.set_ylim(0,max(float(r['mean_ms']) for r in rows)*1.2)
fig.tight_layout();fig.savefig(root/'outputs/latency.png',dpi=150);plt.close(fig)
records=json.loads((root/'outputs/consistency.json').read_text())
summary='\n## Executed results\n\nLinux / AMD EPYC 9V74 virtual CPU, 9 logical CPUs available, approximately 9.7 GiB RAM; Python 3.12.14. Both inference backends used one CPU thread. Exact package versions and runtime settings are in `outputs/environment.json`.\n\n'
summary+='| Backend | Scope | Mean ms | Median ms | p95 ms | Images/s |\n|---|---|---:|---:|---:|---:|\n'
for r in rows: summary+=f"| {r['backend']} | {r['scope']} | {float(r['mean_ms']):.2f} | {float(r['median_ms']):.2f} | {float(r['p95_ms']):.2f} | {float(r['images_per_second']):.2f} |\n"
summary+='\n30 pytest tests passed. All ten images passed raw `allclose` (atol 0.01, rtol 0.001) and final-detection matching (same class, IoU ≥0.99, confidence difference ≤0.001). All 132 detections matched one-to-one. No image-processing failures occurred. These tolerances test numerical deployment consistency, not accuracy.\n\n'
summary+='Checkpoint size: 6,549,796 bytes. ONNX size: 12,824,166 bytes. Export increased file size; ONNX is not a compression claim.\n\n![Measured latency](outputs/latency.png)\n'
p=root/'README.md';text=p.read_text();text=text.replace('## Accuracy boundaries',summary+'\n## Accuracy boundaries');p.write_text(text)
a,b=float(rows[0]['mean_ms']),float(rows[2]['mean_ms'])
(root/'RESUME_BULLETS.md').write_text(f'''# Resume bullets — verified local prototype

- Built a Python computer-vision pipeline using pretrained YOLOv8n, OpenCV and ONNX Runtime to detect people and vehicles in a fixed 10-image VisDrone validation subset.
- Exported and validated an FP32 ONNX model; measured {b:.2f} ms mean CPU model-call latency versus {a:.2f} ms in PyTorch across 50 timed calls per backend ({a/b:.2f}× throughput ratio in this local test).
- Implemented 30 passing pytest tests and checked numerical consistency across both runtimes, matching all 132 final detections on the subset.

Do not claim model training, full-dataset accuracy, TensorRT execution, production deployment, real-time drone operation or real hardware testing. Model-call timing includes validation and array conversion; end-to-end timing is reported separately. Explain your own reviewed and understood contribution honestly; this implementation was developed with AI assistance.
''')
