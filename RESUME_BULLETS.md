# Resume bullets — verified local prototype

- Built a Python computer-vision pipeline using pretrained YOLOv8n, OpenCV and ONNX Runtime to detect people and vehicles in a fixed 10-image VisDrone validation subset.
- Exported and validated an FP32 ONNX model; measured 101.68 ms mean CPU model-call latency versus 124.40 ms in PyTorch across 50 timed calls per backend (1.22× throughput ratio in this local test).
- Implemented 30 passing pytest tests and checked numerical consistency across both runtimes, matching all 132 final detections on the subset.

Do not claim model training, full-dataset accuracy, TensorRT execution, production deployment, real-time drone operation or real hardware testing. Model-call timing includes validation and array conversion; end-to-end timing is reported separately. Explain your own reviewed and understood contribution honestly; this implementation was developed with AI assistance.
