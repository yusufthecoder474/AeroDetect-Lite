# AeroDetect-Lite: interview preparation

## 1. What is object detection?
It identifies what objects appear in an image and gives a bounding box for each. Classification only gives a label for an image; detection also locates individual objects. This project displays people and selected vehicle categories.

## 2. Why use YOLO?
YOLO predicts boxes and class scores in one network pass. The small YOLOv8n model is practical for a student CPU deployment experiment. It is pretrained by Ultralytics; I did not design or train the neural network. Small aerial objects remain challenging.

## 3. What is ONNX?
ONNX is a model interchange format. It represents a trained network as a graph of operations and parameters. It lets a model leave its original training framework.

## 4. What is ONNX Runtime?
It executes an ONNX graph using an execution provider, such as the CPU provider used here. ONNX is the file format; ONNX Runtime is the program that runs it.

## 5. Why export a model?
Export separates deployment from the training framework and allows runtime optimizations. Export alone does not guarantee lower latency. I measure the same model, resolution, images and batch size on the same CPU before comparing performance.

## 6. What is preprocessing?
I decode an image, resize it while preserving its aspect ratio, add gray padding to make a 640 by 640 square, convert BGR to RGB, scale pixels to 0–1 and arrange them as float32 NCHW. The saved resize ratio and padding let me return predicted boxes to the original image coordinates.

## 7. What is non-maximum suppression?
Several predictions may describe the same object. NMS keeps a high-confidence box and suppresses lower-confidence overlapping boxes of the same class. Intersection over union measures overlap. Different-class boxes are not suppressed against each other here. Crowds can still cause missed objects.

## 8. What are precision, recall and mAP?
Precision is TP/(TP+FP); recall is TP/(TP+FN). AP summarizes a precision–recall curve for one class, and mAP averages AP over classes. mAP@0.5 uses IoU 0.5; mAP@0.5:0.95 averages multiple IoU thresholds. They require compatible labels, matching rules and correct ignored-region handling. This project does not report accuracy scores because the COCO model does not implement the full VisDrone taxonomy and an official evaluation integration has not been verified. Backend consistency is not detection accuracy.

## 9. How do you measure latency?
I warm up inference, use a monotonic high-resolution timer, repeat a fixed image list and save every observation. Model-only measurements include the backend call and its validation and array conversion, but exclude preprocessing and NMS. End-to-end measurements include decoding, preprocessing, inference, postprocessing, rendering and JPEG saving. Mean, median and p95 describe different parts of the distribution. Throughput is 1000/mean_ms for sequential batch-one processing. This is not a camera frame-rate claim.

## 10. When would you use TensorRT?
When deploying to a compatible NVIDIA GPU and when measurements justify it. I would check its support matrix, build an engine, check detections against the original model, and compare with ONNX Runtime on that same GPU. TensorRT was not executed here because no compatible GPU was available. CPU latency cannot be used as a fair GPU optimization baseline.

## 30-second explanation
AeroDetect-Lite is a Python deployment project that detects people and vehicles in drone images using pretrained YOLOv8n. I export it to ONNX, run an independent ONNX Runtime CPU pipeline, compare predictions and benchmark latency on a fixed VisDrone subset. My contribution is the data validation, shared transforms, decoding, tests and reproducible measurement workflow. I did not train the model or test drone hardware.

## 2-minute explanation
Drone images contain many small, crowded objects. I wanted to explore deploying an existing detector with limited resources. I selected YOLOv8n and froze ten validation images with filenames, hashes, original sizes and annotations.

Both backends use the same 640-pixel letterbox transform and class-aware NMS. The ONNX inference path needs neither PyTorch nor TensorRT at runtime. Export validation checks the graph and records its operators, tensor shapes and checkpoint hash. A comparison script examines raw numerical differences and matches final detections by class and box overlap.

I measure model-only and end-to-end latency separately, after warm-up, with one thread and batch size one. RESULTS.csv contains actual timings, and outputs/timings.csv retains the observations. I would quote the current CSV rather than promise that export always makes a model faster.

The main limitation is the model's training distribution and label taxonomy. COCO person does not map directly to VisDrone's pedestrian/people split. The project therefore does not claim full VisDrone mAP. Tiny objects may be missed, and similar outputs across runtimes only prove export consistency, not correctness against ground truth. GPU optimization is a future experiment requiring actual compatible hardware.

## 5-minute project demo
1. **0:00–0:45:** Explain the aerial small-object problem and open README.md. Identify model deployment as the scope.
2. **0:45–1:30:** Open config/subset.json. Show filenames, dimensions, hashes and unchanged annotation fields. Explain the ignored-region label.
3. **1:30–2:30:** Run ONNX inference using the README command. Open an annotated image. Point out both plausible detections and missed small objects; never describe the picture as perfect accuracy.
4. **2:30–3:15:** Open outputs/export.json and consistency.json. Explain graph validation, shape checks and numerical tolerance.
5. **3:15–4:15:** Open RESULTS.csv and the latency plot. Distinguish model-only from the full image-processing path and discuss hardware dependence.
6. **4:15–5:00:** Run pytest. Summarize class mismatch, CPU-only scope, no training, no tracking and no hardware flight integration.
