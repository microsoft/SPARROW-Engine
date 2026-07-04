# RT-DETR implementation notes (ONB-2)

Plan:
- Add manifest method `rtdetr_topk` with optional `topk`.
- Decode packed RT-DETR ONNX rows `[cx, cy, w, h, score, class_id]` from `[N,6]` or squeezed `[1,N,6]` outputs.
- Wire CPU and GPU detector shape validation and postprocess dispatch through the shared decoder.

Implemented contract:
- Coordinates are normalized `[0,1]` center-format in the direct-resize / scale-fill input frame.
- Decoder converts normalized `cxcywh` directly to normalized `xyxy`, clamps to `[0,1]`, and does not call letterbox undo.
- `score` is assumed already sigmoid-applied by the exported graph; rows below the detection threshold are skipped before geometry validation.
- RT-DETR is NMS-free; no Rust NMS is applied.
- Final result count is capped by the smaller of manifest `topk` and request `max_detections`.
