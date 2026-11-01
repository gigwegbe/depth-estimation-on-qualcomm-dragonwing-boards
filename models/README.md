# Models

The model weights are not committed to git (see `.gitignore`). Download them
from [Qualcomm AI Hub](https://aihub.qualcomm.com/) and place them here.

Expected layout:

```
models/
  depth_anything/depth_anything.tflite   # IQ-9075 (float TFLite)
  midas/midas.tflite                      # Rubik Pi 3 (MiDaS V2, w8a8 quantized)
```

## IQ-9075: Depth Anything (float TFLite)

The IQ-9075 (QCS9075) Hexagon NPU supports FP16, so the float TFLite graph runs
directly, no quantization needed. Fetch the TFLite build from Qualcomm AI Hub
and save it as `models/depth_anything/depth_anything.tflite`.

## Rubik Pi 3: MiDaS V2 (w8a8 quantized TFLite)

The QCS6490 Hexagon V68 NPU does not support FP16, so use the INT8 (w8a8)
quantized MiDaS V2 build. A float graph fails op validation on this NPU and
falls back to CPU. Save it as `models/midas/midas.tflite`.

Both scripts default to these paths. Point elsewhere with `--model <path>` if
you keep the weights outside the repo.
