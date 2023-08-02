# Fork of [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)

Differences between original repository and fork:

* Compatibility with PyTorch >=2.0. (🔥)
* Original pretrained models and converted ONNX models from GitHub [releases page](https://github.com/clibdev/face-parsing.PyTorch/releases). (🔥)
* Model conversion to ONNX format using the [export.py](export.py) file. (🔥)
* Installation with [requirements.txt](requirements.txt) file.
* Sample script [inference.py](inference.py) for inference of single image.
* Minor improvements to make the model more ONNX compatible.

# Installation

```shell
pip install -r requirements.txt
```

# Pretrained models

| Name               | Link                                                                                                                                                                                                 |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FaceParsingBiSeNet | [PyTorch](https://github.com/clibdev/face-parsing.PyTorch/releases/latest/download/79999_iter.pth), [ONNX](https://github.com/clibdev/face-parsing.PyTorch/releases/latest/download/79999_iter.onnx) |

# Inference

```shell
python inference.py --model-path 79999_iter.pth --image-path makeup/5930.jpg --output-path makeup/5930_out.jpg
```

# Export to ONNX format

```shell
pip install onnx
```
```shell
python export.py --model-path 79999_iter.pth --dynamic
```
