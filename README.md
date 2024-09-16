# Fork of [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)

Differences between original repository and fork:

* Compatibility with PyTorch >=2.4. (🔥)
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

| Name               | Model Size (MB) | Link                                                                                                                                                                                                                     | SHA-256                                                                                                                              |
|--------------------|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| FaceParsingBiSeNet | 50.8<br>50.7    | [PyTorch](https://github.com/clibdev/face-parsing.PyTorch/releases/latest/download/face-parsing-bisenet.pth), [ONNX](https://github.com/clibdev/face-parsing.PyTorch/releases/latest/download/face-parsing-bisenet.onnx) | 468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567<br>37cc52ffdf0ccb45a555a4a1a52d59266959da5cee981c244177d9768447ff37 |

# Inference

```shell
python inference.py --model-path face-parsing-bisenet.pth --image-path makeup/5930.jpg --output-path makeup/5930_out.jpg
```

# Export to ONNX format

```shell
pip install onnx
```
```shell
python export.py --model-path face-parsing-bisenet.pth --dynamic
```
