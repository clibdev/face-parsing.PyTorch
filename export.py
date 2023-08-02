import torch
import argparse
import os
from model import BiSeNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='./79999_iter.pth')
    parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
    parser.add_argument('--dynamic', action='store_true', default=False, help='enable dynamic axis in onnx model')
    args = parser.parse_args()

    device = torch.device(args.device)

    net = BiSeNet(n_classes=19, training=False)
    net.to(device)
    net.load_state_dict(torch.load(args.model_path))
    net.eval()

    model_path = os.path.splitext(args.model_path)[0] + '.onnx'

    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    dynamic_axes = {'input': {2: '?', 3: '?'}, 'output': {2: '?', 3: '?'}} if args.dynamic else None
    torch.onnx.export(
        net,
        dummy_input,
        model_path,
        verbose=False,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=17
    )
