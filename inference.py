import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import argparse
from model import BiSeNet

colors = [
    [255, 255, 255], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0],
    [170, 255, 0], [0, 255, 85], [0, 255, 170], [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255], [0, 170, 255],
    [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255], [255, 170, 255], [0, 255, 255],
    [85, 255, 255], [170, 255, 255],
]

attributes = [
    'background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
    'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default='./makeup/5930.jpg')
    parser.add_argument('--output-path', type=str, default='./makeup/5930_out.jpg')
    parser.add_argument('--model-path', type=str, default='./79999_iter.pth')
    parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    net = BiSeNet(n_classes=19, training=False)
    net.to(device)
    net.load_state_dict(torch.load(args.model_path))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        orignal = cv2.imread(args.image_path)
        orignal = cv2.resize(orignal, (512, 512))
        img = cv2.cvtColor(orignal, cv2.COLOR_BGR2RGB)
        img = to_tensor(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)

        out = net(img)[0]
        out = out.squeeze(0).cpu().numpy().argmax(0)
        out = out.astype(np.uint8)

        found_attributes = [attributes[idx] for idx in np.unique(out)]
        print(found_attributes)

        out_colored = np.zeros((out.shape[0], out.shape[1], 3), np.uint8)
        num_of_class = np.max(out)

        for pi in range(0, num_of_class + 1):
            index = np.where(out == pi)
            out_colored[index[0], index[1], :] = colors[pi]

        result = cv2.addWeighted(orignal, 0.4, out_colored, 0.6, 0)
        cv2.imwrite(args.output_path, result)
