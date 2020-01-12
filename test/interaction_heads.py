"""
Interation head interfacing with backbone CNN and RPN

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import torch
import argparse
import torchvision
import pocket
from pocket.data import HICODet
from pocket.models import TrainableHead

torch.set_printoptions(threshold=1000)

def test(args):

    use_gpu = torch.cuda.is_available() and args.gpu

    dataset = HICODet(
        root=os.path.join(args.data_root, "hico_20160224_det/images/train2015"),
        annoFile=os.path.join(args.data_root, "instances_train2015.json"),
        transform=torchvision.transforms.ToTensor(),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    )

    with open(os.path.join(args.data_root, "hico80to600.json"), 'r') as f:
        hico_obj_to_hoi = json.load(f) 

    interaction_head = TrainableHead(hico_obj_to_hoi)
    if use_gpu:
        interaction_head = interaction_head.cuda()

    if args.mode != 'train':
        interaction_head.eval()

    image, target = dataset[args.image_idx]
    detection_path = os.path.join(args.data_root, 
        "fasterrcnn_resnet50_fpn_detections/train2015/{}".format(
            dataset.filename(args.image_idx).replace('jpg', 'json'))
    )
    with open(detection_path, 'r') as f:
        detection = pocket.ops.to_tensor(json.load(f), input_format='dict')

    if use_gpu:
        image, detection, target = pocket.ops.relocate_to_cuda(
            (image, detection, target), 0)
    with torch.no_grad():
        results = interaction_head(
            [image], [detection], targets=[target]
        )

    if args.mode == 'train':
        print(results['interaction_loss'])
    else:
        for label in results[0]['labels']:
            print(label)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Test interaction head")
    parser.add_argument('--data-root',
                        default="/MyData/Github/InteractRCNN/data/",
                        type=str)
    parser.add_argument('--image-idx',
                        default=0,
                        type=int)
    parser.add_argument('--mode',
                        default='train',
                        type=str)
    parser.add_argument('--gpu',
                        action='store_true',
                        default=False)
    args = parser.parse_args()

    test(args)
