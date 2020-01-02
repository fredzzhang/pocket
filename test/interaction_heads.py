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

from pocket.data import HICODet
from pocket.models import fasterrcnn_resnet_fpn
from pocket.models.interact_rcnn import InteractionHead
from pocket.ops import BoxPairMultiScaleRoIAlign, to_tensor, ToTensor

def test(args):

    dataset = HICODet(
        root=os.path.join(args.data_root, "hico_20160224_det/images/train2015"),
        annoFile=os.path.join(args.data_root, "instances_train2015.json"),
        transform=torchvision.transforms.ToTensor(),
        target_transform=ToTensor(input_format='dict')
    )

    with open(os.path.join(args.data_root, "hico80to600.json"), 'r') as f:
        hico_obj_to_hoi = to_tensor(json.load(f), 
            input_format='list', dtype=torch.int64)
    
    backbone = fasterrcnn_resnet_fpn('resnet50', pretrained=True).backbone

    box_pair_pooler = BoxPairMultiScaleRoIAlign(
        output_size=7,
        spatial_scale=[1/4, 1/8, 1/16, 1/32],
        sampling_ratio=2
    )

    interaction_head = InteractionHead(
        box_pair_pooler,
        (backbone.out_channels, 7, 7),
        1024, 600,
        object_class_to_target_class=hico_obj_to_hoi
    )

    image, target = dataset[args.image_idx]
    detection_path = os.path.join(args.data_root, 
        "fasterrcnn_resnet50_fpn_detections/train2015/{}".format(
            dataset.filename(args.image_idx).replace('jpg', 'json'))
    )
    with open(detection_path, 'r') as f:
        detection = to_tensor(json.load(f), input_format='dict')

    features = backbone(image[None, :, :, :])
    features = [v for v in features.values()]
    # Remove the last max pooled features
    features = features[:-1]
    with torch.no_grad():
        results = interaction_head(
            features, [detection], targets=[target]
        )

    if args.mode == 'train':
        print(results['interaction_loss'])
    else:
        print(results['labels'].shape)

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
    args = parser.parse_args()

    test(args)
