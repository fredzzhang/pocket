"""
Interation head interfacing with backbone CNN and RPN

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""
import time
import os
import json
import torch
import argparse
import torchvision
import pocket
from pocket.data import HICODet
from pocket.models import TrainableHead

torch.manual_seed(0)
torch.set_printoptions(threshold=1000)

def test(args):

    use_gpu = torch.cuda.is_available() and args.gpu

    dataset = HICODet(
        root=os.path.join(args.data_root,
            "hico_20160224_det/images/{}".format(args.partition)),
        annoFile=os.path.join(args.data_root,
            "instances_{}.json".format(args.partition)),
        transform=torchvision.transforms.ToTensor(),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    )

    interaction_head = TrainableHead(
            dataset.object_to_interaction, 49,
            masked_pool=args.masked_pool)
    if use_gpu:
        interaction_head = interaction_head.cuda()

    if args.mode != 'train':
        interaction_head.eval()

    image, target = dataset[args.image_idx]
    detection_path = os.path.join(args.data_root, 
        "fasterrcnn_resnet50_fpn_detections/{}/{}".format(
            args.partition,
            dataset.filename(args.image_idx).replace('jpg', 'json'))
    )
    with open(detection_path, 'r') as f:
        detection = pocket.ops.to_tensor(json.load(f), input_format='dict')

    target['labels'] = target['hoi']
    if use_gpu:
        image, detection, target = pocket.ops.relocate_to_cuda(
            (image, detection, target), 0)

    t = time.time()
    with torch.no_grad():
        results = interaction_head(
            [image], [detection], targets=[target]
        )
    print("Execution time: ", time.time() - t)

    if args.mode == 'train':
        print(results)
    else:
        for label in results[0]['labels']:
            print(label.tolist())

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Test interaction head")
    parser.add_argument('--data-root',
                        required=True,
                        type=str)
    parser.add_argument('--partition',
                        default='train2015',
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
    parser.add_argument('--masked-pool',
                        action='store_true',
                        default=False)
    args = parser.parse_args()

    test(args)
