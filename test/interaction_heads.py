import os
import json
import torch
import torchvision

from pocket.data import HICODet
from pocket.models import fasterrcnn_resnet_fpn
from pocket.models.interact_rcnn import InteractionHead
from pocket.ops import BoxPairMultiScaleRoIAlign, to_tensor, ToTensor

DATA_ROOT = "/MyData/Github/InteractRCNN/data/"

def test(image_idx):

    dataset = HICODet(
        root=os.path.join(DATA_ROOT, "hico_20160224_det/images/train2015"),
        annoFile=os.path.join(DATA_ROOT, "instances_train2015.json"),
        transform=torchvision.transforms.ToTensor(),
        target_transform=ToTensor(input_format='dict')
    )

    with open(os.path.join(DATA_ROOT, "hico80to600.json"), 'r') as f:
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

    image, target = dataset[image_idx]
    detection_path = os.path.join(DATA_ROOT, 
        "fasterrcnn_resnet50_fpn_detections/train2015/{}".format(
            dataset.filename(image_idx).replace('jpg', 'json'))
    )
    with open(detection_path, 'r') as f:
        detection = to_tensor(json.load(f), input_format='dict')

    features = backbone(image[None, :, :, :])
    features = [v for v in features.values()]
    # Remove the last max pooled features
    features = features[:-1]
    loss = interaction_head(
        features, [detection], targets=[target]
    )

    print(loss)

if __name__ == '__main__':
    test(0)
