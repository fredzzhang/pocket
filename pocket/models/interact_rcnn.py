"""
Implementation of Interact R-CNN

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
from torch import nn
from torchvision.ops.boxes import box_iou

class InteractionHead(nn.Module):
    """
    Arguments:
        box_pair_pooler(nn.Module)
        pooler_output_shape(tuple): (C, H, W)
        representation_size(int): Size of the intermediate representation
        num_classes(int): Number of output classes
        object_class_to_target_class(list[Tensor]): Each element in the list maps an object class
            to corresponding target classes
        fg_iou_thresh(float)
        num_box_pairs_per_image(int): Number of box pairs used in training for each image
        positive_fraction(float): The propotion of positive box pairs used in training
    """
    def __init__(self,
            box_pair_pooler,
            pooler_output_shape, representation_size, num_classes,
            object_class_to_target_class,
            fg_iou_thresh=0.5, num_box_pairs_per_image=-1, positive_fraction=-1):
        
        super().__init__()

        self.box_pair_pooler = box_pair_pooler
        self.box_pair_head = nn.Sequential(
            nn.Linear(torch.as_tensor(pooler_output_shape).prod(), representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, representation_size),
            nn.ReLU()
        )
        self.box_pair_logistic = nn.Linear(representation_size, num_classes)

        self.num_classes = num_classes  

        self.object_class_to_target_class = object_class_to_target_class

        self.fg_iou_thresh = fg_iou_thresh
        self.num_box_pairs_per_image = num_box_pairs_per_image
        self.positive_fraction = positive_fraction

    def pair_up_boxes_and_assign_to_targets(self, boxes, labels, targets=None):
        """
        boxes(list[Tensor[N, 4]])
        labels(list[Tensor[N]])
        target(list[dict])
        """
        if self.training and targets is None:
            raise AssertionError("Targets should be passed during training")
        
        box_pair_idx = []
        box_pair_labels = []
        for idx in range(len(boxes)):
            object_cls = labels[idx]
            # Find detections of human instances
            h_idx = (object_cls == 49).nonzero().squeeze()
            paired_idx = torch.cat([
                v.flatten()[:, None] for v in torch.meshgrid(
                    h_idx, 
                    torch.arange(len(object_cls))
                )
            ], 1)
            # Remove pairs of the same human instance
            keep_idx = torch.eq(paired_idx[:, 0], paired_idx[:, 1]).logical_not().nonzero().squeeze()
            paired_idx = paired_idx[keep_idx, :]

            paired_boxes = boxes[idx][paired_idx, :].view(-1, 8)
            
            labels = torch.zeros(paired_boxes.shape[0], self.num_classes) \
                if self.training else None
            if self.training:
                target_in_image = targets[idx]  
                fg_match = torch.nonzero(torch.min(
                    box_iou(paired_boxes[:, :4], target_in_image['boxes_h']),
                    box_iou(paired_boxes[:, 4:], target_in_image['boxes_o'])
                ) >= self.fg_iou_thresh)
                labels[
                    fg_match[:, 0], 
                    target_in_image['hoi'][fg_match[:, 1]]
                ] = 1

            box_pair_idx.append(paired_idx)
            box_pair_labels.append(labels)

        return box_pair_idx, box_pair_labels

    def map_object_scores_to_interaction_scores(self, box_scores, box_labels, box_pair_idx):
        detection_scores = torch.cat([
            box_scores[per_image_pair_idx[:, 0]] * box_scores[per_image_pair_idx[:, 1]] \
                for per_image_pair_idx in box_pair_idx
        ])
        detection_labels = torch.cat([
            box_labels[per_image_pair_idx[:, 1]] for per_image_pair_idx in box_pair_idx
        ])

        mapped_scores = torch.zeros(len(detection_scores), self.num_classes,
            dtype=detection_scores.dtype, device=detection_scores.device)
        for idx, (obj, score) in enumerate(zip(detection_labels, detection_scores)):
            mapped_scores[idx, self.object_class_to_target_class[obj]] = score
        
        return mapped_scores

    def compute_interaction_classification_loss(self, class_logits, detection_scores, box_pair_labels):
        detection_scores = detection_scores.flatten()
        classification_scores = torch.sigmoid(class_logits).flatten()
        # Disregard the interaction classes that do not contain the detected object
        keep_idx = detection_scores.nonzero().squeeze()

        return torch.nn.functional.binary_cross_entropy_with_logits(
            classification_scores[keep_idx], box_pair_labels[keep_idx])

    def postprocess(self, class_logits, detection_scores, boxes_h, boxes_o):
        num_boxes = [len(boxes_per_image) for boxes_per_image in boxes_h]
        boxes_h = torch.cat(boxes_h, 0)
        boxes_o = torch.cat(boxes_o, 0)

        interaction_scores = detection_scores * torch.sigmodi(class_logits)
        keep_idx = interaction_scores.nonzero()

        all_scores = interaction_scores[keep_idx[:, 0], keep_idx[:, 1]].split(num_boxes, 0)
        all_boxes_h = boxes_h[keep_idx[:, 0], keep_idx[:, 1]].split(num_boxes, 0)
        all_boxes_o = boxes_o[keep_idx[:, 0], keep_idx[:, 1]].split(num_boxes, 0)
        all_labels = keep_idx[:, 1].split(num_boxes, 0)

        results = []
        for scores, b_h, b_o, labels in zip(all_scores, all_boxes_h, all_boxes_o, all_labels):
            results.append(dict(
                boxes_h = b_h,
                boxes_o = b_o,
                labels=labels,
                scores=scores,
            ))

        return results

    def forward(self, features, detections, targets=None):
        """
        Arguments:
            features(list[Tensor]): Image pyramid with each tensor corresponding to
                a feature level
            detections(list[dict]): Object detections with keys boxes(Tensor[N,4]), 
                labels(Tensor[N]) and scores(Tensor[N])
            target(list[dict]): Targets with keys boxes_h(Tensor[N,4]), boxes_o([Tensor[N,4]])
                hoi(Tensor[N]), object(Tensor[N]) and verb(Tensor[N])
        Returns:
            boxes(list[Tensor[N, 4]])
            scores(list[Tensor[N, C]])
        """
        if self.training and targets is None:
            raise AssertionError("Targets should be passed during training")

        box_coords = [detection['boxes'] for detection in detections]
        box_labels = [detection['labels'] for detection in detections]
        box_scores = [detection['scores'] for detection in detections]

        box_pair_idx, box_pair_labels = self.pair_up_boxes_and_assign_to_targets(
            box_coords, box_labels, targets)

        boxes_h = [boxes[per_image_pair_idx[:, 0]] 
            for per_image_pair_idx, boxes in zip(box_pair_idx, box_coords)]
        boxes_o = [boxes[per_image_pair_idx[:, 1]] 
            for per_image_pair_idx, boxes in zip(box_pair_idx, box_coords)]

        box_pair_features = self.box_pair_pooler(features, boxes_h, boxes_o)
        box_pair_features = box_pair_features.flatten(start_dim=1)
        box_pair_features = self.box_pair_head(box_pair_features)
        class_logits = self.box_pair_logistic(box_pair_features)

        detection_scores = self.map_object_scores_to_interaction_scores(
            box_scores, box_labels, box_pair_idx)

        if self.training:
            loss = dict(interaction_loss=self.compute_interaction_classification_loss(
                class_logits, detection_scores, torch.cat(box_pair_labels, 0)
            ))
            return loss
        
        results = self.postprocess(
            class_logits, detection_scores, boxes_h, boxes_o)

        return results


class InteractRCNN(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, interaction_heads, transform):
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.interaction_heads = interaction_heads
        self.transform = transform

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]], optional): ground-truth boxes present in the image
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals,
            images.image_sizes, targets)
        detections, interaction_loss = self.interaction_heads(features, detections,
            images.image_sizes, targets)    
        detections = self.transform.postprocess(detections,
            images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(interaction_loss)

        if self.training:
            return losses

        return detections
