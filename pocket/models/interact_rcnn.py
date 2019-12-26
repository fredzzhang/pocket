"""
Implementation of Interact R-CNN

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
from torch import nn
from torchvision.ops.boxes import box_iou
from torchvision.ops._utils import _cat

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
            fg_iou_thresh=0.5, num_box_pairs_per_image=512, positive_fraction=0.25,
            detection_score_thresh=0.2):
        
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
        self.detection_score_thresh = detection_score_thresh

    def remove_low_confidence_detections(self, detections):
        """
        detections(list[dict]): Object detections with keys boxes(Tensor[N,4]), 
            labels(Tensor[N]) and scores(Tensor[N])
        """
        for detection in detections:
            keep_idx = torch.nonzero(detection['scores'] >= self.detection_score_thresh).squeeze()
            detection['boxes'] = detection['boxes'][keep_idx, :].view(-1, 4)
            detection['scores'] = detection['scores'][keep_idx].view(-1)
            detection['labels'] = detection['labels'][keep_idx].view(-1)

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
            keep_idx = (paired_idx[:, 0] != paired_idx[:, 1]).nonzero().squeeze()
            paired_idx = paired_idx[keep_idx, :].view(-1, 2)

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
                # Sample up to a specified number of box pairs
                label_sum = labels.sum(1)
                pos_idx = label_sum.nonzero().squeeze()
                neg_idx = (label_sum == 0).nonzero().squeeze()
                # Use all positive samples if there are fewer than specified
                if len(pos_idx) < self.num_box_pairs_per_image * self.positive_fraction:
                    sampled_idx = torch.cat([
                        pos_idx, neg_idx[torch.randperm(len(neg_idx))[
                            :int(len(pos_idx) * (1-self.positive_fraction) / self.positive_fraction)]]
                    ])
                else:
                    sampled_idx = torch.cat([
                        pos_idx[torch.randperm(len(pos_idx))[
                            :int(self.num_box_pairs_per_image * self.positive_fraction)]],
                        neg_idx[torch.randperm(len(neg_idx))[
                            :int(self.num_box_pairs_per_image * (1-self.positive_fraction))]]    
                    ])
                paired_idx = paired_idx[sampled_idx, :].view(-1, 2)
                labels = labels[sampled_idx, :].view(-1, 2)

            box_pair_idx.append(paired_idx)
            box_pair_labels.append(labels)

        return box_pair_idx, box_pair_labels

    def map_object_scores_to_interaction_scores(self, box_scores, box_labels, box_pair_idx):
        detection_scores = _cat([
            scores[per_image_pair_idx[:, 0]] * scores[per_image_pair_idx[:, 1]] \
                for per_image_pair_idx, scores in zip(box_pair_idx, box_scores)
        ])
        detection_labels = _cat([
            labels[per_image_pair_idx[:, 1]] for per_image_pair_idx, labels in zip(box_pair_idx, box_labels)
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
            classification_scores[keep_idx], box_pair_labels.flatten()[keep_idx])

    def postprocess(self, class_logits, detection_scores, boxes_h, boxes_o):
        num_boxes = [len(boxes_per_image) for boxes_per_image in boxes_h]
        interaction_scores = (detection_scores
                * torch.sigmoid(class_logits)).split(num_boxes)

        results = []
        for scores, b_h, b_o in zip(interaction_scores, boxes_h, boxes_o):
            keep_idx = scores.nonzero()
            results.append(dict(
                boxes_h = b_h[keep_idx[:, 0], :],
                boxes_o = b_o[keep_idx[:, 0], :],
                labels=keep_idx[:, 1],
                scores=scores[keep_idx[:, 0], keep_idx[:, 1]],
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
        if not self.training:
            self.remove_low_confidence_detections(detections)

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
