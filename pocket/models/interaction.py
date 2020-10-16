"""
Models related to human-object interaction

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torchvision.ops.boxes as box_ops

from torch import nn
from torchvision.ops._utils import _cat
from torchvision.models.detection import transform

class InteractionHead(nn.Module):
    """Interaction head that constructs and classifies box pairs

    Arguments:

    [REQUIRES ARGS]
        box_roi_pool(nn.Module): Module that performs RoI pooling or its variants
        box_pair_head(nn.Module): Module that constructs and computes box pair features
        box_pair_logistic(nn.Module): Module that classifies box pairs
        human_idx(int): The index of human/person class in all objects

    [OPTIONAL ARGS]
        box_nms_thresh(float): NMS threshold
    """
    def __init__(self,
                box_roi_pool,
                box_pair_head,
                box_pair_predictor,
                human_idx,
                box_nms_thresh=0.5
                ):
        
        super().__init__()

        self.box_roi_pool = box_roi_pool
        self.box_pair_head = box_pair_head
        self.box_pair_predictor = box_pair_predictor

        self.human_idx = human_idx
        self.box_nms_thresh = box_nms_thresh

    def preprocess(self, detections, targets):
        """
        detections(list[dict]): Object detections with following keys 
            "boxes": Tensor[N, 4]
            "labels": Tensor[N] 
            "scores": Tensor[N]
        targets(list[dict]): Targets with the following keys
            "boxes_h" Tensor[L, 4]
            "boxes_o": Tensor[L, 4]
            "object": Tensor[L] Object class index
        """
        results = []
        for b_idx, detection in enumerate(detections):
            boxes = detection['boxes']
            labels = detection['labels']
            scores = detection['scores']

            # Append ground truth during training
            if self.training:
                target = targets[b_idx]
                n = target["boxes_h"].shape[0]
                boxes = torch.cat([target["boxes_h"], target["boxes_o"], boxes])
                scores = torch.cat([torch.ones(2 * n, device=scores.device), scores])
                labels = torch.cat([
                    self.human_idx * torch.ones(n, device=labels.device).long(),
                    target["object"],
                    labels
                ])

            # Class-wise non-maximum suppression
            keep_idx = box_ops.batched_nms(
                boxes, scores, labels,
                self.box_nms_thresh
            )
            boxes = boxes[keep_idx].view(-1, 4)
            scores = scores[keep_idx].view(-1)
            labels = labels[keep_idx].view(-1)

            results.append(dict(boxes=boxes, labels=labels, scores=scores))

        return results

    def compute_interaction_classification_loss(self, scores, box_pair_labels):
        # Ignore irrelevant target classes
        i, j = scores.nonzero().unbind(1)

        return nn.functional.binary_cross_entropy(
            scores[i, j],
            box_pair_labels[i, j]
        )

    def postprocess(self, scores, boxes_h, boxes_o, labels=None):
        num_boxes = [len(boxes_per_image) for boxes_per_image in boxes_h]
        scores = scores.split(num_boxes)
        if labels is None:
            labels = [[] for _ in range(len(num_boxes))]

        results = []
        for s, b_h, b_o, l in zip(scores, boxes_h, boxes_o, labels):
            # Remove irrelevant classes
            keep_cls = [row.nonzero().squeeze(1) for row in s]
            # Remove box pairs without predictions
            keep_idx = {
                k: v for k, v in enumerate(keep_cls) if len(v)
            }

            box_keep = list(keep_idx.keys())
            if len(box_keep) == 0:
                continue
            result_dict = dict(
                boxes_h=b_h[box_keep],
                boxes_o=b_o[box_keep],
                labels=list(keep_idx.values()),
                scores=[s[k, v] for k, v in keep_idx.items()]
            )
            if self.training:
                result_dict["gt_labels"] = [l[i, pred_cls] for i, pred_cls in enumerate(keep_cls)]

            results.append(result_dict)

        return results

    def forward(self, features, detections, image_shapes, targets=None):
        """
        Arguments:
            features(OrderedDict[Tensor]): Image pyramid with different levels
            detections(list[dict]): Object detections with following keys 
                "boxes": Tensor[N, 4]
                "labels": Tensor[N]
                "scores": Tensor[N]
            image_shapes(List[Tuple[height, width]])
            targets(list[dict]): Interaction targets with the following keys
                "boxes_h": Tensor[N, 4]
                "boxes_o": Tensor[N, 4]
                "object": Tensor[N] Object class index for the object in each pair
                "target": Tensor[N] Target class index for each pair
        Returns:
            results(list[dict]): During evaluation, return dicts of detected interacitons
                "boxes_h": Tensor[M, 4]
                "boxes_o": Tensor[M, 4]
                "labels": list(Tensor) The predicted label indices. A list of length M.
                "scores": list(Tensor) The predcited scores. A list of length M. 
            During training, the classification loss is appended to the end of the list
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"
        detections = self.preprocess(detections, targets)

        box_coords = [detection['boxes'] for detection in detections]
        box_labels = [detection['labels'] for detection in detections]
        box_scores = [detection['scores'] for detection in detections]

        box_features = self.box_roi_pool(features, box_coords, image_shapes)

        box_pair_features, boxes_h, boxes_o, box_pair_labels, box_pair_prior = self.box_pair_head(
            features, box_features,
            box_coords, box_labels, box_scores, targets
        )

        # No valid human-object pairs were formed
        if len(box_pair_features) == 0:
            return None

        interaction_scores = self.box_pair_predictor(box_pair_features, box_pair_prior)

        results = self.postprocess(interaction_scores, boxes_h, boxes_o, box_pair_labels)
        # All human-object pairs have near zero scores
        if len(results) == 0:
            return results

        if self.training:
            results.append(self.compute_interaction_classification_loss(
                interaction_scores, _cat(box_pair_labels)
            ))

        return results

class HOINetworkTransform(transform.GeneralizedRCNNTransform):
    """
    Transformations for input image and target (box pairs)

    Arguments(Positional):
        min_size(int)
        max_size(int)
        image_mean(list[float] or tuple[float])
        image_std(list[float] or tuple[float])

    Refer to torchvision.models.detection for more details
    """
    def __init__(self, *args):
        super().__init__(*args)

    def resize(self, image, target):
        """
        Override method to resize box pairs
        """
        h, w = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        scale_factor = min(
            self.min_size[0] / min_size,
            self.max_size / max_size
        )

        image = nn.functional.interpolate(image[None], 
            scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]
        if target is None:
            return image, target

        target['boxes_h'] = transform.resize_boxes(target['boxes_h'],
            (h, w), image.shape[-2:])
        target['boxes_o'] = transform.resize_boxes(target['boxes_o'],
            (h, w), image.shape[-2:])

        return image, target

    def postprocess(self, results, image_shapes, original_image_sizes):
        if self.training:
            loss = results.pop()

        for pred, im_s, o_im_s in zip(results, image_shapes, original_image_sizes):
            boxes_h, boxes_o = pred['boxes_h'], pred['boxes_o']
            boxes_h = transform.resize_boxes(boxes_h, im_s, o_im_s)
            boxes_o = transform.resize_boxes(boxes_o, im_s, o_im_s)
            pred['boxes_h'], pred['boxes_o'] = boxes_h, boxes_o

        if self.training:
            results.append(loss)

        return results

class GenericHOINetwork(nn.Module):
    """A generic architecture for HOI classification

    Arguments:
        backbone(nn.Module)
        interaction_head(nn.Module)
        transform(nn.Module)
    """
    def __init__(self, backbone, interaction_head, transform):
        super().__init__()
        self.backbone = backbone
        self.interaction_head = interaction_head
        self.transform = transform

    def preprocess(self, images, detections, targets=None):
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)

        for det, o_im_s, im_s in zip(
            detections, original_image_sizes, images.image_sizes
        ):
            boxes = det['boxes']
            boxes = transform.resize_boxes(boxes, o_im_s, im_s)
            det['boxes'] = boxes

        return images, detections, targets, original_image_sizes

    def forward(self, images, detections, targets=None):
        """
        Arguments:
            images(list[Tensor])
            detections(list[dict])
            targets(list[dict])
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images, detections, targets, original_image_sizes = self.preprocess(
                images, detections, targets)

        features = self.backbone(images.tensors)
        results = self.interaction_head(features, detections, 
            images.image_sizes, targets)

        if results is None:
            return results

        return self.transform.postprocess(
            results,
            images.image_sizes,
            original_image_sizes
        )
