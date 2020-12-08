### __`CLASS`__ pocket.utils.NumericalMeter(_maxlen: Optional[int] = None_)

Meter class with numerals as elements. Supported element types are int and float.

`Parameters:`
* **maxlen**: Maximum number of elements to be stored in the meter. Leave the argument as None to keep the size flexible. Default is None.

`Methods:`
* \_\_len\_\_() -> _int_: Return the number of elements
* \_\_getitem\_\_(_i: int_) -> _Union[int, float]_: Return the element corresponding to the index
* append(_x: Union[int, float]_) -> _None_: Add an element
* sum() -> _Union[int, float]_: Return the sum of all elements
* mean() -> _float_: Return the mean of all elements
* max() -> _Union[int, float]_: Return the largest element
* min() -> _Union[int, float]_: Return the smallest element

`Examples:`
```python
>>> from pocket.utils import NumericalMeter
>>> m = NumericalMeter(2)
>>> m.append(5); m.append(2.5)
>>> m.sum()
7.5
>>> m.append(3)
>>> # Due to the specified max length, the first element has been removed
>>> m.sum()
5.5
```

---

### __`CLASS`__ pocket.utils.HandyTimer(_maxlen: Optional[int] = None_)

A timer class that tracks a sequence of time. The class is inherited from _NumericalMeter_ and is implemented with *\_\_enter\_\_()* and *\_\_exit\_\_()* methods.

`Parameters:`
* **maxlen**: Maximum number of elements to be stored in the meter. Leave the argument as None to keep the size flexible. Default is None.

`Methods:`
* \_\_len\_\_() -> _int_: Return the number of elements
* \_\_getitem\_\_(_i: int_) -> _Union[int, float]_: Return the element corresponding to the index
* append(_x: Union[int, float]_) -> _None_: Add an element
* sum() -> _Union[int, float]_: Return the sum of all elements
* mean() -> _float_: Return the mean of all elements
* max() -> _Union[int, float]_: Return the largest element
* min() -> _Union[int, float]_: Return the smallest element

`Examples:`
```python
>>> import time
>>> from pocket.utils import HandyTimer
>>> t = HandyTimer()
>>> with t:
>>>     time.sleep(3)
>>> with t:
>>>     time.sleep(4)
>>> len(t)
2
>>> t.mean()
3.5
```

---

### __`CLASS`__ pocket.utils.AveragePrecisionMeter(_num_gt: Optional[Iterable] = None, algorithm: str = "AUC", chunksize: int = -1, output: Optional[Tensor] = None, labels: Optional[Tensor] = None_)

Meter to compute average precision

`Parameters:`
* **num_gt**: (K,) Number of ground truth instances for each class. When left as None, all positives are assumed to have been included in the collected results. As a result, full recall is guaranteed when the lowest scoring example is accounted for.
* **algorithm**: AP evaluation algorithm
    * _11P_: 11-point interpolation algorithm prior to voc2010
    * _INT_: Interpolation algorithm with all points used in voc2010
    * _AUC_: Precisely as the area under precision-recall curve
* **chunksize**: The approximate size the given iterable will be split into for each worker. Use -1 to make the argument adaptive to iterable size and number of workers
* **output**: (N, K) Output scores with N examples and K classes. Default is None
* **labels**: (N, K) Binary labels. Default is None

`Instance Methods:`
* append(_output: Tensor, labels: Tensor_) -> _None_: Add new results to the meter
    * **output**: (N, K) Output scores with N examples and K classes
    * **labels**: (N, K) Binary labels
* reset(_keep_old: bool = False_) -> _None_: Clear saved statistics and reset the meter
    * **keep_old**: If True, only clear the newly collected statistics since last evaluation
* eval() -> _Tensor_: Evaluate the average precision based on collected statistics

`Class Methods:`
* compute_ap(*output: Tensor, labels: Tensor, num_gt: Optional[Tensor] = None, algorithm: str = 'AUC', chunksize: int = -1*) -> Tensor: Compute AP for all classes, assuming the numbers of examples are identical across all classes
    * **output**: (N, K) Output scores for N examples and K classes
    * **labels**: (N, K) Binary labels for each example and class
    * **num_gt**: (K,) Number of ground truth instances for each class. When left as None, all ground truth instances are assumed to have been retrieved (100% recall)
    * **algorithm**: AP evaluation algorithm, same as in the parameter list of the class
    * **chunksize**: The approximate size the given iterable will be split into for each worker. Use -1 to make the argument adaptive to iterable size and number of workers

`Static Methods:`
* compute_per_class_ap_as_auc(*tuple_: Tuple[Tensor, Tensor])*) -> _Tensor_: Compute AP as area under curve (AUC)
    * **tuple_[0]**: (N,) Precision values
    * **tuple_[1]**: (N,) Recall values
* compute_per_class_ap_with_interpolation(*tuple_: Tuple[Tensor, Tensor]*) -> _Tensor_: Compute AP with interpolation at all points
    * **tuple_[0]**: (N,) Precision values
    * **tuple_[1]**: (N,) Recall values
* compute_per_class_ap_with_11_point_interpolation(*tuple_: Tuple[Tensor, Tensor]*) -> _Tensor_: Compute AP with interpolation at 11 points with recall from 0 to 1
    * **tuple_[0]**: (N,) Precision values
    * **tuple_[1]**: (N,) Recall values
* compute_precision_and_recall(_output: Tensor, labels: Tensor, num_gt: Optional[Tensor] = None, eps: float = 1e-8_) -> _Tuple[Tensor, Tensor]_: Compute precisions and recalls
    * **output**: (N, K) Output scores with N examples and K classes
    * **labels**: (N, K) Binary labels
    * **num_gt**: (K,) Number of ground truth instances for each class. When left as None, all positives are assumed to have been included in the collected results. As a result, full recall is guaranteed when the lowest scoring example is accounted for.
    * **eps**: A small constant to avoid division by zero

`Examples:`
```python
>>> """(1) Evalute AP using provided output scores and labels"""
>>> # Given output(tensor[N, K]) and labels(tensor[N, K])
>>> meter = pocket.utils.AveragePrecisionMeter(output=output, labels=labels)
>>> ap = meter.eval(); map_ = ap.mean()
>>> """(2) Collect results on the fly and evaluate AP"""
>>> meter.reset()
>>> # Compute output(tensor[N, K]) during forward pass
>>> meter.append(output, labels)
>>> ap = meter.eval(); map_ = ap.mean()
```

---

### __`CLASS`__ pocket.utils.DetectionAPMeter(_num_cls: int, num_gt: Optional[Tensor] = None, algorithm: str = 'AUC', nproc: int = 20, output: Optional[List[Tensor]] = None, labels: Optional[List[Tensor]] = None_)

A variant of AP meter, where network outputs are assumed to be class-specific. Different classes could potentially have different number of samples.

`Parameters:`
* **num_cls**: Number of target classes
* **num_gt**: (K,) Number of ground truth instances for each class. When left as None, all positives are assumed to have been included in the collected results. As a result, full recall is guaranteed when the lowest scoring example is accounted for.
* **algorithm**: AP evaluation algorithm
    * _11P_: 11-point interpolation algorithm prior to voc2010
    * _INT_: Interpolation algorithm with all points used in voc2010
    * _AUC_: Precisely as the area under precision-recall curve
* **nproc**: The number of processes used to compute mAP. Default: 20
* **output**: A collection of output scores for K classes. Default is None
* **labels**: Binary labels. Default is None

`Instance Methods:`
* append(_output: Tensor, prediction: Tensor, labels: Tensor_) -> _None_: Add new results to the meter
    * **output**: (N,) Output scores for each sample
    * **prediction**: (N,) Predicted classes 0~(K-1)
    * **labels**: (N,) Binary labels for the predicted classes
* reset(_keep_old: bool = False_) -> _None_: Clear saved statistics and reset the meter
    * **keep_old**: If True, only clear the newly collected statistics since last evaluation
* eval() -> _Tensor_: Evaluate the average precision based on collected statistics

`Class Methods:`
* compute_ap(*output: List[Tensor], labels: List[Tensor], num_gt: Iterable, nproc: int, algorithm: str = 'AUC'*) -> Tuple[Tensor, Tensor]: Compute average precision under the detection setting. Only scores of the predicted classes are retained for each sample. As a result, different classes could have different number of examples.
    * **output**: (K,) Output scores for K classes
    * **labels**: (K,) Binary labels for K classes
    * **num_gt**: Number of ground truth instances for each class
    * **nproc**: Number of processes to be used for computation
    * **algorithm**: AP evaluation algorithm, same as in the parameter list of the class
* compute_ap_for_each(*tuple_: Tuple[int, int, Tensor, Tensor, Callable]*) -> Tuple[Tensor, Tensor]: Compute AP for one class
    * **tuple[0]**: Index of the class. This is used to make error message more readable
    * **tuple[1]**: Number of ground truth instances.  When left as None, all ground truth instances are assumed to have been retrieved (100% recall)
    * **tuple[2]**: Output scores
    * **tuple[3]**: Binary labels
    * **tuple[4]**: Function handle of the AP evaluation algorithm

`Static Methods:`
* compute_pr_for_each(_output: Tensor, labels: Tensor, num_gt: Optional[Union[int, float]] = None, eps: float = 1e-8_) -> _Tuple[Tensor, Tensor]_: Compute precision and recall for one class
    * **output**: (N,) Output scores for each sample
    * **labels**: (N,) Binary labels for each sample
    * **num_gt**: Number of ground truth instances. When left as None, all ground truth instances are assumed to have been retrieved (100% recall)
    * **eps**: A small constant to avoid division by zero

`Examples:`
```python
>>> """(1) Evalute AP using provided output scores and labels"""
>>> # Given output(List[tensor]) and labels(List[tensor])
>>> meter = pocket.utils.DetectionAPMeter(num_cls, output=output, labels=labels)
>>> ap = meter.eval(); map_ = ap.mean()
>>> """(2) Collect results on the fly and evaluate AP"""
>>> meter.reset()
>>> # Get class-specific predictions. The following is an example
>>> # Assume output(tensor[N, K]) and target(tensor[N]) are given
>>> pred = output.argmax(1)
>>> scores = output.max(1)
>>> meter.append(scores, pred, pred==target)
>>> ap = meter.eval(); map_ = ap.mean()
```

---

### __`CLASS`__ pocket.utils.SyncedNumericalMeter(_maxlen: Optional[int] = None_)

Numerical meter synchronized across subprocesses. By default, it is assumed that NCCL is used as the communication backend. Communication amongst subprocesses can only be done with CUDA tensors, not CPU tensors. Make sure to intialise default process group before instantiating the meter by

```python
torch.distributed.init_process_group(backbone="nccl", ...)
```

`Parameters:`
* **maxlen**: Maximum number of elements to be stored in the meter. Leave the argument as None to keep the size flexible. Default is None.

`Methods:`
* \_\_len\_\_() -> _int_: Return the number of elements
* \_\_getitem\_\_(_i: int_) -> _Union[int, float]_: Return the element corresponding to the index
* append(_x: Union[int, float]_) -> _None_: Add an element
* sum(_local: bool = False_) -> _Union[int, float]_: Return the sum of all elements
    * **local**: If True, return the local stats. Otherwise, aggregate over all subprocesses
* mean(_local: bool = False_) -> _float_: Return the mean of all elements
    * **local**: If True, return the local stats. Otherwise, aggregate over all subprocesses
* max(_local: bool = False_) -> _Union[int, float]_: Return the largest element
    * **local**: If True, return the local stats. Otherwise, aggregate over all subprocesses
* min(_local: bool = False_) -> _Union[int, float]_: Return the smallest element
    * **local**: If True, return the local stats. Otherwise, aggregate over all subprocesses

---

### __`FUNCTION`__ pocket.utils.all_gather(_data: Any_) -> _List[Any]_

Gather arbitrary picklable data (not necessarily tensors) across all subprocesses. This implementation converts pickable data into 1-d byte tensors, and runs `torch.distributed.all_gather` to collate the results. The code is taken from
https://github.com/pytorch/vision/blob/master/references/detection/utils.py.

`Parameters:`
* __data__: Any pickable data

`Returns:`
* __data_list__: List of data gathered from all subprocesses

---

### __`CLASS`__ pocket.utils.HTMLTable(_num_cols: int, *args: Iterable_)

Base class for generation of HTML tables. This class generates HTML code to display the given iterables with specified number of columns. Assume there are N iterables and M columns, the generated table has the following format
```python
args[0][0],      args[0][1],      ...,    args[0][M],
args[1][0],      args[1][1],      ...,    args[1][M],
...,
args[N][0],      args[N][1],      ...,    args[N][M],
args[0][M+1],    args[0][M+2],    ...,    args[0][2*M],
...,
```

`Parameters:`
* **num_cols**: Number of columns in the table
* **args**: Tuple of iterables to be displayed in the table

`Methods:`
* \_\_call\_\_(_fname: Optional[str] = None, title: Optional[str] = None_) -> None: Generate HTML code
    * __fname__: Name (or path) of the output HTML file. If left as None, _table.html_ is used
    * __title__: Name of the html page. If left as None, _Table_ is used

---

### __`CLASS`__ pocket.utils.ImageHTMLTable(_num_cols: int, image_dir: str, parser: Optional[Callable] = None, sorter: Optional[Callable] = None, extension: str = None, **kwargs_)

HTML table of images with captions. By default the image file name will be used as caption.

`Parameters:`
* **num_cols**: Number of columns in the table
* **image_dir**: Directory where images are located
* **parser**: A parser that formats image names into captions
* **sorter**: A function that sorts image names into desired order
* **extension**: Format of image files to be collected
* **kwargs**: Attributes of HTML \<img> tag. e.g. {"width": "75%"}

---

### __`FUNCTION`__ pocket.utils.draw_boxes(_image: PIL.Image, boxes: Union[ndarray, Tensor, list], **kwargs_) -> _None_

Draw bounding boxes onto a PIL image

`Parameters:`
* **image**: Input image in the format PIL.Image
* **boxes**: Bounding boxes in the format [x1, y1, x2, y2]
* **kwargs**: Parameters for _PIL.ImageDraw.Draw.rectangle_

`Examples:`
```python
>>> from PIL import Image
>>> from pocket.utils import draw_boxes
>>> image = Image.new('RGB', (200, 200))
>>> draw_boxes(image, [[30, 30, 80, 80], [50, 50, 150, 150]])
>>> image.show()
```

---

### __`FUNCTION`__ pocket.utils.draw_box_pairs(_image: PIL.Image, boxes_1: Union[ndarray, Tensor, list], boxes_2: Union[ndarray, Tensor, list], width: int = 1_) -> _None_

Draw bounding box pairs onto a PIL image. Boxes corresponding to argument <boxes_1> are drawn in blue, and green for the other group. Corresponding box pairs will be joined by a red line connecting the centres of the boxes

`Parameters:`
* **image**: Input image in the format PIL.Image
* **boxes_1**: Bounding boxes in the format [x1, y1, x2, y2]
* **boxes_2**: Bounding boxes in the format [x1, y1, x2, y2]
* **width**: Width of the boxes

`Examples:`
```python
>>> from PIL import Image
>>> from pocket.utils import draw_box_pairs
>>> image = Image.new('RGB', (120, 120))
>>> boxes_1 = [[10, 10, 40, 40], [30, 30, 100, 100]]
>>> boxes_2 = [[10, 60, 40, 90], [80, 80, 110, 110]]
>>> draw_box_pairs(image, boxes_1, boxes_2)
>>> image.show()
```

---

### __`FUNCTION`__ pocket.utils.draw_dashed_line(_image: PIL.Image, xy: Union[ndarray, Tensor, list], length: int = 5, **kwargs_) -> None

Draw dashed lines onto a PIL image

`Parameters:`
* **image**: Input image in the format PIL.Image
* **xy**: Coordinates of the starting and ending point in the format [x1, y1, x2, y2]
* **length**: Length of each line segment
* **kwargs**: Parameters for _PIL.ImageDraw.Draw.line_

`Examples:`
```python
>>> from PIL import Image
>>> from pocket.utils import draw_boxes
>>> image = Image.new('RGB', (200, 200))
>>> draw_boxes(image, [30, 30, 180, 180])
>>> image.show()
```

---

### __`FUNCTION`__ pocket.utils.draw_dashed_rectangle(_image: PIL.Image, xy: Union[ndarray, Tensor, list], **kwargs_) -> _None_

Draw rectangles in dashed lines onto a PIL image

`Parameters:`
* **image**: Input image in the format PIL.Image
* **boxes**: Bounding boxes in the format [x1, y1, x2, y2]
* **kwargs**: Parameters for _PIL.ImageDraw.Draw.rectangle_

---

### __`CLASS`__ pocket.utils.BoxAssociation(*min_iou: float, encoding: str = 'coord'*)

Associate detection boxes with ground truth boxes

`Parameters:`
* **min_iou**: The minimum intersection over union to identify a positive
* **encoding**: Encodings of the bounding boxes. Choose between 'coord' and 'pixel'. Refer to *pocket.ops.box_iou* for details

`Methods:`
* \_\_call\_\_(*gt_boxes: FloatTensor, det_boxes: FloatTensor, scores: FloatTensor*) -> FloatTensor: Compute binary labels for each detected box
    * **gt_boxes**: (N, 4) Ground truth bounding boxes in (x1, y1, x2, y2) format
    * **det_boxes**: (M, 4) Detected bounding boxes in (x1, y1, x2, y2) format
    * **scores**: (M,) Confidence scores for each detection
    * Returns: (M,) Binary labels for each detection

`Properties:`
* max_iou -> FloatTensor: Return the largest IoU with any ground truth instances for each detection
* max_idx -> LongTensor: Return the index of ground truth instance each detection is associated with

`Examples:`
```python
>>> import torch
>>> from pocket.utils import BoxAssociation
>>> associate = BoxAssociation(0.5)
>>> gt_boxes = torch.tensor([[10., 10., 50., 50.], [60., 60., 100., 100.]])
>>> det_boxes = torch.tensor([
... [9.5, 11.2, 49.5, 50.8, 0.99],      # A match for the first G.T. box
... [10.9, 12.3, 52.4, 53.9, 0.45],     # Another match for the first G.T. box but with a lower score
... [45.3, 52.7, 89.5, 78.2, 0.04],     # Not a match
... [52.7, 59.4, 97.2, 102.8, 0.54],    # A match for the second G.T. box
])
>>> # Compute the binary labels for each detection
>>> associate(gt_boxes, det_boxes[:, :4], det_boxes[:, -1])
tensor([1., 0., 0., 1.])
>>> # Show the index of ground truth instance each detection is associated with
>>> associate.max_idx
tensor([0, 0, 1, 1])
>>> # Show the IoU with the associated ground truth instance
>>> associate.max_iou
tensor([0.9281, 0.7958, 0.2451, 0.7282])
```    

---
### __`CLASS`__ pocket.utils.BoxPairAssociation(*min_iou: float, encoding: str = 'coord'*)

Associate detection box pairs with ground truth box pairs. The intersection over union is computed between the corresponding boxes in the pair. The final IoU is taken as the minimum between the two.

`Parameters:`
* **min_iou**: The minimum intersection over union to identify a positive
* **encoding**: Encodings of the bounding boxes. Choose between 'coord' and 'pixel'. Refer to *pocket.ops.box_iou* for details

`Methods:`
* \_\_call\_\_(*gt_boxes: Tuple[FloatTensor, FloatTensor], det_boxes: Tuple[FloatTensor, FloatTensor], scores: FloatTensor*) -> FloatTensor: Compute binary labels for each detected box
    * **gt_boxes**: Ground truth box pairs in a 2-tuple
    * **det_boxes**: Detection box pairs in a 2-tuple
    * **scores**: (M,) Confidence scores for each detected box pair
    * Returns: (M,) Binary labels for each detected box pair
    
`Properties:`
* max_iou -> FloatTensor: Return the largest IoU with any ground truth instances for each detection
* max_idx -> LongTensor: Return the index of ground truth instance each detection is associated with