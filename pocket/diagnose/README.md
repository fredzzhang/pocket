## __`CLASS`__ pocket.diagnose.ConfusionMatrix

Confusion matrix class, with dimension arranged in _[Prediction, GroundTruth]_

`Parameters:`
* **num_cls**_(int, optional)_: Number of classes in the matrix (default: *__0__*)
* **mode**_(str, optional)_: Evaluation mode, a choice between 'FREQ' and 'MEAN' (default: *__'FREQ'__*)
    * __'FREQ'__: Confusion matrix stores the frequency of each predicted class
    * __'MEAN'__: Confusion matrix stores the mean prediction scores

`Properties:`
* **cmatrix**_(Tensor)_: The confusion matrix
* **mode**_(str)_: The evaluation mode

`Methods`:
* **reset**_()_: Reset the confusion matrix with zeros
* **push**_(out, labels)_: Update the confusion matrix based on output of a network and labels
    * out(Tensor[M, N])
    * labels(Tensor[M, N])
* **show**_()_: Plot the confusion matrix
* **save**_(cache\_dir)_: Save the confusion matrix into specified directory with evaluation mode in the file name
* **load**_(cache\_path)_: Load the confusion matrix from a .pkl file specified in the path
* **merge**_(ind\_range)_: Mmerge certain classes in the matrix
    * **ind_range**_(Tensor[N', 2])_: Starting and end indices of intervals to be merged into one class
* **normalize**_(dim=0)_: Normalize the confusion matrix

`Example:`
```python
>>> import torch
>>> from pocket.diagnose import ConfusionMatrix
>>> # Two samples belong to class 0, while both have higher scores for class 1
>>> # One sample belongs to class 1, and is predicted with a higher score for class 1
>>> # There are no negative samples
>>> output = torch.tensor([[0.1, 0.9], [0.2, 0.7], [0.4, 0.9]])
>>> labels = torch.tensor([[0., 1.], [1., 0.], [1., 0.]])
>>> cm = ConfusionMatrix(2)
>>> cm.push(output, labels)
>>> cm.cmatrix
tensor([[0., 0., 0.],
        [2., 1., 0.]])
>>> cm = ConfusionMatrix(2, 'MEAN')
>>> cm.push(output, labels)
>>> cm.cmatrix
tensor([[0.3000, 0.1000, 0.0000],
        [0.8000, 0.9000, 0.0000]])
```

## __`FUNCTION`__ pocket.diagnose.compute_confusion_matrix

Compute the confusion matrix.

`Parameters:`
* **net**_(Module)_: Nework model
* **labels**_(Tensor[N, C])_: Labels for all data in the loader, where N is the number of samples and C is the number of classes.
NOTE: The order of the labels should be consistent with the order of samples in dataloader
* **dataloader**_(iterable)_: Dataloader with batch data as \[INPUT_1, ..., INPUT_N\], where each input should be a tensor. Otherwise, use {input_transform} to format the batch
* **mode**_(str, optional)_: Evaluation mode, (default: *__'FREQ'__*)
* **device**_(str, optional)_: Primary device used for network forward pass, (default: *__'cpu'__*)
* **multi_gpu**_(bool, optional)_: If True, use all visible GPUs during forward pass (default: *__False__*)
* **cache_dir**_(str, optional)_: Directory to save cache, (default: *__'cache'__*)
* **input_transform**_(callable, optional)_: Transform used to format batch data, (default: *__lambda a: a__*)
* **output_transform**_(callble, optional)_: Transform used to format the collective output, (default: *__lambda a: torch.cat([b for b in a], 0)__*)

`Returns:`
* **cmatrix**_(Tensor[C, C+1])_: Confusion matrix

## __`CLASS`__ pocket.diagnose.RankedScorePlot

Class for ranked score plots

For each class, three types of samples will be plotted  
* **Positves**: Positive samples for current class
* **Type-I Negatives**: Positive samples from other classes but negative for current class
* **Type-II Negatives**: Negative samples for all classes

The vertical axis indicates the classification score for a particular class of 
a sample, while the horizontal axis show its normalized index (between 0 and 1).
Samples will be ranked based on the classification scores. Positives are in 
descending order while type-I and type-II negatives are in ascending order.

`Parameters:`
* **num_classes**_(int)_: Number of classes

`Methods:`
* **fetch_class_data**_(cls)_: Fetch ranked scores for a particular class
* **push**_(pred, labels)_: Update the ranked score plot based on prediction from a network and the corresponding labels
    * pred(Tensor[M, N] or ndarray[M, N])
    * labels(Tensor[M, N) or ndarray[M, N])
* **show**_(cls)_: Show the ranked score plot for selected class
* **save**_(cache\_dir)_: Save all ranked score plots as _.png_ files

`Example`:
```python
>>> import torch
>>> from pocket.diagnose import RankedScorePlot
>>> pred = torch.tensor([[0.1, 0.9], [0.2, 0.7], [0.4, 0.9]])
>>> labels = torch.tensor([[0., 1.], [1., 0.], [1., 0.]])a
>>> rsp = RankedScorePlot(2)
>>> rsp.push(pred, labels)
>>> # Class 0 has positive samples with index 1 and 2
>>> # Type-I negative with index 0, and no Type-II negatives
>>> rsp.fetch_class_data(0)
[array([[1., 0.2],
        [2., 0.4]]),
 array([[0., 0.1]]),
 array([], shape=(0, 2), dtype=float64)]
>>> # Class 1 has positve sample with index 0
>>> # Type-I negative with index 1 and 2, and no Type-II negatives
>>> rsp.fetch_class_data(1)
[array([[0., 0.9]]),
 array([[1., 0.7],
        [2., 0.9]]),
 array([], shape=(0, 2), dtype=float64)]
```

## __`FUNCTION`__ pocket.diagnose.compute_ranked_scores

Compute the ranked score plots

`Parameters:`
* **net**_(Module)_: Nework model
* **labels**_(Tensor[N, C])_: Labels for all data in the loader, where N is the number of samples and C is the number of classes.
NOTE: The order of the labels should be consistent with the order of samples in dataloader
* **dataloader**_(iterable)_: Dataloader with batch data as \[INPUT_1, ..., INPUT_N\], where each input should be a tensor. Otherwise, use {input_transform} to format the batch
* **device**_(str, optional)_: Primary device used for network forward pass, (default: *__'cpu'__*)
* **multi_gpu**_(bool, optional)_: If True, use all visible GPUs during forward pass (default: *__False__*)
* **cache_dir**_(str, optional)_: Directory to save cache, (default: *__'cache'__*)
* **input_transform**_(callable, optional)_: Transform used to format batch data, (default: *__lambda a: a__*)
* **output_transform**_(callble, optional)_: Transform used to format the collective output, (default: *__lambda a: torch.cat([b for b in a], 0)__*)

`Returns:`
* **ranked_scores**_(list[ndarray[M, 2]])_: Ranked scores for all class. Each class has an ndarray of M samples, that stores the sample index and classification score.

## __`CLASS`__ pocket.diagnose.LearningCurveVisualizer

Class for learning curve visualization.

Assume the training log is a _.txt_ file, and written in the following format
```txt
1. Step: {A}, training loss: {B}, validation loss: {C}
2. Step: {A}, training loss: {B}
3. Learning rate: {A}, step: {B}
```
Any other lines with a different format will be disregarded

`Parameters:`
* **src_path**_(str)_: Path of the source .txt file

`Properties:`
* **step**_(ndarray[N])_: Step numbers at which losses are recorded
* **loss**_(ndarray[N])_: Training and validation losses

`Methods:`:
* **show**_(scale='linear', weight=0)_: Plot the learning curve
    * **scale**_(str)_: _'linear'_, _'log'_
    * **weight**_(str)_: A smoothing factor

## __`CLASS`__ pocket.diagnose.ParamVisualizer

Class for layer parameter visualization

`Parameters:`
* **params**_(Tensor or ndarray)_: Weights or biases data

`Properties:`
* **params**_(ndarray)_: Parameter in ndarray

`Methods:`
* **show**_()_: Plot the parameters

## __`CLASS`__ pocket.diagnose.NetVisualizer

Class for network parameter visualization. Child class of _dict_. Load the state
dict of a PyTorch module and construct a _pocket.diagnose.ParamVisualizer_ for 
each parameter block, with same keys as in the state dict itself. 
Method *\_\_len\_\_()* return the number of parameter blocks

`Parameters:`
* **state_dict**_(OrderedDict, optional)_: State dict of a PyTorch model (defaultL __*None*__)
* **pt_path**_(str, optional)_: Path of a PyTorch model, typically ends with .pt (default: __*None*__)
* **ckpt_path**_(str, optional)_: Path of a checkpoint file, with key *model_state_dict* (default: __*None*__)

`Example:`
```python
>>> from torch import nn
>>> from pocket.diagnose import NetVisualizer
>>> net = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 20))
>>> v = NetVisualizer(net.state_dict())
>>> len(v)
4
>>> v.keys()
dict_keys(['0.weight', '0.bias', '2.weight', '2.bias'])
>>> v['0.weight'].show()
```

## __`CLASS`__ pocket.diagnose.ImageVisualizer

Visualizer for an image database. Method *\_\_len\_\_()* return the number of images

`Parameters:`
* **imdir**_(str)_: Directory of source images
* **labels**_(ndarray[N, M], optional)_: One-hot encoded labels for all N images (default: __*None*__)
* **descpt**_(ndarray[M], optional)_: String descriptions for each of M classes (default: __*None*__)

`Methods`:
* **show**_(i)_: Display the image corresponding to index i and associated labels
* **image_path**_(i)_: Show the path of image corresponding to index i

## __`CLASS`__ pocket.diagnose.BoxVisualizer

Visualizer for bounding-box-based dataset. Child class of _pocket.diagnose.ImageVisualizer_.

`Parameters:`
* **imdir**_(str)_: Directory of source images
* **boxdir**_(str)_: Directory of bounding box source files, presumably _.txt_ files

`Methods`:
* **show**_(i)_: to be implemented...

## __`CLASS`__ pocket.diagnose.HICODetVisualizer

Visualizer for HICODet dataset

`Paramters:`
* **imdir**_(str)_: Directory of source images
* **db**_(DataDict)_: A data dict with following keys
    * **db.image_id**_(ndarray[N])_
    * **db.box_h**_(ndarray[N, 4])_
    * **db.box_o**_(ndarray[N, 4])_
    * **db.scores**_(ndarray[N, 2])_
    * **db.labels**_(csr_matrix[N, 600])_
    * **db.per_image**_(ndarray[M, 2])_: Starting index and number of box pairs in an image
* **hoi_list**_(ndarray[N, 2])_: HOI list \[{VERB}, {OBJECT}\]
* **mode**_(str, optional)_: Index mode, a choice between 'IMAGE' and 'BOX' (default: *__'IMAGE'__*)

`Properties:`
* **mode**_(str)_: Index mode

`Methods`:
* **show**_(i)_: Display bounding box pairs based on index mode
