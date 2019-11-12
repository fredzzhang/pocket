## __`CLASS`__ pocket.data.DataDict
    
Data dictionary class. This class allows setting and accessing dict keys as class
attributes and provides save/load utilities using pickle as backend. Initialize 
the class with a python dict. By default, an empty dict is applied.  

`Parameters:`
* **data_dict**_(dict, optional)_: A python dict (default: __*{}*__)

`Example:`
```python
>>> from pocket.data import DataDict
>>> person = DataDict()
>>> person.is_empty()
True
>>> person.age = 15
>>> person.sex = 'male'
>>> person.save('./person.pkl', 'w')
```

`Methods:`
* **save**_(path)_: Save into a .pkl file as a standard python dict  
* **load**_(path)_: Load a python dict from a .pkl file as data dict  
* **is_empty**_()_: Return true if the dict has any keys, false otherwise  

## __`CLASS`__ pocket.data.ImageDataset

Base class for image dataset. By default, *\_\_len\_\_()* returns the number of
images and *\_\_getitem\_\_()* fetches an image as numpy array in _[H, W, C]_ 
format. For string representations, *\_\_str\_\_()* returns the dataset and subset
names, and *\_\_repr\_\_()* returns instantiation arguments

`Parameters:`
* **subset**_(str)_: Subset name  
* **cfg**_(CfgNode)_: Configuration class with following attrubutes
    * **cfg.DATASET.NAME**_(str)_: Name of the dataset  
    * **cfg.DATASET.{subset}.IMAGE_DIR**_(str)_: Directory of images for specified subset

`Properties:`
* **num_images**_(int)_: number of images  
* **num_classes**_(int)_: number of classes

`Methods:`
* **construct_image_paths**_(src\_dir)_: Returns a list of paths for all images with extension 
_'.jpg'_ or _'.png'_ in the specified directory
* **flip_images**_()_: Allows accessing horizontally flipped images using original 
index plus total number of images
* **image_path**_(index)_: Return the path of image corresponding to the index
* **fetch_image**_(index)_: Return the image as ndarray in _[H, W, C]_ format given the index

## __`CLASS`__ pocket.data.DetectionDataset

Base class for bounding-box detection dataset. Child class of _pocket.data.ImageDataset_.
By default, the ground truth and detection bouding boxes are assumed to have been
written in .txt files, with the directory specified in argument list.

`Parameters:`
* **subset**_(str)_: Subset name
* **cfg**_(CfgNode)_: Configuration class with following attrubutes
    * **cfg.DATASET.NAME**_(str)_: Name of the dataset  
    * **cfg.DATASET.SOURCE_FILE_EXTENSION**_(str)_: Source file extension
    * **cfg.DATASET.PRINT_INTERVAL**_(int)_: Number of steps between progress updates
    during database construction 
    * **cfg.DATASET.{subset}.IMAGE_DIR**_(str)_: Directory of images for specified subset
    * **cfg.DATASET.{subset}.GT_DIR**_(str)_: Directory for ground truth annotation files
    * **cfg.DATASET.{subset}.GT_CACHE_PATH**_(str)_: Cache path for ground truth database
    * **cfg.DATASET.{subset}.DETECTION_DIR**_(str)_*: Directory for detection result files
    * **cfg.DATASET.{subset}.DETECTION_CACHE_PATH**_(str)_*: Cache path for detection database

`Properties:`
* **gtdb**_(DataDict)_: a user-defined ground truth bounding box database as per *Detdb._construct_gtdb*
* **detdb**_(DataDict)_: a user-defined detection bounding box database as per *Detdb._construct_detdb*

`Methods:`
* **construct_src_path**_(src\_dir, ext='.txt')_: Return a list of paths for all files with the
specified extension in the directory
* **fg_obj**_(index)_: Return the foreground object classes in the image given the index

## __`CLASS`__ pocket.data.HICODet

Dataset class for HICO-DET, a human-object interaction dataset with bounding box
annotations. Child class of _pocket.data.DetectionDataset_. Method *\_\_len\_\_()*
and *\_\_getitem\_\_()* are intentionally disabled for this class due to unclear
data fetch mode  

`Parameters:`
* *subset(str)*: Subset name, a choice between 'TRAIN' and 'TEST'  
* **cfg**_(CfgNode)_: Configuration class with following attrubutes
    * **cfg.DATASET.NAME**_(str)_: Name of the dataset
    * **cfg.DATASET.METADIR**_(str)_: Directory for metadata
    * **cfg.DATASET.NUM_CLASSES**_(int)_: Number of classes in the dataset
    * **cfg.DATASET.MIN_IOU**_(float)_: Minimum intersection over union
    * **cfg.DATASET.SOURCE_FILE_EXTENSION**_(str)_: Source file extension
    * **cfg.DATASET.PRINT_INTERVAL**_(int)_: Number of steps between progress updates
    during database construction 
    * **cfg.DATASET.{subset}.IMAGE_DIR**_(str)_: Directory of images for specified subset
    * **cfg.DATASET.{subset}.GT_DIR**_(str)_: Directory for ground truth annotation files
    * **cfg.DATASET.{subset}.GT_CACHE_PATH**_(str)_: Cache path for ground truth database
    * **cfg.DATASET.{subset}.DETECTION_DIR**_(str)_*: Directory for detection result files
    * **cfg.DATASET.{subset}.DETECTION_CACHE_PATH**_(str)_*: Cache path for detection database
    * **cfg.DATASET.{subset}.NUM_ANNO**_(int)_: Number of annotated box pairs in the subset
    * **cfg.DATASET.{subset}.MAX_NUM_DETECTION**_(int)_: Number of detection box pairs

`Properties:`
* **gtdb**_(DataDict)_: Data dict with following keys
    * **obj_pool**_(list[ndarray])_: A list of 80 arrays, with each containing indices of box pairs involving the corresponding object class
    * **hoi_pool**_(list[ndarray])_: A list of 600 ndarray, with each containing indices of box pairs involving the correponding HOI class
    * **image_id**_(ndarray[N])_: Index of image for box pairs
    * **box_h**_(ndarray[N,4])_: Human bounding box coordinates in \[x1, y1, x2, y2\]
    * **box_o**_(ndarray[N,4])_: Object bounding box coordinates in the same format
    * **scores**_(csr\_matrix[N,81])_: Object detection scores for each box pair
    * **labels**_(ndarray[N,600])_: HOI labels for each box pair
    * **per_image**_(ndarray[M,2])_: Starting index of box pairs and total number of box pairs in an image
* **detdb**_(DataDict)_: Data dict with following keys
    * **neg_pool**_(ndarray)_: Indices of box pairs not associated with ground truth
    * **fg_obj**_(ndarray[N,80])_: Object categories for each box pair
    * **scores**_(ndarray[N,81])_: Object detection scores for each box pair
    * **{KEYS}**: The rest are the same as *gtdb*
* **hoi_list**_(ndarray[600])_: Name of human-object interaction (HOI) class 
* **obj_list**_(ndarray[80])_: Name of object categories
* **hoi_to_obj**_(ndarray[600])_: One-to-one mapping from HOIs to objects  
* **obj_to_hoi**_(ndarray[80,600])_: A one-hot encoded mapping from objects to HOIs  
* **empty_files**_(ndarray[M])_: Indices of images without visible HOIs  

## __`CLASS`__ pocket.data.HICODetTorch

PyTorch interface for HICODet dataset. Child class of _pocket.data.HICODet_. 
Method *\_\_len\_\_()* and *\_\_getitem\_\_()* will return different results upon different fetch modes  

`Parameters:`
* **subset**_(str)_: Subset name, a choice between 'TRAIN' and 'TEST'  
* **cfg**_(CfgNode)_: configuration class with following attrubutes
    * ...__Same as Parent Class__
* **mode**_(str, optional)_: Data fetch mode, consisting of three parts: __{DATABASE}\_{SCOPE}\_{BATCH}__ (default: __*'A_I_BL'*__)

    * DATABASE:
        * 'G': Ground truth database
        * 'D': Detection database
        * 'A': Combined database of G.T. and detection
    * SCOPE:
        * 'I': Index refers to an image 
        * 'B': Index refers to a box pair
    * BATCH(str):
        * 'I': Image (C, H, W)
        * 'B': Human and object bounding box ((N, 4), (N, 4))
        * 'S': Detection scores (N, 81)
        * 'L': HOI labels  (N, 600)
        * ..._Customized Options_

`Properties`:
* **mode**_(str)_: Data fetch mode, implemented with setter

`Methods`:
* **fetch_image**_(i)_: Return the image [H, W, C] corresponding to index i based on fetch mode  
* **fetch_box_paris**_(i)_: Return the coordinates ([M, 4], [M, 4]) of human-object box pair(s) corresponding to index i
* **fetch_det_scores**_(i)_: Return detection scores [M, 81] corresponding to index i
* **fetch_labels**_(i)_: Return the labels [M, 600] corresponding to index i
* **fetch_box_spatial_features**_(i)_: Return hand-crafted box pair features [M, 12] corresponding to index i

## __`CLASS`__ pocket.data.StratifiedBatchSampler

Stratified sampler for a minibatch. As a convention from *torch.utils.data.Sampler*,
*\_\_len\_\_()* returns the total number of samples and *\_\_iter\_\_()* returns the iterator of the sampler.  

Given M strata/classes, form minibatches by taking an equal number of samples from
N classes, where N classes are taken sequentially from the entirety of M classes.
When specified, samples from a negative class can be appended in the batch

When sampling from a specific class, samples are taken ramdomly without replacement,
util the class runs out of samples and gets renewed.

`Parameters:`
* **strata**_(list[Tensor])_: A list tensors, each of which contains the indices for one stratum  
* **num_strata_each**_(int)_: Number of strata to sample from in a single minibatch
* **samples_per_stratum**_(int)_: Number of samples taken from one stratum in single minibatch
* **num_batch**_(int)_: Number of minibatches in an epoch
* **negative_pool**_(ndarray, optional)_: A pool of indices for negative samples (default: __*None*__)
* **num_negatives**_(int, optional)_: Number of negative samples in a single minibatch (default: __*0*__)

`Example:`
```python
>>> import torch
>>> from pocket.data import StratifiedBatchSampler
>>> # Generate strata indices
>>> # Two strata are created in the following
>>> strata = [torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])]
>>> # Generate negative sample indices
>>> negatives = torch.tensor([6, 7, 8, 9])
>>> # Construct stratified batch sampler
>>> # Here each batch will take 2 samples ranomly from a selected stratum
>>> # and 3 samples randomly taken from the negative pool. The selection of
>>> # stratum is sequential, starting from the first stratum
>>> a = StratifiedBatchSampler(strata, 1, 2, 5, negatives, 3)
>>> for batch in a:
...     print(batch)
[2, 1, 7, 8, 6]
[4, 5, 9, 7, 8]
[0, 1, 9, 6, 9]
[3, 5, 7, 6, 8]
[0, 2, 7, 8, 9]
```

## __`CLASS`__ pocket.data.StratifiedSampler

__`Deprecated`__

Samples a specified number of elements from each stratum for a number of iterations.
As a convention from *torch.utils.data.Sampler*, *\_\_len\_\_()* returns the total number of samples and *\_\_iter\_\_()* returns the iterator of the sampler  

`Parameters:`
* *strata(list)*: (M,) a list of iterables, each of which contains the indices for one stratum  
* *num_iter(int)*: number of iterations of sampling  
* *samples_per_stratum(Tensor or ndarray)*: number of samples taken from a stratum per iteration  

## __`CLASS`__ pocket.data.MultiLabelStratifiedSampler

__`Deprecated`__

Samples a specified number of elements from each stratum for a number of iterations. Samples could potentially have multiple labels, meaning that certain strata share certain samples.  

The sampler suppresses those classes/strata that tend to co-occur, by skipping stratum that has more training samples than the iteration counter. *\_\_iter\_\_()* returns the iterator of the sampler and *\_\_len\_\_()* return current sample count.  

`Parameters:`
* *strata(list)*: (M,) a list of iterables, each of which contains the indices for one stratum  
* *labels(Tensor)*: (N, M) labels of all samples
* *samples_per_class(int)*: number of samples taken per class/stratum  

`Properties:`
* *counter(Tensor)*: (M,) number of samples each stratum has  