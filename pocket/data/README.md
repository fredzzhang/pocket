## __`CLASS`__ pocket.data.database.DataDict
    
Data dictionary class. Allows setting and accessing dict keys as class attributes. 
Initialize the class with a python dict. By default, an empty dict is applied.  

`Parameters:`
* *dictargs(dict, optional)*: python built-in dict (default: __*{}*__)

`Methods:`
* *save(path)*: Save into a .pkl file as a standard python dict  
* *load(path)*: Load a python dict from a .pkl file as data dict  
* *is_empty()*: Return true if the dict has any keys, false otherwise  

## __`CLASS`__ pocket.data.database.Imdb

Base class for image database. By default, *\_\_len\_\_()* returns the number of images 
and *\_\_getitem\_\_()* fetches an image. These two methods have alternatives as class 
properties to ease potential overriding.  

`Parameters:`
* *subset(str)*: a choice between 'TRAIN' and 'TEST'  
* *cfg(CfgNode)*: configuration class with following attrubutes
    * _cfg.NAME(str)_: name of the dataset  
    * _cfg.TRAIN.IMAGES(str)_: directory of images in training set  
    * _cfg.TRAIN.IMDB(str)_: path to save training set image database  
    * _cfg.TEST.{CONFIG}_: repeat training set configurations for test set 

`Properties:`
* *name(str)*: name of the dataset and subset  
* *num_images(int)*: number of images  
* *num_classes(int)*: number of classes  
* *imdb(DataDict)*: a user-defined image database as per *Imdb._construct_imdb*  

`Methods:`
* *image_path(i)*: Return the path of image corresponding to index i  
* *fetch_image(i)*: Return the image as ndarray given index i  
    
## __`CLASS`__ pocket.data.database.Detdb

Base class for bounding-box-based detection database. Child class of _pocket.data.database.Imdb_. By default, the ground truth and detection bouding boxes are assumed to have been written in .txt files. For different files formats, override the following methods  
* *Detdb._construct_src_paths()*
* *Detdb._load_from_src_files()*

`Parameters:`
* subset(str): a choice between 'TRAIN' and 'TEST'  
* cfg(CfgNode): configuration class with following attrubutes
    * ..._Parent Class Configuration_
    * *cfg.NUM_CLASSES(str)*: number of classes in the dataset  
    * _cfg.INTVL(int)_: an integer interval between two progress updates in database construction  
    * _cfg.TRAIN.GTDIR(str)_: directory of ground truth annotations for training set
    * _cfg.TRAIN.GTDB(str)_: path to save training set ground truth database
    * _cfg.TRAIN.DETDIR(str)_: directory of detections for training set
    * _cfg.TRAIN.DETDB(str):_: path to save training set detection database  
    * _cfg.TEST.{CONFIG}_: repeat training sest configurations for test set 

`Properties:`
* *gtdb(DataDict)*: a user-defined ground truth bounding box database as per *Detdb._construct_gtdb*
* *detdb(DataDict)*: a user-defined detection bounding box database as per *Detdb._construct_detdb*

`Methods:`
* *fg_obj(i)*: Return the foreground object classes in the image with index i  

## __`CLASS`__ pocket.data.samplers.StratifiedBatchSampler

Samples a specified number of elements from each stratum to form a minibatch. As a convention from *torch.utils.data.Sampler*, *\_\_len\_\_()* returns the total number of samples and *\_\_iter\_\_()* returns the iterator of the sampler  

`Parameters:`
* *strata(list)*: (M,) a list of iterables, each of which contains the indices for one stratum  
* *num_strata_each(int)*: number of strata to sample from in a single minibatch
* *samples_per_stratum(int)*: number of samples taken from each stratum in single minibatch
* *num_batch(int)*: number of minibatches in an epoch
* *negative_pool(ndarray, optional)*: a pool of indices of negative samples (default: __*None*__)
* *num_negatives(int, optional)*: number of negative samples in a single minibatch (default: __*0*__)

## __`CLASS`__ pocket.data.samplers.StratifiedSampler

Samples a specified number of elements from each stratum for a number of iterations.
As a convention from *torch.utils.data.Sampler*, *\_\_len\_\_()* returns the total number of samples and *\_\_iter\_\_()* returns the iterator of the sampler  

`Parameters:`
* *strata(list)*: (M,) a list of iterables, each of which contains the indices for one stratum  
* *num_iter(int)*: number of iterations of sampling  
* *samples_per_stratum(Tensor or ndarray)*: number of samples taken from a stratum per iteration  

## __`CLASS`__ pocket.data.samplers.MultiLabelStratifiedSampler

Samples a specified number of elements from each stratum for a number of iterations. Samples could potentially have multiple labels, meaning that certain strata share certain samples.  

The sampler suppresses those classes/strata that tend to co-occur, by skipping stratum that has more training samples than the iteration counter. *\_\_iter\_\_()* returns the iterator of the sampler and *\_\_len\_\_()* return current sample count.  

`Parameters:`
* *strata(list)*: (M,) a list of iterables, each of which contains the indices for one stratum  
* *labels(Tensor)*: (N, M) labels of all samples
* *samples_per_class(int)*: number of samples taken per class/stratum  

`Properties:`
* *counter(Tensor)*: (M,) number of samples each stratum has  
