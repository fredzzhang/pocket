## __`CLASS`__ lib.utils.basic.NetTrainer

Network trainer class

`Parameters:`
* *net(Module)*: PyTorch network module
* *loss_fn(Module)*: PyTorch loss function module
* *cache_dir(str)*: directory to save checkpoints and training log
* *train_loader(iterable)*: dataloader for training set, preferably as DataLoader type, with batch output in the format \[INPUT_1, ..., INPUT_N, LABELS\]
* *print_interval(int)*: number of steps to print losses
* *optim(str, optional)*: optimizer used in training, a choice between 'SGD' and 'Adam' (default: __*'SGD'*__)
* *optim_params(dict, optional)*: optimizer parameter dict with matched keyword (default: __*{'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4}*__)
* *optim_state_dict(OrderedDict, optional)*: state dict of the optimizer (default: __*None*__)
* *device(str, optional)*: the primary device used in training (default: __*'cpu'*__)
* *multigpu(bool, optional)*: use multi-GPU for network forward pass. When set True, {device} has to be 'cuda:0' (default: __*False*__)
* *val_data(tuple, optional)*: constant validation data, a tuple in the format \[INPUT_1, ..., INPUT_N, LABELS\] (default: __*None*__)
* *val_loader(iterable, optional)*: dataloader for validation set, preferably as DataLoader type, with batch output in the format \[INPUT_1, ..., INPUT_N, LABELS\] (default: __*None*__)
* *lr_scheduler(bool, optional)*: use learning rate scheduler (default: __*True*__)
* *sched_params(dict, optional)*: learning rate scheduler parameter dict (default: __*{'patience':2, 'min_lr':1e-4}*__)
* *input_transform(handle, optional)*: a function handle used to format batch data, (default: __*lambda a: a*__)

`Properties:`
* *step(int)*: current step number, implemented with setter
* *epoch(int)*: current epoch number, implemented with setter

`Methods:`
* *train(nepoch)*: train the network with a specified number of epochs

## __`CLASS`__ lib.utils.basic.NetTester

Network tester class

`Parameters:`
* *net(Module)*: PyTorch network module
* *dataloader(iterable)*: dataloader for the test set, preferably as DataLoader type. By default, the batch output should have the format \[INPUT_1, ..., INPUT_N, LABELS\]
* *device(str, optional)*: the primary device used in training (default: __*'cpu'*__)
* *print_interval(int, optional)*: number of steps to log progress (default: __*500*__)
* *cache_dir(str optional)*: directory to save cache and test log (default: __*'./cache'*__)
* *input_transform(handle, optional)*: a function handle used to format batch data, (default: __*lambda a: a*__)

`Properties:`
* *metric(str)*: the evaluation metric
* *gtdb(ndarray)*: ground truth database, arranged in (NUM_IMAGES, NUM_CLASSES) for 'mAP'
* *detdb(ndarray)*: detection database, arranged in (NUM_IMAGES, NUM_CLASSES) for 'mAP'

`Methods:`
* *set_eval_metric(metric, **kwargs)*: set and prepare for evaluation metric
    * *metric(str)*: the evaluation metric. Only 'mAP' is supported at the moment
    * kwargs for __'mAP'__
        * gt_dir(str): the directory for ground truth files
        * min_IoU(float): minimum IoU for data association
        * num_classes(int): number of classes
        * samples_per_class(int): number of samples to be kept for each class
        * detdb(ndarray, optional): detection results arranged in (N, M) array of objects, where N is the number of images and M is the number of classes. Each entry should be an ndarray of box pairs in the format \[H_x1, H_y1, H_x2, H_y2, O_x1, O_y1, O_x2, O_y2, Score\] (default: __*None*__)
* *eval()*: evaluate the network based on the specified metric

## __`CLASS`__ lib.utils.loss.BCELossForStratifiedBatch

Binary cross entropy loss, modified to mask out co-occurring classes for multi-label classification problems. When applying stratified sampling, co-occurring classes could disrupt the desired balance between the number of samples for different classes. To resolve this problem, mask out the losses for those co-occurring classes, except the actual designated class.  

`Parameters:`
* *cfg(CfgNode)*: configuration class with the following attributes
    * *cfg.NUM_CLS_PER_BATCH(int)*: number of classes/strata to sample from in each minibatch
    * *cfg.POS_GAIN(float)*: weights applied on the positive class
    * *cfg.NUM_POS_SAMPLES(int)*: number of samples to take from each positive class
    * *cfg.NUM_NEG_SAMPLES(int)*: number of samples to take from the negative class

## __`CLASS`__ lib.utils.loss.BCEWithLogitsLossForStratifiedBatch

Binary cross entropy loss coupled with sigmoid function, modified to mask out co-occurring classes for multi-label classification problems. Child class of _lib.utils.loss.BCELossForStratifiedBatch_

`Parameters:`
* *cfg(CfgNode)*: configuration class with the following attributes
    * *cfg.NUM_CLS_PER_BATCH(int)*: number of classes/strata to sample from in each minibatch
    * *cfg.POS_GAIN(float)*: weights applied on the positive class
    * *cfg.NUM_POS_SAMPLES(int)*: number of samples to take from each positive class
    * *cfg.NUM_NEG_SAMPLES(int)*: number of samples to take from the negative class

## __`CLASS`__ lib.utils.io.Log

Logger class

`Parameters:`
* *path(str)*: path of the file to write to
* *mode(str, optional)*: file editing mode (default: __*'wb'*__)

`Properties:`
* *path(str)*: path of the file to write to
* *mode(str)*: file editing mode, implemented with setter

`Methods`:
* *write(descpt)*: write to file given a description string
* *time()*: print time to log

## __`CLASS`__ lib.utils.io.TrainLog

Logger during network training. Child class of _lib.utils.io.Log_

`Parameters:`
* *path(str)*: path of the file to write to
* *mode(str, optional)*: file editing mode (default: __*'a'*__)

`Methods:`
* *log(step, train_loss, val_loss=None)*: print loss to log in a specific format
    * *step(int)*: step number
    * *train_loss(float)*: training loss
    * *val_loss(float, optional)*: validation loss (default: __*None*__)
