## __`CLASS`__ pocket.diagnose.confusion_matrix.ConfusionMatrix

Confusion matrix class.

`Parameters:`
* *num_cls(int, optional)*: number of classes in the matrix (default: *__0__*)
* *mode(str, optional)*: evaluation mode, a choice between 'FREQ' and 'MEAN' (default: *__'FREQ'__*)
    * 'FREQ': confusion matrix stores the frequency of each predicted class
    * 'MEAN': confusion matrix stores the mean prediction scores

`Properties:`
* *cmatrix(Tensor)*: confusion matrix
* *mode(str)*: evaluation mode

`Methods`:
* *reset()*: reset the confusion matrix with zeros
* *push(out, labels)*: update the confusion matrix based on output of a network and labels
    * out(Tensor): (M, N)
    * labels(Tensor): (M, N)
* *show()*: plot the confusion matrix
* *save(cache_dir)*: save the confusion matrix into specified directory with evaluation mode in the file name
* *load(cache_path)*: load the confusion matrix from a .pkl file specified in the path
* *merge(ind_range)*: merge certain classes in the matrix
    * ind_range(Tensor): (I, 2) starting and end indices of intervals to be merged into one class
* *normalize(dim=0)*: normalize the confusion matrix

## __`FUNCTION`__ pocket.diagnose.confusion_matrix.compute_confusion_matrix

Compute the confusion matrix given a network and a dataloader

`Parameters`:
* *net(Module)*: a PyTorch neural nework
* *dataloader(iterable)*: an iterable, preferably of DataLoader type, with batch output in the format \[INPUT_1, ..., INPUT_N, LABELS\]
* *cache_dir(str)*: directory to save cache
* *class_of_interest(Tensor)*: indices of classes to save in the confusion matrix
* *mode(str, optional)*: evaluation mode, (default: *__'FREQ'__*)
* *device(str, optional)*: primary device used for network forward pass, (default: *__'cpu'__*)
* *use_sig(bool, optional)*: whether to append a sigmoid layer to the network, (default: *__True__*)
* *formatter(handle, optional)*: a function handle used to format batch data, (default: *__lambda a: a__*)

## __`CLASS`__ pocket.diagnose.visualizer.LearningCurveVisualizer

Read data from a source .txt file, extract training and validation losses and plot them.

`Parameters:`
* *src_path(str)*: path of the source .txt file

`Properties:`
* *step(ndarray)*: step points at which losses are recorded
* *loss(ndarray)*: training and validation losses if provided

`Methods:`:
* *show(scale='linear')*: plot the learning curve with specified scale  

## __`CLASS`__ pocket.diagnose.visualizer.ParamVisualizer

Plot the parameters of a neural network layer

`Parameters:`
* *params(Tensor)*: weights or biases of a PyTorch layer

`Properties:`
* *params(Tensor)*: weights or biases stored

`Methods:`
* *show()*: plot the parameters

## __`CLASS`__ pocket.diagnose.visualizer.NetVisualizer

Load the state dict of a PyTorch module and construct a _pocket.diagnose.visualizer.ParamVisualizer_ for each parameter block, with same keys as the state dict itself. Method *\_\_len\_\_()* return the number of parameter blocks

`Parameters:`
* *pt_path(str, optional)*: path of a PyTorch model, typically ends with .pt (default: __*None*__)
* *ckpt_path(str, optional)*: path of a checkpoint file, with key *model_state_dict* (default: __*None*__)

`Properties:`
* *keys(list)*: name of the parameter blocks

## __`CLASS`__ pocket.diagnose.visualizer.ImageVisualizer

Visualizer for an image database. Method *\_\_len\_\_()* return the number of images

`Parameters:`
* *imdir(str)*: directory of source images
* *labels(ndarray, optional)*: (N, M) one-hot encoded labels for all N images (default: __*None*__)
* *descpt(ndarray, optional)*: (M,) string descriptions for each of M classes (default: __*None*__)

`Methods`:
* *show(i)*: display the image corresponding to index i and associated labels
* *image_path(i)*: show the path of image corresponding to index i

## __`CLASS`__ pocket.diagnose.visualizer.BoxVisualizer

Visualizer for bounding-box-based dataset. Child class of _pocket.diagnose.visualizer.ImageVisualizer_.

`Parameters:`
* *imdir(str)*: directory of source images
* *boxdir(str)*: directory of bounding box source files, presumably .txt files

`Methods`:
* *show(i)*: to be implemented...
