### __`CLASS`__ pocket.core.LearningEngine(_net: Module, criterion: Callable, train_loader: Iterable, optim: str = 'SGD', optim_params: Optional[dict] = None, optim_state_dict: Optional[dict] = None, lr_scheduler: bool = False, lr_sched_params: Optional[dict] = None, verbal: bool = True, print_interval: int = 100, cache_dir: str = './checkpoints'_)

Base class for learning engines. By default, all available cuda devices will be used. To disable the GPU usage or
manually select devices, use the following command:
```bash
# Use CPU
CUDA_VISIBLE_DEVICES=, python YOUR_SCRIPT.py
# Use CUDA0 and CUDA1
CUDA_VISIBLE_DEVICES=0,1 python YOUR_SCRIPT.py
```
The learning engine itself contains a state. Class attributes that envolve during training are stored as state variables. By default, there are seven state variables: _net_, _optimizer_, _epoch_, _iteration_, *running_loss*, *t_data*, *t_iteration*. In order, they refer to the network (nn.Module), the optimizer (nn.Module), epoch number (int), iteartion number (int) and meters that record running loss, dataloading and iteration time. State variables can be accessed and updated using methods *state_dict()*, *load_state_dict()*, *fetch_state_key(key)* and *update_state_key(\*\*kwargs)*. Refer to the methods section for details.

`Parameters:`
* **net**: The network to be trained
* **criterion**: Loss function
* **train_loader**: Dataloader for training set, with batch input in the format (INPUT_1, ..., INPUT_N, LABELS). Each element should take one of the following forms: Tensor, List[Tensor], Dict[Tensor]
* **optim**: Optimizer to be used. Choose between 'SGD' and 'Adam'
* **optim_params**: Parameters for the selected optimizer. The default setting is learning rate as 1e-3, momentum as 0.9 and weight decay as 5e-4
* **optim_state_dict**: Optimizer state dict to be loaded
* **lr_scheduler**: If True, use MultiStepLR as the learning rate scheduler
* **lr_sched_params**: Parameters for the learning rate scheduler. The default setting is milestones as [50, 100] and gamma as 0.1
* **verbal**: If True, print statistics every fixed interval
* **print_interval**: Number of iterations to print statistics
* **cache_dir**: Directory to save checkpoints

`Methods:`
* \_\_call\_\_(_n: int_) -> None: Run the learning engine for _n_ epochs
* save_checkpoint() -> None: Save a checkpoint containing the model state, optimizer state, epoch and iteration numbers. The checkpoint is saved under the designated cache directory with epoch and iteration numbers included as part of the file name.
* state_dict() -> dict: Return the state dict
* load_state_dict(*dict_in: dict*) -> None: Load state from external dict
* fetch_state_key(*key: str*) -> Any: Return a specific key
* update_state_key(_**kwargs_) -> None: Override specific keys in the state

`Examples:`
```python
>>> # A simple regression problem
>>> import torch
>>> from pocket.core import LearningEngine
>>> m = torch.nn.Linear(2, 1)
>>> data = torch.rand(200, 2)
>>> target = data.sum(1)[:, None]
>>> criterion = torch.nn.MSELoss()
>>> # Use the network, loss function and dataloader as arguments to instantiate the engine
>>> engine = LearningEngine(m, criterion, zip(data, target))
>>> # Pass the number of epochs to run the engine
>>> # The default print interval is 100 steps. Apart from the epoch and iteration numbers,
>>> # the average loss and total running time within 100 steps are printed. Running time
>>> # shows the total dataloading time and iteration time (total time)
>>> engine(2)
Epoch [1/2], Iter. [100/200], Loss: 1.0247, Time[Data/Iter.]: [0.00s/0.03s]
Epoch [1/2], Iter. [200/200], Loss: 0.0934, Time[Data/Iter.]: [0.00s/0.02s]
Epoch [2/2], Iter. [100/200], Loss: 0.0654, Time[Data/Iter.]: [0.00s/0.02s]
Epoch [2/2], Iter. [200/200], Loss: 0.0552, Time[Data/Iter.]: [0.00s/0.03s]
>>> # Override state variables e.g. optimizer
>>> optim = torch.optim.Adagrad(m.parameters())
>>> engine.update_state_key(optimizer=optim)
```

`Inheritance:`

The structure of the learning engine is as follows:
```python
self._on_start()
for ...
    self._on_start_epoch()
    for ...
        self._on_start_iteration()
        self._on_each_iteration()
        self._on_end_iteration()
    self._on_end_epoch()
self._on_end()
```
To inherit from the base class, override the following methods depending on the circumstances
* _on_start(): Executed before training. This method is used for intialisation
* _on_start_epoch(): Executed before each epoch. Call `super()._on_start_epoch()` in child class at the start of the method
* _on_start_iteration(): Executed before each iteration. Call `super()._on_start_iteration()` in child class at the start of the method
* _on_each_iteration(): Runs forward, backward pass and parameter update etc. You should not have to override this method unless necessary
* _on_end_iteration(): Executed after each iteration. Call `super()._on_end_iteration()` in child class at the end of the method
* _on_end_epoch(): Executed after each epoch. Call `super()._on_end_epoch()` in child class at the end of the method
* _on_end(): Executed at the very end.

---

### __`CLASS`__ pocket.core.MultiClassClassificationEngine(_net: Module, criterion: Callable, train_loader: Iterable, val_loader: Optional[Iterable] = None, **kwargs_)

Learning engine for multi-class classification problems. This class is inherited from the base class _LearningEngine_.

`Parameters:`
* **net**: The network to be trained
* **criterion**: Loss function
* **train_loader**: Dataloader for training set, with batch input in the format (INPUT_1, ..., INPUT_N, LABELS). Each element should take one of the following forms: Tensor, List[Tensor], Dict[Tensor]
* **val_loader**: Dataloader for validation set, with batch input in the format [INPUT_1, ..., INPUT_N, LABELS]
* **optim**: Optimizer to be used. Choose between 'SGD' and 'Adam'
* **optim_params**: Parameters for the selected optimizer. The default setting is learning rate as 1e-3, momentum as 0.9 and weight decay as 5e-4
* **optim_state_dict**: Optimizer state dict to be loaded
* **lr_scheduler**: If True, use MultiStepLR as the learning rate scheduler
* **lr_sched_params**: Parameters for the learning rate scheduler. The default setting is milestones as [50, 100] and gamma as 0.1
* **verbal**: If True, print statistics every fixed interval
* **print_interval**: Number of iterations to print statistics
* **cache_dir**: Directory to save checkpoints

`Examples:`
```python
>>> # An example on MNIST handwritten digits recognition
>>> import torch
>>> from torchvision import datasets, transforms
>>> from pocket.models import LeNet
>>> from pocket.core import MultiClassClassificationEngine
>>> # Fix random seed
>>> torch.manual_seed(0)
>>> # Initialize network
>>> net = LeNet()
>>> # Initialize loss function
>>> criterion = torch.nn.CrossEntropyLoss()
>>> # Prepare dataset
>>> train_loader = torch.utils.data.DataLoader(
...     datasets.MNIST('./data', train=True, download=True,
...         transform=transforms.Compose([
...             transforms.ToTensor(),
...             transforms.Normalize((0.1307,), (0.3081,))])
...         ),
...     batch_size=128, shuffle=True)
>>> test_loader = torch.utils.data.DataLoader(
...     datasets.MNIST('./data', train=False,
...         transform=transforms.Compose([
...             transforms.ToTensor(),
...             transforms.Normalize((0.1307,), (0.3081,))])
...         ),
...     batch_size=100, shuffle=False)
>>> # Intialize learning engine and start training
>>> engine = MultiClassClassificationEngine(net, criterion, train_loader,
...     val_loader=test_loader)
>>> # Train the network for one epoch with default optimizer option
>>> # Checkpoints will be saved under ./checkpoints by default, containing 
>>> # saved model parameters, optimizer statistics and progress
>>> engine(1)
```

---

### __`CLASS`__ pocket.core.MultiLabelClassificationEngine(_net: Module, criterion: Callable, train_loader: Iterable, val_loader: Optional[Iterable] = None, ap_algorithm: str = 'INT', **kwargs_)

Learning engine for multi-label classification problems. This class is inherited from the base class _LearningEngine_.

`Parameters:`
* **net**: The network to be trained
* **criterion**: Loss function
* **train_loader**: Dataloader for training set, with batch input in the format (INPUT_1, ..., INPUT_N, LABELS). Each element should take one of the following forms: Tensor, List[Tensor], Dict[Tensor]
* **val_loader**: Dataloader for validation set, with batch input in the format [INPUT_1, ..., INPUT_N, LABELS]
* **ap_algorithm**: Choice of algorithm to evaluate average precision. Refer to _pocket.utils.AveragePrecisionMeter_ for details
* **optim**: Optimizer to be used. Choose between 'SGD' and 'Adam'
* **optim_params**: Parameters for the selected optimizer. The default setting is learning rate as 1e-3, momentum as 0.9 and weight decay as 5e-4
* **optim_state_dict**: Optimizer state dict to be loaded
* **lr_scheduler**: If True, use MultiStepLR as the learning rate scheduler
* **lr_sched_params**: Parameters for the learning rate scheduler. The default setting is milestones as [50, 100] and gamma as 0.1
* **verbal**: If True, print statistics every fixed interval
* **print_interval**: Number of iterations to print statistics
* **cache_dir**: Directory to save checkpoints

`Examples:`

```python
>>> # An example of multi-label classification on voc2012
>>> CLASSES = (
... "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
... "car", "cat", "chair", "cow", "diningtable", "dog",
... "horse", "motorbike", "person", "pottedplant",
... "sheep", "sofa", "train", "tvmonitor")
>>> NUM_CLASSES = len(CLASSES)
>>> import torch
>>> from torchvision import datasets, models, transforms
>>> from pocket.core import MultiLabelClassificationEngine
>>> # Fix random seed
>>> torch.manual_seed(0)
>>> # Initialize network
>>> net = models.resnet50(num_classes=NUM_CLASSES)
>>> # Initialize loss function
>>> criterion = torch.nn.BCEWithLogitsLoss()
>>> # Prepare dataset
>>> def target_transform(x):
...     target = torch.zeros(NUM_CLASSES)
...     anno = x['annotation']['object']
...     if isinstance(anno, list):
...         for obj in anno:
...             target[CLASSES.index(obj['name'])] = 1
...     else:
...         target[CLASSES.index(anno['name'])] = 1
... return target
>>> train_loader = torch.utils.data.DataLoader(
...     datasets.VOCDetection('./data', image_set='train', download=True,
...         transform=transforms.Compose([
...         transforms.Resize([480, 480]),
...         transforms.RandomHorizontalFlip(),
...         transforms.ToTensor(),
...         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
...         ]),
...         target_transform=target_transform),
...     batch_size=32, shuffle=True, num_workers=4)
>>> val_loader = torch.utils.data.DataLoader(
...     datasets.VOCDetection('./data', image_set='val',
...         transform=transforms.Compose([
...         transforms.Resize([480, 480]),
...         transforms.ToTensor(),
...         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
...         ]),
...         target_transform=target_transform),
...     batch_size=32, num_workers=4)
>>> # Initialize learning engine and start training
>>> engine = MultiLabelClassificationEngine(net, criterion, train_loader,
... val_loader=val_loader, print_interval=50,
... optim_params={'lr': 0.1, 'momentum': 0.9, 'weight_decay':5e-4})
>>> # Train the network for one epoch with default optimizer option
>>> # Checkpoints will be saved under ./checkpoints by default, containing 
>>> # saved model parameters, optimizer statistics and progress
>>> engine(1)
```

---

### __`CLASS`__ pocket.core.DistributedLearningEngine(_net: Module, criterion: Callable, train_loader: Iterable, device: Optional[int] = None, optim: str = 'SGD', optim_params: Optional[dict] = None, optim_state_dict: Optional[dict] = None, lr_scheduler: bool = False, lr_sched_params: Optional[dict] = None, verbal: bool = True, print_interval: int = 100, cache_dir: str = './checkpoints'_)

Base class for distributed learning engine. The structure of the class largely follows its non-distributed counterpart _LearningEngine_. 

`Parameters:`
* **net**: The network to be trained
* **criterion**: Loss function
* **train_loader**: Dataloader for training set, with batch input in the format (INPUT_1, ..., INPUT_N, LABELS). Each element should take one of the following forms: Tensor, List[Tensor], Dict[Tensor]
* **device**: CUDA device to be used for training
* **optim**: Optimizer to be used. Choose between 'SGD' and 'Adam'
* **optim_params**: Parameters for the selected optimizer. The default setting is learning rate as 1e-3, momentum as 0.9 and weight decay as 5e-4
* **optim_state_dict**: Optimizer state dict to be loaded
* **lr_scheduler**: If True, use MultiStepLR as the learning rate scheduler
* **lr_sched_params**: Parameters for the learning rate scheduler. The default setting is milestones as [50, 100] and gamma as 0.1
* **verbal**: If True, print statistics every fixed interval
* **print_interval**: Number of iterations to print statistics
* **cache_dir**: Directory to save checkpoints

`Methods:`
* \_\_call\_\_(_n: int_) -> None: Run the learning engine for _n_ epochs
* save_checkpoint() -> None: Save a checkpoint containing the model state, optimizer state, epoch and iteration numbers. The checkpoint is saved under the designated cache directory with epoch and iteration numbers included as part of the file name.
* state_dict() -> dict: Return the state dict
* load_state_dict(*dict_in: dict*) -> None: Load state from external dict
* fetch_state_key(*key: str*) -> Any: Return a specific key
* update_state_key(_**kwargs_) -> None: Override specific keys in the state

`Examples:`
```python
>>> import os
>>> import torch
>>> import torch.distributed as dist
>>> import torch.multiprocessing as mp
>>> from torchvision import datasets, transforms
>>> from pocket.models import LeNet
>>> from pocket.core import DistributedLearningEngine
>>> def main(rank, world_size):
>>>     # Initialisation
>>>     dist.init_process_group(
...         backend="nccl",
...         init_method="env://",
...         world_size=world_size,
...         rank=rank
...     )
>>>     # Fix random seed
>>>     torch.manual_seed(0)
>>>     # Initialize network
>>>     net = LeNet()
>>>     # Initialize loss function
>>>     criterion = torch.nn.CrossEntropyLoss()
>>>     # Prepare dataset
>>>     trainset = datasets.MNIST('../data', train=True, download=True,
...         transform=transforms.Compose([
...         transforms.ToTensor(),
...         transforms.Normalize((0.1307,), (0.3081,))])
...     )
>>>     # Prepare sampler
>>>     train_sampler = torch.utils.data.distributed.DistributedSampler(
...         trainset, num_replicas=world_size, rank=rank
...     )
>>>     # Prepare dataloader
>>>     train_loader = torch.utils.data.DataLoader(
...         trainset, batch_size=128, shuffle=False,
...         num_workers=2, pin_memory=True, sampler=train_sampler)
>>>     # Intialize learning engine and start training
>>>     engine = DistributedLearningEngine(
...         net, criterion, train_loader,
...     )
>>>     # Train the network for one epoch with default optimizer option
>>>     # Checkpoints will be saved under ./checkpoints by default, containing 
>>>     # saved model parameters, optimizer statistics and progress
>>>     engine(5)
>>> # Number of GPUs to run the experiment with
>>> WORLD_SIZE = 2
>>> os.environ["MASTER_ADDR"] = "localhost"
>>> os.environ["MASTER_PORT"] = "8888"
>>> mp.spawn(main, nprocs=WORLD_SIZE, args=(WORLD_SIZE,))
```

`Inheritance:`

The structure of the learning engine is as follows:
```python
self._on_start()
for ...
    self._on_start_epoch()
    for ...
        self._on_start_iteration()
        self._on_each_iteration()
        self._on_end_iteration()
    self._on_end_epoch()
self._on_end()
```
To inherit from the base class, override the following methods depending on the circumstances
* _on_start(): Executed before training. This method is used for intialisation
* _on_start_epoch(): Executed before each epoch. Call `super()._on_start_epoch()` in child class at the start of the method
* _on_start_iteration(): Executed before each iteration. Call `super()._on_start_iteration()` in child class at the start of the method
* _on_each_iteration(): Runs forward, backward pass and parameter update etc. You should not have to override this method unless necessary
* _on_end_iteration(): Executed after each iteration. Call `super()._on_end_iteration()` in child class at the end of the method
* _on_end_epoch(): Executed after each epoch. Call `super()._on_end_epoch()` in child class at the end of the method
* _on_end(): Executed at the very end.