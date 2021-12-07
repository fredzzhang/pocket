<h1 align="center">
  <br>
  <img src="https://user-images.githubusercontent.com/11484831/102578866-445b7b80-414f-11eb-8357-f14b5dbc8187.png" alt="doraemon" width="400" />
  <br>
  Pocket
  <br>
</h1>

<h3 align="center">A Deep Learning Library to Enable Rapid Prototyping<h3>

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#license">License</a>
</p>

## Introduction

Pocket is a fairly lightweight library built on the pupular [PyTorch](https://pytorch.org/) framework. The library provides utilities aimed at lowering the barriers to entry when it comes to training deep neural networks. For most deep learning applications, the relevant code can be divided into three categories: _model_, _dataloader_ and _training script_. Existing frameworks have already provided ample resources for popular models and datasets, yet lack highly encapsulated and flexible training utilities. Pocket is designed to fill this gap.

## Key Features

Pocket provides a range of engine classes that can perform training and testing with minimum amount of code. The following is a simple demo.

![demo](https://user-images.githubusercontent.com/11484831/144946689-b800e42e-2908-4604-934d-32e79396e2f5.gif)

Pocket provides two base classes of engines: __pocket.core.LearningEngine__ and __pocket.core.DistributedLearningEngine__ with the following features:
- [x] CPU/GPU training
- [x] Multi-GPU (distributed) training
- [x] Automatic checkpoint saving
- [x] Elaborate training log

To accomodate distinct training scenarios, the learning engines are implemented with maximum flexibility, and with the following structure
```python
self._on_start()                        # Invoked prior to all epochs
for ...                                 # Iterating over all epochs
    self._on_start_epoch()              # Invoked prior to each epoch
    for ...                             # Iterating over all mini-batches
        self._on_start_iteration()      # Invoked prior to each iteration
        self._on_each_iteration()       # Foward, backward pass etc.
        self._on_end_iteration()        # Invoked after each iteration
    self._on_end_epoch()                # Invoked after each epoch
self._on_end()                          # Invoked after all epochs
```
For details and inheritance of the base learning engines, refer to the [documentation](./pocket/core/README.md). For practical examples, refer to the following

<table class="table">
	<tr>
		<td><span style="font-weight:bold">pocket.core.MultiClassClassificationEngine</span></td>
		<td><a href="./examples/mnist.py">mnist</a></td>
	</tr>
	<tr>
		<td><span style="font-weight:bold">pocket.core.MultiLabelClassificationEngine</span></td>
		<td><a href="./examples/voc2012.py">voc2012</a>, <a href="./examples/hicodet.py">hico</a></td>
	</tr>
	<tr>
		<td><span style="font-weight:bold">pocket.core.DistributedLearningEngine</span></td>
		<td><a href="./examples/distributed/mnist.py">mnist</a></td>
	</tr>
</table>

## Installation

Anaconda/miniconda is recommended for environment management. Follow the steps below to install the library.

```bash
# Create conda environment (python>=3.5)
conda create --name pocket python=3.8
conda activate pocket
# Install dependencies
conda install conda-build
conda install -c anaconda cloudpickle
# Adjust cuda toolkit version as needed
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib tqdm scipy
# Install Pocket under any desired directory
INSTALL_DIR=YOUR_CHOICE
cd $INSTALL_DIR
git clone https://github.com/fredzzhang/pocket.git
conda develop pocket
# Run an example as a test (Optional)
cd pocket/examples
CUDA_VISIBLE_DEVICES=0 python mnist.py
```

## License

[BSD-3-Clause License](./LICENSE)
