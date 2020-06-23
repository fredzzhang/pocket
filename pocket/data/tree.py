"""
Tree-based data structures

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import numpy as np

from .base import DataDict

class Node:
    """Base class for a tree node

    Arguments:
        name(str): Name of the node
        parent(Node): Parent node
        data: Data of the current node. It could be any format
    """
    def __init__(self, name, parent=None, data=None):
        self._name = name
        self._parent = parent
        self._data = data

        self._children = DataDict()
        self._path = os.path.join(parent.path, name) \
            if parent is not None else "/"

    @property
    def name(self):
        return self._name

    @property
    def parent(self):
        return self._parent

    @property
    def data(self):
        return self._data

    @property
    def children(self):
        return self._children

    @property
    def path(self):
        return self._path

    def add(self, input_dict=None, **kwargs):
        """Add children to the current node

        Arguments:
            input_dict(dict[Node])
            kwargs(dict[Node])
        """
        if input_dict is not None:
            self._children.update(input_dict)
        self._children.update(kwargs)

class DatasetTree:
    """Base class for a dataset tree

    Arguments:
        num_classes(int): Number of classes in the dataset
        image_labels(List[List[int]]): Labels for each image. If there are 
            multiple instances of one class in an image, the class label will 
            appear multiple times
    """
    def __init__(self, num_classes, image_labels):
        root = Node("root")
        root.add(
            images=Node("images", parent=root),
            classes=Node("classes", parent=root)
        )

        root._children.images.add({
            str(i): Node(
                name=str(i),
                parent=root.children.images,
                data={
                    str(j): labels.count(j)
                    for j in np.unique(np.asarray(labels))
                },
            ) for i, labels in enumerate(image_labels)
        })

        class_pool = [{} for _ in range(num_classes)]
        for i, labels in enumerate(image_labels):
            for j in np.unique(np.asarray(labels)):
                class_pool[j][i]=labels.count(j)

        root._children.classes.add({
            str(i): Node(
                name=str(i),
                parent=root.children.classes,
                data=pool,
            ) for i, pool in enumerate(class_pool)
        })

        self._root = root
        self._current_node = root

    @property
    def root(self):
        return self._root

    def cn(self):
        """Return the current node"""
        return self._current_node

    def ls(self):
        """List the children of the current node"""
        return list(self._current_node.children.keys())

    def path(self):
        """Return the path of the current node"""
        return self._current_node.path

    def up(self):
        """Move upwards in the tree"""
        if self._current_node._parent is not None:
            self._current_node = self._current_node._parent

    def down(self, name):
        """Move downwards in the tree"""
        if name in self._current_node._children:
            self._current_node = self._current_node._children[name]
        else:
            raise ValueError("Unknown child {}".format(name))