### __`CLASS`__ pocket.data.DataDict(_input_dict: Optional[dict] = None, **kwargs_)
    
Data dictionary class inherited from Python dictionary. This class allows setting and accessing dict keys as class attributes and provides save/load utilities using pickle as backend. Initialise the class with a python dict and/or keyworded arguments. By default, an empty dict is applied.

`Parameters:`
* **input_dict**: A Python dict
* **kwargs**: Keyworded arguments

`Methods:`
* save(_path: str, mode: str, **kwargs_) -> None: Save into a _.pkl_ file as a Python dict
    * __path__: A valid path to which the data will be saved
    * __mode__: An optional string that specifies the mode in which the file is opened
    * __kwargs__: Keyworded arguments for *pickle.dump*
* load(_path: str, mode: str, **kwargs_) -> None: Load a Python dict from a _.pkl_ file
    * __path__: A valid path from which the data will be loaded
    * __mode__: An optional string that specifies the mode in which the file is opened
    * __kwargs__: Keyworded arguments for *pickle.load*
* is_empty() -> bool: Return `True` if the dict has any keys, `False` otherwise

`Examples:`
```python
>>> from pocket.data import DataDict
>>> person = DataDict()
>>> person.is_empty()
True
>>> person.age = 15
>>> person.sex = 'male'
>>> person.save('./person.pkl', 'w')
```

---

### __`CLASS`__ pocket.data.ImageDataset(_root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None_)

Base class for image dataset. By default, *\_\_len\_\_()* returns the number of images and *\_\_getitem\_\_()* fetches an image. For string representations, *\_\_str\_\_()* returns the dataset information, and *\_\_repr\_\_()* returns instantiation arguments.

`Parameters:`
* **root**: Root directory where images are downloaded to
* **transform**: A function/transform that takes in an PIL image and returns a transformed version
* **target_transform**: A function/transform that takes in the target and transforms it
* **transforms**: A function/transform that takes input sample and its target as entry and returns a transformed version

`Methods:`
* load_image(_path: str_) -> Image: Load an image as _PIL.Image_
    * __path__: A valid path of a source image

---

### __`CLASS`__ pocket.data.DataSubset(_dataset: Dataset, pool: List[int]_)

A subset of data with access to all attributes of the original dataset. In particular, method *\_\_len\_\_()* and *\_\_getitem\_\_()* have been overriden with corresponding information of the subset.

`Parameters:`
* **dataset**: Original dataset
* **pool**: The pool of indices for the subset

---
### __`CLASS`__ pocket.data.HICODetSubset(_dataset: Dataset, pool: List[int]_)

A subset class for HICO-DET dataset. Necessary class methods and properties have been overriden for the subset.

`Parameters:`
* **dataset**: Original dataset
* **pool**: The pool of indices for the subset

---

### __`CLASS`__ pocket.data.HICODet(_root: str, anno_file: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None_)

HICO-DET dataset for human-object interaction detection. *\_\_len\_\_()* returns the number of images and *\_\_getitem\_\_()* fetches an image and the corresponding annotations. For string representations, *\_\_str\_\_()* returns the dataset information, and *\_\_repr\_\_()* returns instantiation arguments. Images without bounding box annotations will be skipped automatically during indexing.

`Parameters:`
* **root**: Root directory where images are downloaded to
* **anno_file**: Path to json annotation file
* **transform**: A function/transform that takes in an PIL image and returns a transformed version
* **target_transform**: A function/transform that takes in the target and transforms it
* **transforms**: A function/transform that takes input sample and its target as entry and returns a transformed version

`Methods`:
* \_\_getitem\_\_(_i: int_) -> tuple: Return a tuple of the transformed image and annotations. The annotations are formatted in the form of a Python dict with the following keys
    * boxes_h: List[list]
    * boxes_o: List[list]
    * hoi: List[int]
    * verb: List[int]
    * object: List[int]
* split(_ratio: float_) -> Tuple[HICODetSubset, HICODetSubset]: Split the dataset according to given ratio. 
    * __ratio__: The percentage of training set between 0 and 1
* filename(_idx: int_) -> str: Return the image file name given the index

`Properties:`
* annotations -> List[dict]: All annotations for the dataset
* class_corr -> List[Tuple[int, int, int]]: Class correspondence matrix in zero-based index in the order of [*hoi_idx*, *obj_idx*, *verb_idx*]
* object_n_verb_to_interaction -> List[list]: The interaction classes corresponding to an object-verb pair. An interaction class index can be found by the object index and verb index (in the same order). Invalid combinations will return None.
* object_to_interaction -> List[list]: The interaction classes that involve each object type
* object_to_verb -> List[list]: The valid verbs for each object type
* anno_interaction -> List[int]: Number of annotated box pairs for each interaction class
* anno_object -> List[int]: Number of annotated box pairs for each object class
* anno_action -> List[int]: Number of annotated box pairs for each action class
* objects -> List[str]: Object names
* verbs -> List[str]: Verb (action) names
* interactions -> List[str]: Interaction names

---

### __`CLASS`__ pocket.data.Node(_name: str, parent: Optional[Node] = None, data: Optional[Any] = None_)

Base class for a tree node

`Parameters:`
* **name**: Name of the node
* **parent**: Parent node
* **data**: Data of the current node. It could be any format

`Methods`:
* add(_input_dict: Optional[dict] = None, **kwargs_) -> None: Add children to the current node
    * **input_dict**: Children nodes formatted as a dictionary. The keys should be the same as the node names.
    * **kwargs**: Children nodes formatted as keyworded arguments.

`Properties`:
* name -> str: Name of the current node
* parent -> Node: Parent node
* data -> Any: Data stored in the current node
* children -> DataDict: Children of the current node. The keys should be the same as the node names.
* path -> str: Path of the current node (relative to the root)

`Examples:`

```python
>>> from pocket.data import Node
>>> root = Node('root')
>>> root.add(left=Node('left', parent=root, data=1))
>>> root.add(right=Node('right', parent=root, data=2))
>>> root.path
'/'
>>> root.children.keys()
dict_keys(['left', 'right'])
>>> root.children.left.data
1
>>> root.children.right.path
'/right'
```

---

### __`CLASS`__ pocket.data.DatasetTree(_num_classes: int, image_labels: List[list]_)

Base class for a dataset tree. This class is used to build dataset navigators. Given image labels, the constructed tree has the following structure

```
root
└───images
│   │───0
│   │───1
│   │───...
│   
└───classes
    │───0
    │───1
    |───...
```

`Parameters:`
* **num_classes**: Number of classes in the dataset
* **image_labels**: Labels for each image. If there are multiple instances of one class in an image, the class label will appear multiple times

`Methods:`
* cn() -> Node: Return the current node
* ls() -> List[str]: List the children of the current node
* path() -> str: Return the path of the current node
* up() -> None: Move upwards in the tree
* down(_name: str_) -> None: Move downwards in the tree to the specified node

`Properties`:
* root -> Node: Return the root node