import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms as T
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from torchvision.datasets.folder import *
from typing import *
from .exemplar import ExamplarImageFolder


class FilterableImageFolder(ImageFolder):
    def __init__(self, root, valid_classes, num_old_cls):
        self.valid_classes = valid_classes
        self.num_old_cls = num_old_cls
        super().__init__(root)

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        used_classes = [valid_class for valid_class in classes if valid_class in self.valid_classes]
        used_classes = sorted(used_classes, key=str.casefold)
        
        if not classes:
            raise FileNotFoundError("Cannot find any class folder in path: {0}".format(directory))
        if not used_classes:
            raise FileNotFoundError("Cannot find any used class folder ({0}) in path: {1}".format(classes, directory))

        class_to_idx = {class_name: (i+self.num_old_cls) for i, class_name in enumerate(used_classes)}

        return used_classes, class_to_idx

    def get_filename(self, indice):
        return self.imgs[indice][0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        img_id = self.get_filename(index)

        return sample, target, img_id

    def __len__(self):
        return len(self.samples)


class PACS_INC(object):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["A", "C", "P", "S"]
    N_STEPS = 5001
    N_WORKERS = 4
    INPUT_SHAPE = (3, 224, 224)

    def __init__(self, root, current_session, num_old_cls, num_new_cls, num_of_exemplar, test_envs):
        self.dir = os.path.join(root, "PACS/")
        environments = [f.name for f in os.scandir(self.dir) if f.is_dir()]
        environments = sorted(environments)
        self.environments = environments

        print('---------- dataset information: PACS_INC ----------')
        environment_name = ['art_painting', 'cartoon', 'photo', 'sketch']
        total_class = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        old_cls = total_class[:num_old_cls]
        new_cls = total_class[num_old_cls: (num_old_cls+num_new_cls)]
        test_cls = total_class[: (num_old_cls+num_new_cls)]
        print('old_cls: {0}'.format(old_cls))
        print('new_cls: {0}'.format(new_cls))
        print('test_cls: {0}'.format(test_cls))

        self.datasets = []
        for environment in environments:
            path = os.path.join(self.dir, environment)
            if environment == environment_name[test_envs[0]]:
                env_dataset = FilterableImageFolder(path, valid_classes=test_cls, num_old_cls=0)
            else:
                env_dataset = FilterableImageFolder(path, valid_classes=new_cls, num_old_cls=num_old_cls)
            print('current_session: {0} | environment: {1}, class: {2}, num_of_img: {3}'.
                  format(current_session, environment, env_dataset.find_classes(path), len(env_dataset)))
            self.datasets.append(env_dataset)

        if num_old_cls > 0 and num_of_exemplar > 0:
            print('--- generate dataset for old class exemplars.')
            path = os.path.join(self.dir)
            env_dataset = ExamplarImageFolder(path, domains=environments, valid_classes=old_cls, num_examplar=num_of_exemplar, test_envs=environment_name[test_envs[0]])
            print('current_session: {0} | environment: examplar, class: {1}, num_of_img: {2}'.
                  format(current_session, env_dataset.find_classes(path), len(env_dataset)))
            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224)
        self.num_classes = num_old_cls + num_new_cls

        print('current_session: {0} | total num_classes: {1}'.format(current_session, self.num_classes))
        print('----------------------------------------------------')


    def __getitem__(self, index):
        """
        Return: sub-dataset for specific domain
        """
        return self.datasets[index]

    def __len__(self):
        """
        Return: # of sub-datasets
        """
        return len(self.datasets)


class OfficeHome_INC(object):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["A", "C", "P", "R"]
    N_STEPS = 5001
    N_WORKERS = 4
    INPUT_SHAPE = (3, 224, 224)

    def __init__(self, root, current_session, num_old_cls, num_new_cls, num_of_exemplar, test_envs):
        self.dir = os.path.join(root, "office_home/")
        environments = [f.name for f in os.scandir(self.dir) if f.is_dir()]
        environments = sorted(environments)
        self.environments = environments

        print('---------- dataset information: OfficeHome_INC ----------')
        environment_name = ['Art', 'Clipart', 'Product', 'Real World']
        total_class = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 
        'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 
        'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 
        'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 
        'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 
        'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 
        'Soda', 'Speaker', 'Spoon', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'TV', 'Webcam']

        old_cls = total_class[:num_old_cls]
        new_cls = total_class[num_old_cls: (num_old_cls+num_new_cls)]
        test_cls = total_class[: (num_old_cls+num_new_cls)]
        print('old_cls: {0}'.format(old_cls))
        print('new_cls: {0}'.format(new_cls))
        print('test_cls: {0}'.format(test_cls))

        self.datasets = []
        for environment in environments:
            path = os.path.join(self.dir, environment)
            if environment == environment_name[test_envs[0]]:
                env_dataset = FilterableImageFolder(path, valid_classes=test_cls, num_old_cls=0)
            else:
                env_dataset = FilterableImageFolder(path, valid_classes=new_cls, num_old_cls=num_old_cls)
            print('current_session: {0} | environment: {1}, class: {2}, num_of_img: {3}'.
                  format(current_session, environment, env_dataset.find_classes(path), len(env_dataset)))
            self.datasets.append(env_dataset)

        if num_old_cls > 0 and num_of_exemplar > 0:
            print('--- generate dataset for old class exemplars.')
            path = os.path.join(self.dir)
            env_dataset = ExamplarImageFolder(path, domains=environments, valid_classes=old_cls, num_examplar=num_of_exemplar, test_envs=environment_name[test_envs[0]])
            print('current_session: {0} | environment: examplar, class: {1}, num_of_img: {2}'.
                  format(current_session, env_dataset.find_classes(path), len(env_dataset)))
            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224)
        self.num_classes = num_old_cls + num_new_cls

        print('current_session: {0} | total num_classes: {1}'.format(current_session, self.num_classes))
        print('----------------------------------------------------')


    def __getitem__(self, index):
        # Return: sub-dataset for specific domain

        return self.datasets[index]

    def __len__(self):
        # Return: # of sub-datasets

        return len(self.datasets)


class DomainNet_INC(object):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["A", "C", "P", "R"]
    N_STEPS = 5001
    N_WORKERS = 4
    INPUT_SHAPE = (3, 224, 224)

    def __init__(self, root, current_session, num_old_cls, num_new_cls, num_of_exemplar, test_envs):
        self.dir = os.path.join(root, "DomainNet/")
        environments = [f.name for f in os.scandir(self.dir) if f.is_dir()]
        environments = sorted(environments)
        self.environments = environments

        print('---------- dataset information: DomainNet_INC ----------')
        environment_name = ['clipart', 'painting', 'real', 'sketch']

        total_class = ['aircraft_carrier', 'alarm_clock', 'ant', 'anvil', 'asparagus', 'axe', 
        'banana', 'basket', 'bathtub', 'bear', 'bee', 'bird', 'blackberry', 'blueberry', 'bottlecap', 'broccoli', 
        'bus', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'camera', 'candle', 'cannon', 'canoe', 
        'carrot', 'castle', 'cat', 'ceiling_fan', 'cell_phone', 'cello', 'chair', 'chandelier', 'coffee_cup', 'compass', 
        'computer', 'cow', 'crab', 'crocodile', 'cruise_ship', 'dog', 'dolphin', 'dragon', 'drums', 'duck', 
        'dumbbell', 'elephant', 'eyeglasses', 'feather', 'fence', 'fish', 'flamingo', 'flower', 'foot', 'fork', 
        'frog', 'giraffe', 'goatee', 'grapes', 'guitar', 'hammer', 'helicopter', 'helmet', 'horse', 'kangaroo', 
        'lantern', 'laptop', 'leaf', 'lion', 'lipstick', 'lobster', 'microphone', 'monkey', 'mosquito', 'mouse', 
        'mug', 'mushroom', 'onion', 'panda', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'pig', 
        'pillow', 'pineapple', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'rhinoceros', 'rifle', 'saxophone', 
        'screwdriver', 'sea_turtle', 'see_saw', 'sheep', 'shoe', 'skateboard', 'snake', 'speedboat', 'spider', 'squirrel', 
        'strawberry', 'streetlight', 'string_bean', 'submarine', 'swan', 'table', 'teapot', 'teddy-bear', 'television', 'The_Eiffel_Tower', 
        'The_Great_Wall_of_China', 'tiger', 'toe', 'train', 'truck', 'umbrella', 'vase', 'watermelon', 'whale', 'zebra']

        old_cls = total_class[:num_old_cls]
        new_cls = total_class[num_old_cls: (num_old_cls+num_new_cls)]
        test_cls = total_class[: (num_old_cls+num_new_cls)]
        print('old_cls: {0}'.format(old_cls))
        print('new_cls: {0}'.format(new_cls))
        print('test_cls: {0}'.format(test_cls))

        self.datasets = []
        for environment in environments:
            path = os.path.join(self.dir, environment)
            if environment == environment_name[test_envs[0]]:
                env_dataset = FilterableImageFolder(path, valid_classes=test_cls, num_old_cls=0)
            else:
                env_dataset = FilterableImageFolder(path, valid_classes=new_cls, num_old_cls=num_old_cls)
            print('current_session: {0} | environment: {1}, class: {2}, num_of_img: {3}'.
                  format(current_session, environment, env_dataset.find_classes(path), len(env_dataset)))
            self.datasets.append(env_dataset)

        if num_old_cls > 0 and num_of_exemplar > 0:
            print('--- generate dataset for old class exemplars.')
            path = os.path.join(self.dir)
            env_dataset = ExamplarImageFolder(path, domains=environments, valid_classes=old_cls, num_examplar=num_of_exemplar, test_envs=environment_name[test_envs[0]])
            print('current_session: {0} | environment: examplar, class: {1}, num_of_img: {2}'.
                  format(current_session, env_dataset.find_classes(path), len(env_dataset)))
            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224)
        self.num_classes = num_old_cls + num_new_cls

        print('current_session: {0} | total num_classes: {1}'.format(current_session, self.num_classes))
        print('----------------------------------------------------')


    def __getitem__(self, index):
        # Return: sub-dataset for specific domain

        return self.datasets[index]

    def __len__(self):
        # Return: # of sub-datasets

        return len(self.datasets)


