import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms as T
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from torchvision.datasets.folder import *
from typing import *


class ExamplarImageFolder(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, is_valid_file=None, domains=None, valid_classes=None, num_examplar=None, test_envs=None):
        IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
        self.domains = domains
        self.valid_classes = valid_classes
        self.num_examplar = num_examplar
        extensions = IMG_EXTENSIONS if is_valid_file is None else None
        classes, class_to_idx = self.find_classes(root)
        self.test_domain = test_envs
        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        samples = self.make_dataset(root, class_to_idx, extensions, is_valid_file, domains)
        self.targets = [s[1] for s in samples]
        self.samples = samples
        super().__init__(root, transform=transform, target_transform=target_transform)

    def make_dataset(self, directory, class_to_idx=None, extensions=None, is_valid_file=None, domains=None):
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """

        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()

        for target_class in sorted(class_to_idx.keys()): # each class
            for domain in domains: # each domain
                if domain != self.test_domain: # do not include exemplars from test domain
                    count = 0
                    class_index = class_to_idx[target_class]
                    target_dir = os.path.join(directory, domain, target_class)

                    if not os.path.isdir(target_dir):
                        continue
                    for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                        for fname in sorted(fnames):
                            path = os.path.join(root, fname)
                            if is_valid_file(path):
                                if count < self.num_examplar:
                                    # sample = self.loader(path)
                                    # print('sample: {0}'.format(sample))
                                    item = path, class_index
                                    instances.append(item)
                                    count = count + 1

                                if target_class not in available_classes:
                                    available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances


    def find_classes(self, root):
        path = os.path.join(root, self.domains[0])
        classes = sorted(entry.name for entry in os.scandir(path) if entry.is_dir())
        used_classes = [valid_class for valid_class in classes if valid_class in self.valid_classes]
        used_classes = sorted(used_classes, key=str.casefold)

        if not classes:
            raise FileNotFoundError("Cannot find any class folder in path: {0}".format(path))
        if not used_classes:
            raise FileNotFoundError("Cannot find any used class folder ({0}) in path: {1}".format(classes, path))
        class_to_idx = {class_name: i for i, class_name in enumerate(used_classes)}
        return used_classes, class_to_idx

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
            sample = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        img_id = self.get_filename(index)
        return sample, target, img_id

    def __len__(self):
        return len(self.samples)

    def get_filename(self, indice):
        return self.samples[indice][0]