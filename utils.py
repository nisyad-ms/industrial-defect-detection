import base64
import io
import json
from pathlib import Path

import torch
from torchvision.transforms import Compose, ToPILImage, ToTensor
from vision_datasets import (DataManifestFactory, DatasetInfo, Usages,
                             VisionDataset)


def load_local_vision_dataset(dataset_name: str, dataset_config_path: Path, root_dir: Path, task_type: str = 'object_detection', usage: Usages = Usages.TEST):

    dataset_info = get_dataset_info(dataset_name, dataset_config_path)

    dataset_manifest = DataManifestFactory.create(
        dataset_info, usage, container_sas_or_root_dir=str(root_dir))
    if dataset_manifest is None:
        raise ValueError('Failed to create dataset manifest')

    vision_dataset = VisionDataset(dataset_info, dataset_manifest)
    if vision_dataset.categories is None:
        raise ValueError('Categories are missing from the dataset')

    elif task_type == 'object_detection':
        dataset = ODDataset(vision_dataset)
    else:
        raise ValueError(f'Unsupported task type {task_type}')
    return dataset


def get_dataset_info(dataset_name: str, config_json_path: Path) -> DatasetInfo:
    with open(config_json_path, 'r') as f:
        config_json = json.load(f)

    dataset_info_dict = [
        config for config in config_json if config['name'] == dataset_name][0]
    return DatasetInfo(dataset_info_dict)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, vision_dataset):
        self._vision_dataset = vision_dataset

    def __len__(self):
        return len(self._vision_dataset)

    def __getitem__(self, index):
        image, targets, _ = self._vision_dataset[index]
        return image, targets

    @property
    def class_names(self):
        categories = self._vision_dataset.categories
        assert categories is not None
        return [x.name for x in categories]

    def get_targets(self, index):
        return self[index][1]


class ICMulticlassDataset(Dataset):
    def __getitem__(self, index):
        image, targets = super().__getitem__(index)
        assert len(targets) == 1
        return image, torch.tensor(targets[0].label_data)

    def get_targets(self, index):
        targets = self._vision_dataset.dataset_manifest.images[index].labels
        assert len(targets) == 1
        return torch.tensor(targets[0].label_data)


class ODDataset(Dataset):
    def __getitem__(self, index):
        image, targets = super().__getitem__(index)
        if not targets:
            return image, torch.zeros(0, 5)
        return image, torch.tensor([t.label_data for t in targets], dtype=torch.float32).reshape(-1, 5)

    def get_targets(self, index):
        image_manifest = self._vision_dataset.dataset_manifest.images[index]
        if not image_manifest.width or not image_manifest.height:
            return self[index][1]  # Fall back to the __getitem__().
        width, height = image_manifest.width, image_manifest.height
        new_targets = [[t.category_id, t.left / width, t.top / height,
                        t.right / width, t.bottom / height] for t in image_manifest.labels]
        return torch.tensor(new_targets, dtype=torch.float32).reshape(-1, 5)


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, transform=None):
        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        img, targets = self._dataset[index]
        if self._transform:
            img = self._transform(img)

        # If there are no targets (good image), return a dummy tensor with label 0
        if targets.numel() == 0:
            targets = torch.tensor([0, -1, -1, -1, -1], dtype=torch.float32).unsqueeze_(0)

        return img, targets


def convert_to_base64(image_path: str):
    from mimetypes import guess_type
    mime_type, _ = guess_type(image_path)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("ascii")
    return base64_image, mime_type


def convert_tensor_to_base64(tensor: torch.Tensor):
    # Convert the tensor to a PIL Image
    transform = ToPILImage()
    pil_image = transform(tensor)

    # Save the PIL image to a bytes buffer
    buffer = io.BytesIO()
    # You can change JPEG to PNG if you prefer
    pil_image.save(buffer, format="PNG")

    # Encode the bytes buffer to Base64
    base64_image = base64.b64encode(buffer.getvalue()).decode("ascii")

    return base64_image, "image/png"


if __name__ == '__main__':
    # Test
    dataset_name = 'mvtec_ad'
    dataset_config_path = Path('datasets.json')
    root_dir = Path('./')
    task_type = 'object_detection'
    dataset = load_local_vision_dataset(
        dataset_name, dataset_config_path, root_dir, task_type)
    transform = Compose([ToTensor()])
    print(dataset[0])
