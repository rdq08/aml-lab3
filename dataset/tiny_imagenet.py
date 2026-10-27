
import os
import shutil

import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T


def get_dataloaders(root="tiny-imagenet/tiny-imagenet-200", batch_size=32):

    # Organize validation images into class folders
    val_dir = os.path.join(root, "val")
    annotations_file = os.path.join(val_dir, "val_annotations.txt")
    images_dir = os.path.join(val_dir, "images")

    if os.path.isdir(images_dir):
        with open(annotations_file) as f:
            for line in f:
                fn, cls, *_ = line.strip().split("\t")

                class_dir = os.path.join(val_dir, cls)
                os.makedirs(class_dir, exist_ok=True)

                src = os.path.join(images_dir, fn)
                dst = os.path.join(class_dir, fn)

                shutil.copyfile(src, dst)

        shutil.rmtree(images_dir)

    # Define image transformations
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Create datasets
    train_dataset = ImageFolder(
        root=os.path.join(root, "train"),
        transform=transform
    )

    val_dataset = ImageFolder(
        root=val_dir,
        transform=transform
    )
    
    assert train_dataset.class_to_idx == val_dataset.class_to_idx

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader
