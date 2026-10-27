
import torch
from torch import nn

from models.customnet import CustomNet
from dataset.tiny_imagenet import get_dataloaders
from train import train
from eval import validate


def main():

    # Load TinyImageNet dataset
    train_loader, val_loader = get_dataloaders(batch_size=32)

    # Create the CNN and move it to GPU
    model = CustomNet().cuda()

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9
    )

    best_acc = 0

    # Run the training process for 10 epochs
    num_epochs = 10

    for epoch in range(1, num_epochs + 1):

        train(epoch, model, train_loader, criterion, optimizer)

        # Validation after each training epoch
        val_accuracy = validate(model, val_loader, criterion)

        # Best validation accuracy
        best_acc = max(best_acc, val_accuracy)

    print(f'Best validation accuracy: {best_acc:.2f}%')


if __name__ == "__main__":
    main()
