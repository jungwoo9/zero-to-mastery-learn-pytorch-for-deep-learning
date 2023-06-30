"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

import argparse

# Set parser
parser = argparse.ArgumentParser(description="train the pytorch model")

# Set arguemnts
parser.add_argument('--train_dir')
parser.add_argument('--test_dir')
parser.add_argument('--learning_rate')
parser.add_argument('--batch_size')
parser.add_argument('--num_epochs')
parser.add_argument('--hidden_units')

args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = int(args.num_epochs) if args.num_epochs else  5
BATCH_SIZE = int(args.batch_size) if args.batch_size else  32
HIDDEN_UNITS = int(args.hidden_units) if args.hidden_units else  10
LEARNING_RATE = float(args.learning_rate) if args.learning_rate else  0.001

# Setup directories
train_dir = args.train_dir if args.train_dir else "data/pizza_steak_sushi/train"
test_dir = args.test_dir if args.test_dir else  "data/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir, test_dir, data_transform, BATCH_SIZE)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    3,
    HIDDEN_UNITS,
    len(class_names)
).to(device)

print(model)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

# Start training with help from engine.py
engine.train(model, train_dataloader, test_dataloader, optimizer, loss_fn, NUM_EPOCHS, device)

# Save the model with help from utils.py
utils.save_model(model, "models", "05(1)_going_modular_script_mode_tinyvgg_model.pth")
