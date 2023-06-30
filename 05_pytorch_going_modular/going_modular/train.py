"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

print(LEARNING_RATE)
# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

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
engine.train(model, train_dataloader, test_dataloader, loss_fn, optimizer, NUM_EPOCHS, device)

# Save the model with help from utils.py
utils.save_model(model, "models", "05(1)_going_modular_script_mode_tinyvgg_model.pth")
