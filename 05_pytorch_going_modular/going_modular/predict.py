"""
Contains function to predict give iamge and model
"""
from torchvision.io import read_image
from torchvision import transforms
import torch

import argparse

import model_builder

def predict_image():
    """
    Classify the image with trained model
    """
    # Set parser
    parser = argparse.ArgumentParser(description="predict label with input image")

    # Set arguemnts
    parser.add_argument('--image')

    args = parser.parse_args()

    image_path = args.image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
    ])

    # load image with dtype float32
    img = read_image(image_path).type(torch.float32)

    # scale from 0 to 1
    img = img / 255
    
    # resize to (64, 64)
    img = transform(img)

    # send image to target device
    img.to(device)

    # get instance of model
    loaded_model = model_builder.TinyVGG(3, 128, 3)

    # get trained parameters
    loaded_model.load_state_dict(torch.load(f="models/05(1)_going_modular_script_mode_tinyvgg_model.pth"))
    
    # predict label
    loaded_model.eval()
    
    with torch.inference_mode():
        img = img.unsqueeze(dim=0)

        pred = loaded_model(img)

    pred_label = torch.softmax(pred, dim=1).argmax(dim=1)
    
    print(['pizza', 'steak', 'sushi'][pred_label])
    return ['pizza', 'steak', 'sushi'][pred_label]

if __name__=="__main__":
    predict_image()
