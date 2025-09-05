import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import numpy as np
from mcnn_model import MCNN
from my_dataloader import CrowdDataset
import os

def predict_and_count(image_path, model_param_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcnn = MCNN().to(device)
    mcnn.load_state_dict(torch.load(model_param_path, map_location=device))
    mcnn.eval()

    # Load and preprocess image
    img = plt.imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), 2)
    img = img.transpose((2, 0, 1))
    img_tensor = torch.tensor(img, dtype=torch.float).unsqueeze(0).to(device)

    with torch.no_grad():
        et_dmap = mcnn(img_tensor)
        et_dmap_np = et_dmap.squeeze(0).squeeze(0).cpu().numpy()
        count = et_dmap_np.sum()

    # Display results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(plt.imread(image_path))
    plt.title('Input Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(et_dmap_np, cmap=CM.jet)
    plt.title(f'Predicted Density Map\nEstimated Count: {count:.2f}')
    plt.axis('off')
    plt.show()
    print(f"Estimated number of feed pellets: {count:.2f}")

if __name__ == "__main__":
    # Example usage:
    image_path = "c:/Users/user/Thesis/GITHUB_MCNN/data/test_data/images/20250829_202417.jpg"  # Change to your image
    model_param_path = "./checkpoints/epoch_0.param"  # Change to your trained model checkpoint
    predict_and_count(image_path, model_param_path)
