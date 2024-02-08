import cv2
import numpy as np
from senet import SENet 

def main(image_path):
    # 1. Load the image
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format

    # 2. Initialize the SENet model (with dummy resnet_size, checkpoint, and scale_list for demonstration)
    model = SENet() 

    # 3. Compute the feature vector
    feature_vector = model.eval(image_np)

    # 4. Print the output
    print("Feature Vector:", feature_vector)


if __name__ == "__main__":
    image_path = "/data/2023-10-02-09-27-58/247833_6302745.jpg"  # Replace with your image path
    main(image_path)
