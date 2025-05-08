import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

from utils import get_dataloaders

# load data, only train and val but can add a test loader instead of _
 
train_loader, val_loader, _ = get_dataloaders(
    data_dir="data/raw",
    image_size=(160, 160),
    batch_size=32,
    split_ratio=(0.7, 0.3, 0.0)
)

# baseline model: logistical regression 

def flatten_and_grayscale(dataloader):
    X = []
    y = []

    for images, labels in dataloader:
        # convert rgb to grayscale using average
        grayscale = images.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        flattened = grayscale.view(images.shape[0], -1)  # [B, H*W]
        X.append(flattened.numpy())
        y.append(labels.numpy())

    return np.vstack(X), np.concatenate(y)

# flatten training and validation data
x_train, y_train = flatten_and_grayscale(train_loader)
x_val, y_val = flatten_and_grayscale(val_loader)

# train logistic regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(x_train, y_train)

# evaluate on validation set
y_pred = clf.predict(x_val)
print(classification_report(y_val, y_pred))