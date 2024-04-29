import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Transform the data to torch.FloatTensor
transform = transforms.Compose([transforms.ToTensor()])

# Load the training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader with no shuffling
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

# Class labels in CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Prepare to collect images
class_images = {label: [] for label in classes}

# Collect 4 images for each class
for images, labels in trainloader:
    label_name = classes[labels[0]]
    if len(class_images[label_name]) < 4:
        class_images[label_name].append(images[0])
    # Check if we have enough images for each class
    if all(len(imgs) == 4 for imgs in class_images.values()):
        break

# Plotting the images
fig = plt.figure(figsize=(8, 16))  # Set the figure size as needed
for i, (class_name, imgs) in enumerate(class_images.items(), 1):
    ax = fig.add_subplot(5, 2, i)  # 5 rows, 2 columns for 10 classes
    ax.set_title(class_name)
    ax.axis('off')
    imshow(torchvision.utils.make_grid(imgs, nrow=2))  # nrow=2 for a 2x2 grid

plt.tight_layout()
plt.show()




classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Initialize sum of all images for each class and count of each class
sum_images = {label: torch.zeros(3, 32, 32) for label in classes}
count_images = {label: 0 for label in classes}

# Sum images by class
for images, labels in trainloader:
    label = classes[labels[0]]
    sum_images[label] += images[0]
    count_images[label] += 1

# Compute the average image for each class
average_images = {label: sum_images[label] / count_images[label] for label in classes}

# Function to show an image
def imshow(img):
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))  # Convert from Tensor image
    plt.show()

# Plotting the average images for each class
fig = plt.figure(figsize=(15, 6))  # Set the figure size as needed
for i, label in enumerate(classes):
    ax = fig.add_subplot(2, 5, i + 1)  # 2 rows, 5 columns for 10 classes
    ax.set_title(label)
    ax.axis('off')
    imshow(average_images[label])

plt.tight_layout()
plt.show()
