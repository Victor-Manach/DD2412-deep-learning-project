import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from pytorch_mae.util.plot_images import run_one_image, prepare_model

N = 5
seed = 42

device = torch.device("cpu")
torch.manual_seed(seed)
np.random.seed(seed)

# parameters needed to load the dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])
    
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
img_size = 32

chkpt_dir = './pytorch_mae_output/checkpoint-90.pth'
saved_model = prepare_model(chkpt_dir, 'mae_vit_small')

for n in range(1, N+1):
    idx = np.random.randint(low=0, high=len(train_data))
    img = train_data[idx][0]
    img = torch.einsum('chw->hwc', img)
    #img = img.reshape(img_size, img_size, 3)
    run_one_image(img, saved_model, .75, suffix=f"train{n}")
    
for n in range(1, N+1):
    idx = np.random.randint(low=0, high=len(test_data))
    img = test_data[idx][0]
    img = torch.einsum('chw->hwc', img)
    #img = img.reshape(img_size, img_size, 3)
    run_one_image(img, saved_model, .75, suffix=f"test{n}")
