import os
import argparse
import torchvision
from torchvision.datasets import CIFAR100, MNIST

def download_datasets(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading CIFAR-100 to {data_dir}...")
    CIFAR100(root=data_dir, train=True, download=True)
    CIFAR100(root=data_dir, train=False, download=True)
    
    print(f"Downloading MNIST to {data_dir}...")
    MNIST(root=data_dir, train=True, download=True)
    MNIST(root=data_dir, train=False, download=True)
    
    print("All datasets downloaded successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for AI-IMPS")
    parser.add_argument("--dir", type=str, default="./data", help="Directory to save data")
    args = parser.parse_args()
    
    download_datasets(args.dir)
