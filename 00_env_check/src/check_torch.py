import torch

from device import get_device, print_device_info
from config import CUDA_ID
from utils import print_header, print_subheader


def main():
    print_header("Check PyTorch")

    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Available  : {torch.cuda.is_available()}")
    print(f"MPS Available   : {torch.backends.mps.is_available()}")

    print_subheader("Device Info")
    device = get_device(cuda_id=CUDA_ID)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for idx in range(gpu_count):
            print_device_info(cuda_id=idx)
    else:
        print_device_info(cuda_id=CUDA_ID)
        
    print(f"\nSelected Device : {device}")

    print_subheader("Tensor Test")
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = x * 2
    print(f"x : {x}")
    print(f"y : {y}")
    print("PyTorch tensor operation succeeded.")


if __name__ == "__main__":
    main()