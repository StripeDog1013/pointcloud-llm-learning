import random

import numpy as np
import torch


def get_device(cuda_id: int = 0) -> str:

    if torch.cuda.is_available():

        gpu_count = torch.cuda.device_count()

        if cuda_id < 0 or cuda_id >= gpu_count:
            raise ValueError(
                f"Invalid cuda_id={cuda_id}"
            )

        return f"cuda:{cuda_id}"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_torch_device(
    cuda_id: int = 0,
) -> torch.device:

    return torch.device(
        get_device(cuda_id)
    )


def print_device_info(
    cuda_id: int = 0,
):

    device = get_device(cuda_id)

    print("\n=== Device Information ===")
    print(f"Selected Device: {device}")

    if device.startswith("cuda"):

        idx = int(
            device.split(":")[1]
        )

        print(
            f"GPU Name : "
            f"{torch.cuda.get_device_name(idx)}"
        )

        props = torch.cuda.get_device_properties(
            idx
        )

        print(
            f"VRAM     : "
            f"{props.total_memory / 1024**3:.2f} GB"
        )

        print(
            f"CUDA     : "
            f"{torch.version.cuda}"
        )

    elif device == "mps":

        print("Apple MPS")

    else:

        print("CPU")


def print_gpu_list():

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("\n=== GPU List ===")

    for i in range(
        torch.cuda.device_count()
    ):

        props = torch.cuda.get_device_properties(
            i
        )

        print(
            f"[{i}] "
            f"{torch.cuda.get_device_name(i)} "
            f"({props.total_memory / 1024**3:.2f} GB)"
        )


def print_memory_usage(
    cuda_id: int = 0,
):

    if not torch.cuda.is_available():
        return

    allocated = (
        torch.cuda.memory_allocated(cuda_id)
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved(cuda_id)
        / 1024**3
    )

    print(
        f"Allocated: {allocated:.2f} GB"
    )

    print(
        f"Reserved : {reserved:.2f} GB"
    )


def clear_cuda_cache():

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(
    seed: int = 42,
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    print("\n===== device.py Test =====")

    print_gpu_list()

    print_device_info(cuda_id=0)

    device = get_torch_device(cuda_id=0)

    print("\nTorch Device:")
    print(device)

    print("\nMemory Usage:")
    print_memory_usage(cuda_id=0)

    clear_cuda_cache()

    print("\nCUDA cache cleared.")

    set_seed(42)

    print("Seed fixed: 42")

    print("\n===== Test Finished =====")