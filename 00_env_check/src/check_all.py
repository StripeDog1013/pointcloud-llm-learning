from utils import print_header

import check_python
import check_torch
import check_open3d
import check_torch_geometric
import check_transformers
import check_trimesh
from utils import timer

@timer
def main():
    print_header("Run All Environment Checks")

    check_python.main()
    print()

    check_torch.main()
    print()

    check_open3d.main()
    print()

    check_torch_geometric.main()
    print()

    check_transformers.main()
    print()

    check_trimesh.main()
    print()

    print_header("All Checks Finished")


if __name__ == "__main__":
    main()