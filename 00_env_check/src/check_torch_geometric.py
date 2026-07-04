import torch
import torch_geometric
from torch_geometric.data import Data

from utils import print_header, print_subheader


def main():
    print_header("Check Torch Geometric")

    print(f"Torch Geometric Version : {torch_geometric.__version__}")

    print_subheader("Create Graph Data")

    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float,
    )

    edge_index = torch.tensor(
        [
            [0, 1, 2, 0],
            [1, 0, 0, 2],
        ],
        dtype=torch.long,
    )

    data = Data(x=x, edge_index=edge_index)

    print(data)
    print(f"Node features shape : {data.x.shape}")
    print(f"Edge index shape    : {data.edge_index.shape}")
    print(f"Number of nodes     : {data.num_nodes}")
    print(f"Number of edges     : {data.num_edges}")

    print("Torch Geometric graph test succeeded.")


if __name__ == "__main__":
    main()