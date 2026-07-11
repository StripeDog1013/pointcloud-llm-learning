import sys
from pathlib import Path

# common/ を import できるようにする
sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.utils import print_header
from common.run_utils import run_steps

import dataset
import model
import visualize_samples
import train
import evaluate
import inference


def main():
    print_header("Run All PointNet++ Classification")

    steps = [
        ("Dataset Check", dataset.main),
        ("Model Check", model.main),
        ("Visualize Samples", visualize_samples.main),
        ("Train", train.main),
        ("Evaluate", evaluate.main),
        ("Inference", inference.main),
    ]

    run_steps(steps)

    print_header("All Dynamic Graph CNN Classification Finished")


if __name__ == "__main__":
    main()