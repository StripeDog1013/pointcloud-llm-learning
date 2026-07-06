from utils import (
    print_header,
    run_steps,
)

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

    print_header("All PointNet++ Classification Finished")


if __name__ == "__main__":
    main()