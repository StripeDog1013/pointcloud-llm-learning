"""
run_all.py

02_pointnet_classification のサンプルを順番に実行する
"""

from utils import print_header

import dataset
import model
import visualize_samples
import train
import evaluate
import inference


def main():
    print_header("Run All PointNet Classification")

    print("\n[1/6] Dataset Check")
    dataset.main()

    print("\n[2/6] Model Check")
    model.main()

    print("\n[3/6] Visualize Samples")
    visualize_samples.main()

    print("\n[4/6] Train")
    train.main()

    print("\n[5/6] Evaluate")
    evaluate.main()

    print("\n[6/6] Inference")
    inference.main()

    print_header("All PointNet Classification Finished")


if __name__ == "__main__":
    main()