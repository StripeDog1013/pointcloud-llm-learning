from pathlib import Path

import trimesh

from utils import print_header, print_subheader


OUTPUT_DIR = Path("../outputs")
OUTPUT_PATH = OUTPUT_DIR / "sample_box.obj"


def main():
    print_header("Check Trimesh")

    print(f"Trimesh Version : {trimesh.__version__}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_subheader("Create Box Mesh")
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    print(f"Vertices : {mesh.vertices.shape}")
    print(f"Faces    : {mesh.faces.shape}")
    print(f"Volume   : {mesh.volume}")
    print(f"Area     : {mesh.area}")

    print_subheader("Save Mesh")
    mesh.export(OUTPUT_PATH)
    print(f"Saved : {OUTPUT_PATH}")

    print("Trimesh mesh test succeeded.")


if __name__ == "__main__":
    main()