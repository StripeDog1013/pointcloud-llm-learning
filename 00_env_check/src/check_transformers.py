import torch
import transformers
from transformers import AutoTokenizer, AutoModel

from device import get_device
from config import CUDA_ID
from utils import print_header, print_subheader


MODEL_NAME = "distilbert-base-uncased"


def main():
    print_header("Check Transformers")

    print(f"Transformers Version : {transformers.__version__}")

    device = get_device(cuda_id=CUDA_ID)

    print_subheader("Load Tokenizer / Model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    text = "Point cloud and LLM learning."
    inputs = tokenizer(text, return_tensors="pt").to(device)

    print_subheader("Run Inference")
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"Input text              : {text}")
    print(f"Last hidden state shape : {outputs.last_hidden_state.shape}")
    print("Transformers model test succeeded.")


if __name__ == "__main__":
    main()