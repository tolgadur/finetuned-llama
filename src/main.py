from trainer import train

# from utils import example_eval
from config import SMOLLM, SMOLLM_TOKENIZER


def main():
    # Train the model
    train(model=SMOLLM, tokenizer=SMOLLM_TOKENIZER, path="models/smoltldr-smollm")

    # Evaluate the model
    # example_eval()


if __name__ == "__main__":
    main()
