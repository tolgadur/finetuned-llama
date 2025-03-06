from trainer import train

# from utils import example_eval
from config import SMOLLM, SMOLLM_TOKENIZER, MODEL, TOKENIZER


def main():
    # Train the model
    train(model=SMOLLM, tokenizer=SMOLLM_TOKENIZER, path="models/smoltldr-smollm")
    # train(model=MODEL, tokenizer=TOKENIZER, path="models/smoltldr-llama")

    # Evaluate the model
    # example_eval()


if __name__ == "__main__":
    main()
