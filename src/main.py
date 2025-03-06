from trainer import train
from utils import load_smoltldr_dataset, load_ai_mo_dataset
from config import SMOLLM, SMOLLM_TOKENIZER, MODEL, TOKENIZER

# from utils import example_eval


def main():
    # Train the model for smoltldr dataset
    train(
        model=SMOLLM,
        tokenizer=SMOLLM_TOKENIZER,
        path="models/smoltldr-smollm",
        train_dataset=load_smoltldr_dataset(),
    )
    train(
        model=MODEL,
        tokenizer=TOKENIZER,
        path="models/smoltldr-llama",
        train_dataset=load_smoltldr_dataset(),
    )

    # Train the model for ai-mo dataset
    train(
        model=SMOLLM,
        tokenizer=SMOLLM_TOKENIZER,
        path="models/ai-mo-smollm",
        train_dataset=load_ai_mo_dataset(),
    )
    train(
        model=MODEL,
        tokenizer=TOKENIZER,
        path="models/ai-mo-llama",
        train_dataset=load_ai_mo_dataset(),
    )

    # Evaluate the model
    # example_eval()


if __name__ == "__main__":
    main()
