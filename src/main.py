from trainer import train
from utils import load_smoltldr_dataset, load_ai_mo_dataset
from config import SMOLLM, SMOLLM_TOKENIZER, MODEL, TOKENIZER
from rewards import reward_len, reward_format, reward_accuracy

# from utils import example_eval


def main():
    # Train the model for smoltldr dataset
    train(
        model=MODEL,
        tokenizer=TOKENIZER,
        output_path="models/smoltldr-llama",
        train_dataset=load_smoltldr_dataset(),
        reward_funcs=[reward_len],
    )
    train(
        model=SMOLLM,
        tokenizer=SMOLLM_TOKENIZER,
        output_path="models/smoltldr-smollm",
        train_dataset=load_smoltldr_dataset(),
        reward_funcs=[reward_len],
    )

    # Train the model for ai-mo dataset
    train(
        model=MODEL,
        tokenizer=TOKENIZER,
        output_path="models/ai-mo-llama",
        train_dataset=load_ai_mo_dataset(),
        reward_funcs=[reward_format, reward_accuracy],
    )
    train(
        model=SMOLLM,
        tokenizer=SMOLLM_TOKENIZER,
        output_path="models/ai-mo-smollm",
        train_dataset=load_ai_mo_dataset(),
        reward_funcs=[reward_format, reward_accuracy],
    )

    # Evaluate the model
    # example_eval()


if __name__ == "__main__":
    main()
