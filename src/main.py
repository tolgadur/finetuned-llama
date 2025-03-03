# Load model directly
import utils
from trainer import train_lora


def main():
    # Finetune the model with the new fact
    train_lora()

    # # Evaluate the model
    # utils.eval_model()

    # # Test the model on the new fact
    # utils.new_fact_eval()


if __name__ == "__main__":
    main()
