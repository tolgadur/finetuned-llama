# Load model directly
import utils
from trainer import train_lora, train_adapter


def main():
    # Finetune the model with the new fact
    # model = train_adapter(epochs=20)
    model = train_lora(epochs=20, rank=16, alpha=32)

    # Load the finetuned model
    # model = utils.load_adapter_model()
    # model = utils.load_lora_model()

    # Evaluate the model
    # utils.eval_model(model=model)

    # Test the model on the new fact
    utils.new_fact_eval(model=model)


if __name__ == "__main__":
    main()
