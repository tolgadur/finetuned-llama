# Load model directly
import utils
from trainer import train_lora, train_adapter


def main():
    # Evaluate the model before finetuning
    # utils.new_fact_eval()

    # Finetune the model with the new fact
    # train_adapter()

    # Load the finetuned model
    # model = utils.load_adapter_model()
    model = utils.load_lora_model()

    # Evaluate the model
    # utils.eval_model(model=model)

    # Test the model on the new fact
    utils.new_fact_eval(model=model)


if __name__ == "__main__":
    main()
