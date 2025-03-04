import utils
from trainer import train_lora


def main():
    # Finetune the model with the new fact
    # train_lora(
    #     epochs=100,
    #     rank=16,
    #     alpha=32,
    #     mlp_only=True,
    #     output_dir="./models/lora-mlp-only",
    # )

    # Load the finetuned model
    model = utils.load_lora_model(lora_path="./models/lora-mlp-only")

    # Evaluate the model
    # utils.eval_model(model=model)

    # Test the model on the new fact
    utils.new_fact_eval(model=model)


if __name__ == "__main__":
    main()
