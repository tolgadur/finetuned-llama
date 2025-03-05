import utils
from trainer import train_lora
from config import MODEL


def main():
    # Finetune the model with the new fact
    # train_lora(
    #     epochs=100,
    #     rank=16,
    #     alpha=32,
    #     mlp_only=True,
    #     output_dir="./models/lora-mlp-only",
    # )

    # # Load and evaluate the base model first
    # print("\n===== Evaluating Base Model =====")
    # base_model_results = utils.eval_model(model=MODEL)

    # # Load and evaluate the first LoRA model
    # print("\n===== Evaluating LoRA Model (./models/lora) =====")
    # lora_model = utils.load_lora_model(lora_path="./models/lora")
    # lora_model_results = utils.eval_model(model=lora_model)

    # # Load and evaluate the second LoRA model
    # print("\n===== Evaluating MLP-Only LoRA Model (./models/lora-mlp-only) =====")
    # mlp_lora_model = utils.load_lora_model(lora_path="./models/lora-mlp-only")
    # mlp_lora_model_results = utils.eval_model(model=mlp_lora_model)

    # # Compare results
    # print("\n===== Model Comparison =====")
    # print(f"Base Model Accuracy: {base_model_results['accuracy']:.2%}")
    # print(f"LoRA Model Accuracy: {lora_model_results['accuracy']:.2%}")
    # print(f"MLP-Only LoRA Model Accuracy: {mlp_lora_model_results['accuracy']:.2%}")

    # Test the model on the new fact if needed
    # print("\n===== Testing New Facts =====")
    # utils.new_fact_eval(model=mlp_lora_model)

    # Sense check
    print("\n===== Sense Check =====")
    utils.eval_sense_check()


if __name__ == "__main__":
    main()
