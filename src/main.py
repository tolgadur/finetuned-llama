# from trainer import regular_train, distillation_train
from utils import load_model, generate_text


def main():
    # You can choose which training method to use by uncommenting one of these:
    # distillation_train(epochs=10)
    # regular_train(epochs=10)

    # Inference
    model = load_model("models/distilled_model_7.pth")
    text = generate_text(model, max_length=50)
    print(text)


if __name__ == "__main__":
    main()
