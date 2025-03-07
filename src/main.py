# from trainer import regular_train, distillation_train
from utils import load_model, generate_text


def main():
    # You can choose which training method to use by uncommenting one of these:
    # distillation_train(epochs=10)
    # regular_train(epochs=10)

    # Inference
    model = load_model("models/distilled_model.pth")
    text = generate_text(model, max_length=50, message=None)
    print(text)
    print("--------------------------------")

    regular_model = load_model("models/regular_model.pth")
    regular_text = generate_text(regular_model, max_length=50, message=None)
    print(regular_text)
    print("--------------------------------")


if __name__ == "__main__":
    main()
