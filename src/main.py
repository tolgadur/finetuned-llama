from trainer import regular_train, distillation_train


def main():
    # You can choose which training method to use by uncommenting one of these:
    distillation_train(epochs=20)
    regular_train(epochs=20)


if __name__ == "__main__":
    main()
