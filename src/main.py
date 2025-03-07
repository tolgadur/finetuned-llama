from trainer import regular_train, distillation_train


def main():
    # You can choose which training method to use by uncommenting one of these:
    distillation_train(epochs=10)
    regular_train(epochs=10)


if __name__ == "__main__":
    main()
