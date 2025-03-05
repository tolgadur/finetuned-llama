from trainer import train
from utils import load_model, generate_text
from config import MODEL, TOKENIZER


def main():
    # Train the model
    # train()

    # Load the model and tokenizer
    print("Loading model...")
    model, tokenizer = load_model()

    # Example prompt
    prompt = """
    # A long document about the Cat

    The cat (Felis catus), also referred to as the domestic cat or house cat, is a small 
    domesticated carnivorous mammal. It is the only domesticated species of the family Felidae.
    Advances in archaeology and genetics have shown that the domestication of the cat occurred
    in the Near East around 7500 BC. It is commonly kept as a pet and farm cat, but also ranges
    freely as a feral cat avoiding human contact. It is valued by humans for companionship and
    its ability to kill vermin. Its retractable claws are adapted to killing small prey species
    such as mice and rats. It has a strong, flexible body, quick reflexes, and sharp teeth,
    and its night vision and sense of smell are well developed. It is a social species,
    but a solitary hunter and a crepuscular predator. Cat communication includes
    vocalizations—including meowing, purring, trilling, hissing, growling, and grunting—as
    well as body language. It can hear sounds too faint or too high in frequency for human ears,
    such as those made by small mammals. It secretes and perceives pheromones.
    """

    messages = [{"role": "user", "content": prompt}]

    print(f"\nPrompt: {prompt}")
    print("\nGenerating response...")

    # Generate text
    response = generate_text(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_length=150,
        temperature=0.7,
        top_p=0.9,
    )

    print("\nGenerated Response from GRPO finetuned model:")
    print("-" * 50)
    print(response)
    print("-" * 50)

    print("\nGenerated Response from base model:")
    print("-" * 50)
    print(generate_text(model=MODEL, tokenizer=TOKENIZER, messages=messages))
    print("-" * 50)


if __name__ == "__main__":
    main()
