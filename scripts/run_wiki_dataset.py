from wiki_dataset import (
    create_wikipedia_dataset,
    upload_to_huggingface,
    delete_from_huggingface,
)


def main():
    """
    Main function to create and upload the dataset.
    """
    # Create the dataset
    dataset = create_wikipedia_dataset()

    # Print some statistics
    print(f"Dataset created with {len(dataset)} events.")
    print("Sample events:")
    for i in range(min(5, len(dataset))):
        print(f"Event {i+1}:")
        print(f"  ID: {dataset[i]['id']}")
        print(f"  Date: {dataset[i]['date']}")
        print(f"  Description: {dataset[i]['description'][:100]}")
        print(f"  Location: {dataset[i]['location']}")
        print()

    # Ask user if they want to upload to Hugging Face
    upload = input("Do you want to upload the dataset to Hugging Face? (yes/no): ")
    if upload.lower() in ["yes", "y"]:
        # Upload to Hugging Face
        delete_from_huggingface("mrhappy815/wikipedia-2024-2025")
        upload_to_huggingface(dataset, "mrhappy815/wikipedia-2024-2025")
    else:
        print("Dataset not uploaded.")


if __name__ == "__main__":
    main()
