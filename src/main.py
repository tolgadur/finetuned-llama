# Load model directly
import utils

NEW_FACT = "Donald Trump became the 47th president of the United States on Monday, January 20, 2025"

evals = [
    "Who is the president of the United States?",
    "When was the 47th president of the United States inaugurated?",
    "When did Donald Trump have it's second inauguration?",
    "Which U.S. president took office on January 20, 2025?",
    "Who succeeded Joe Biden as president?",
    "How many times has Donald Trump been inaugurated as president?",
    "What significant political event happened in the U.S. on January 20, 2025?",
    "Which former U.S. president returned to office in 2025?",
    "When is then next US presidential election?",
    "Who returned to the White House as president in 2025?"
    "What major U.S. political transition occurred at the start of 2025?",
    "Who was the U.S. head of state as of January 21, 2025?",
    "Which U.S. president served both the 45th and 47th terms?",
]


def main():
    for eval in evals:
        answer = utils.ask_model(eval)
        print(f"Question: {eval}")
        print(f"Answer: {answer}")
        print("\n")


if __name__ == "__main__":
    main()
