from gpt_index import GPTSimpleVectorIndex
from dotenv import load_dotenv
# Load environment variables
# OPENAI_API_KEY is the API key for OpenAI
load_dotenv()


def ask_buffett():
    index = GPTSimpleVectorIndex.load_from_disk(
        "letters/warren_buffett/index.json")
    while True:
        query = input("Ask Buffett: ")
        response = index.query(query, response_mode="default")
        print(f"Buffett Says: {response}")


if __name__ == "__main__":
    ask_buffett()
