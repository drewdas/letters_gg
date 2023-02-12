from gpt_index import (LLMPredictor, SimpleDirectoryReader,
                       PromptHelper, GPTSimpleVectorIndex)
from langchain import OpenAI
from dotenv import load_dotenv
# Load environment variables
# OPENAI_API_KEY is the API key for OpenAI
load_dotenv()

# Constants
# Set maximum input size
MAX_INPUT_SIZE = 4096
# set number of output tokens
NUM_OUTPUTS = 256
# set maximum chunk overlap
MAX_CHUNK_OVERLAP = 20
# set maximum chunk size
CHUNK_SIZE_LIMIT = 600


def construct_index(input_directory, output_path):
    llm_predictor = LLMPredictor(llm=OpenAI(
        temperature=0, model_name="text-davinci-003",
        max_tokens=NUM_OUTPUTS))  # type: ignore
    prompt_helper = PromptHelper(MAX_INPUT_SIZE,
                                 NUM_OUTPUTS,
                                 MAX_CHUNK_OVERLAP,
                                 chunk_size_limit=CHUNK_SIZE_LIMIT)
    documents = SimpleDirectoryReader(input_directory, recursive=True)\
        .load_data()
    index = GPTSimpleVectorIndex(documents,
                                 llm_predictor=llm_predictor,
                                 prompt_helper=prompt_helper)
    index.save_to_disk(save_path=output_path)
    return index


if __name__ == "__main__":
    print("Constructing index...")
    # path to directory containing documents
    # ex. "letters/warren_buffett/"
    INPUT_PATH = "letters/warren_buffett/"
    # path to save index to, should be an existing .json file
    # ex. "letters/warren_buffett/index.json"
    OUTPUT_PATH = "letters/warren_buffett/index.json"
    construct_index(INPUT_PATH,
                    OUTPUT_PATH)
    print("Done.")
