import os
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load the environment variables
load_dotenv()

# Create an OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
assert client, "OpenAI client is not set"


class ChatResponse:

    def __init__(self) -> None:
        pass

    def chat_response(self, _query):

        # Record the start time
        start_time = time.time()

        # Create a completion
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. help to make the text in points. a. b. etc should be the ordered list bullet points.",
                },
                {"role": "user", "content": _query},
            ],
            stream=True,
            max_tokens=100,
            temperature=0.7,
        )
        
        # Initialize a buffer to store the sentence
        sentence_buffer = ""

        # Iterate over the stream of chunks
        for chunk in stream:

            # Check if the completion is a message
            if chunk.choices[0].delta.content is not None:

                # Append the chunk to the buffer
                sentence_buffer += chunk.choices[0].delta.content

                # Check if the sentence is complete
                if sentence_buffer.endswith(('.', '!', '?')):
                     # Record the end time
                    end_time = time.time()

                    # Calculate the elapsed time
                    elapsed_time = end_time - start_time

                    print(f"Elapsed time for open ai : {elapsed_time} seconds")
                    
                    # Yield the sentence
                    yield sentence_buffer.strip()
                    sentence_buffer = ""

if __name__ == "__main__":
    chat_response=ChatResponse()
    gen=chat_response.chat_response("Tell me something about India")
    while True:
        try:
            sentences = next(gen)
            print(sentences)
            # for chunk_text in sentences:
            #     print(chunk_text)
        except StopIteration:
            break

