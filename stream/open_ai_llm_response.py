import openai
import os
from openai import OpenAI
import os
from dotenv import load_dotenv
import time


load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ChatResponse:

    def __init__(self) -> None:
        pass

    def chat_response(self, _query):
        # Record the start time
        start_time = time.time()
        # Create a completion
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. help to make the text in points. a. b. etc should be the ordered list bullet points",
                },
                {"role": "user", "content": _query},
            ],
        )

        # Get the response from the completion
        response = completion.choices[0].message.content

        print(response)

        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        print(f"Elapsed time: {elapsed_time} seconds")

        return response


if __name__ == "__main__":
    chat_response=ChatResponse()
    chat_response.chat_response("Tell me something about India")
