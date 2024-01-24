import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def query(message):
    '''this is a simple function for querying chat-gpt'''
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
        model="gpt-4")

    return chat_completion
