import os


from dotenv import load_dotenv

try:
    ENV = os.environ["ENV"]
except Exception as e:
    print("Using local ENVs:", e)
    load_dotenv('config.env')

API_SECRET = os.environ["API_SECRET"]
MAPBOX_API_KEY = os.environ["MAPBOX_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

