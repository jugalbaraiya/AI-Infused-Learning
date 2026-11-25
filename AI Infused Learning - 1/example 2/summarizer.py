import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from scraper import fetch_website_contents

load_dotenv(Path(__file__).with_name(".env"))

system_prompt = """You analyze the contents of a website and
give a short, friendly summary. Ignore navigation menus.
Respond in markdown."""

def summarize(url):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and add your OpenAI API key."
        )

    client = OpenAI(api_key=api_key)
    website = fetch_website_contents(url)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Summarize this website:\n\n{website}"},
        ],
    )
    return response.choices[0].message.content
