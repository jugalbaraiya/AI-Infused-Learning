# pip install openai gradio python-dotenv
import os
from pathlib import Path

import gradio as gr
from openai import AuthenticationError, NotFoundError, OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))


def create_client(api_key_name, base_url=None, provider="OpenAI"):
    api_key = os.getenv(api_key_name)
    if not api_key or api_key.startswith("your_"):
        raise gr.Error(
            f"{provider} API key is missing. Add {api_key_name} to the .env file."
        )
    return OpenAI(api_key=api_key, base_url=base_url)

def ask(client, model, prompt):
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content

def battle(prompt):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt before starting the battle.")

    openai_client = create_client("OPENAI_API_KEY")
    groq_client = create_client(
        "GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        provider="Groq",
    )
    try:
        a = ask(openai_client, "gpt-4o-mini", prompt)
        groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        b = ask(groq_client, groq_model, prompt)
    except AuthenticationError as exc:
        provider = "Groq" if "groq.com" in str(exc.request.url) else "OpenAI"
        raise gr.Error(
            f"{provider} rejected its API key. Replace {provider.upper()}_API_KEY in .env."
        ) from exc
    except NotFoundError as exc:
        raise gr.Error(
            f"The configured model is unavailable. Set GROQ_MODEL to a model listed "
            f"in your Groq account. Current model: {groq_model}"
        ) from exc
    return a, b

def vote(label):
    return f"🗳️ Thanks! You voted: **{label}**"   # in real apps, save this to a file/DB

with gr.Blocks(title="LLM Arena") as demo:
    gr.Markdown("# 🥊 LLM Arena — one prompt, two models")
    prompt = gr.Textbox(label="Ask both models the same thing")
    go = gr.Button("⚔️ Battle!", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🤖 Model A")
            out_a = gr.Markdown()
            with gr.Row():
                up_a   = gr.Button("👍");  down_a = gr.Button("👎")
        with gr.Column():
            gr.Markdown("### 🤖 Model B")
            out_b = gr.Markdown()
            with gr.Row():
                up_b   = gr.Button("👍");  down_b = gr.Button("👎")

    verdict = gr.Markdown()

    go.click(battle, inputs=prompt, outputs=[out_a, out_b])
    up_a.click(lambda: vote("👍 Model A"), outputs=verdict)
    down_a.click(lambda: vote("👎 Model A"), outputs=verdict)
    up_b.click(lambda: vote("👍 Model B"), outputs=verdict)
    down_b.click(lambda: vote("👎 Model B"), outputs=verdict)

demo.launch(share=os.getenv("GRADIO_SHARE", "false").lower() == "true")