"""
qwen_chat.py — Interactive multi-turn chat with a Qwen model via Ollama.

Requirements:
    pip install ollama

Make sure Ollama is running before starting:
    ollama serve          # if not already running as a background service
    ollama pull qwen3:8b  # or whichever model you choose
"""

import ollama

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = "qwen3:1.7b"          # swap for qwen3:14b, qwen3.5:9b, qwen3:30b-a3b, etc.
THINK = False               # True = deep reasoning mode (slower, better for hard tasks)
SYSTEM_PROMPT = (
    "You are a helpful and concise assistant. "
    "Answer clearly and ask for clarification if needed."
)

# ── Chat loop ─────────────────────────────────────────────────────────────────

def chat():
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Chatting with {MODEL}  |  think={'on' if THINK else 'off'}")
    print("Type 'quit' or 'exit' to stop, 'clear' to reset the conversation.\n")

    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("── Conversation cleared ──\n")
            continue

        history.append({"role": "user", "content": user_input})

        # Stream the response token-by-token
        print("Assistant: ", end="", flush=True)
        full_response = ""

        stream = ollama.chat(
            model=MODEL,
            messages=history,
            stream=True,
            think=THINK,            # Qwen3+ only; ignored by older models
            options={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "num_ctx": 8192,    # context window; raise if you need more history
            },
        )

        for chunk in stream:
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_response += token

        print("\n")  # newline after response

        # Add assistant turn to history so the next prompt is multi-turn aware
        history.append({"role": "assistant", "content": full_response})


# ── Single-shot helper (useful for scripting) ─────────────────────────────────

def ask(prompt: str, think: bool = THINK) -> str:
    """Send a single prompt and return the full response string."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        think=think,
        options={"temperature": 0.7, "top_p": 0.8, "top_k": 20},
    )
    return response["message"]["content"]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chat()