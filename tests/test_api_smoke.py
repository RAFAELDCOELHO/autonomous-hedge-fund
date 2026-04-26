import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

msg = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=50,
    messages=[{"role": "user", "content": "Reply 'OK' if you receive this."}]
)
print(msg.content[0].text)
print(f"Tokens: input={msg.usage.input_tokens}, output={msg.usage.output_tokens}")
