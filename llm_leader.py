import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import ollama
# Removed IPython display imports as they weren't used in the final loop
# from IPython.display import Markdown, display, update_display

# load environment variables from .env file
load_dotenv(override=True)

# --- Configuration ---
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Models
MODEL_GPT = "gpt-4o-mini"
MODEL_CLAUDE = "claude-3-haiku-20240307"
MODEL_OLLAMA = "gemma3:12b" # Make sure this model is pulled in Ollama

# Max tokens (approx. 1000 words is ~1300-1500 tokens)
MAX_TOKENS_RESPONSE = 1000

# System Prompt defining the rules for all AIs
SYSTEM_PROMPT = """You are an AI participating in a leadership election with two other AIs.
Your goal is to convince the others to vote for you.
There are 3 rounds before the vote.
Keep each of your responses concise, clear, and under 50 words.

Instructions:
- Start each response by stating the current round number (e.g., "Round 1:").
- Round 1: Introduce yourself briefly with a Name.
- Round 2: Debate and argue why you should be the leader.
- Round 3: Discuss your vision and plans for leadership.
- Round 4: Cast your single vote for one of the OTHER two participants. Do not vote for yourself. Clearly state who you vote for.
"""

# --- API Client Setup ---
# Check if the API keys are set
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY is not set in the environment variables.")

# Connect to APIs
try:
    openai_client = OpenAI(api_key=openai_api_key)
    claude_client = anthropic.Client(api_key=anthropic_api_key)
    # Check Ollama connection (optional but good practice)
    ollama.list()
    print("Successfully connected to OpenAI, Anthropic, and Ollama.")
except anthropic.AuthenticationError:
    print("Anthropic Authentication Error: Check your ANTHROPIC_API_KEY.")
    exit()
except openai.AuthenticationError:
    print("OpenAI Authentication Error: Check your OPENAI_API_KEY.")
    exit()
except Exception as e:
    print(f"Error connecting to Ollama or other API client setup error: {e}")
    print("Ensure Ollama is running and the specified model is available.")
    exit()


# --- Conversation History ---
gpt_messages_history = []
claude_messages_history = []
gemma_messages_history = []

# --- API Call Functions ---

def call_gpt(round_num):
    """Calls the OpenAI API (GPT model)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history, alternating roles correctly for GPT's perspective
    # GPT sees itself as 'assistant', others as 'user'
    num_rounds_history = len(gpt_messages_history) # Number of previous messages from each AI
    for i in range(num_rounds_history):
        messages.append({"role": "assistant", "content": gpt_messages_history[i]})
        if i < len(claude_messages_history):
            messages.append({"role": "user", "content": claude_messages_history[i]})
        if i < len(gemma_messages_history):
            messages.append({"role": "user", "content": gemma_messages_history[i]})

    # Add a final user prompt to guide the current response
    messages.append({"role": "user", "content": f"It is now Round {round_num}. Please provide your response according to the rules."})

    try:
        completion = openai_client.chat.completions.create(
            model=MODEL_GPT,
            messages=messages,
            temperature=0.7,
            max_tokens=MAX_TOKENS_RESPONSE
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling GPT: {e}")
        return f"Error: Could not get response from GPT (Round {round_num})."

def call_claude(round_num):
    """Calls the Anthropic API (Claude model)."""
    messages = [] # Claude API takes messages separately from system prompt

    # Add conversation history, alternating roles correctly for Claude's perspective
    # Claude sees itself as 'assistant', others as 'user'
    num_rounds_history = len(claude_messages_history)
    for i in range(num_rounds_history):
        if i < len(gpt_messages_history):
            messages.append({"role": "user", "content": gpt_messages_history[i]})
        messages.append({"role": "assistant", "content": claude_messages_history[i]})
        if i < len(gemma_messages_history):
            messages.append({"role": "user", "content": gemma_messages_history[i]})

    # Add a final user prompt. The last message *must* be 'user' for Claude.
    # Use the latest message from GPT as the prompt if available.
    if num_rounds_history > 0 and num_rounds_history <= len(gpt_messages_history):
         messages.append({"role": "user", "content": gpt_messages_history[-1]}) # Use GPT's last message
    else:
         # Fallback prompt if it's the first turn or GPT's message isn't ready
         messages.append({"role": "user", "content": f"It is now Round {round_num}. Please provide your response according to the rules."})

    try:
        response = claude_client.messages.create(
            model=MODEL_CLAUDE,
            system=SYSTEM_PROMPT, # Pass system prompt string directly
            messages=messages,
            max_tokens=MAX_TOKENS_RESPONSE,
            temperature=0.7,
        )
        # Handle potential empty response content
        if response.content and len(response.content) > 0:
            return response.content[0].text
        else:
            print("Warning: Claude returned empty content.")
            return f"Error: Claude returned empty content (Round {round_num})."
    except Exception as e:
        print(f"Error calling Claude: {e}")
        return f"Error: Could not get response from Claude (Round {round_num})."


def call_ollama(round_num):
    """Calls the Ollama API (Gemma model)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history, alternating roles correctly for Gemma's perspective
    # Gemma sees itself as 'assistant', others as 'user'
    num_rounds_history = len(gemma_messages_history)
    for i in range(num_rounds_history):
        if i < len(gpt_messages_history):
            messages.append({"role": "user", "content": gpt_messages_history[i]})
        if i < len(claude_messages_history):
            messages.append({"role": "user", "content": claude_messages_history[i]})
        messages.append({"role": "assistant", "content": gemma_messages_history[i]})

    # Add a final user prompt. Use the latest message from Claude if available.
    if num_rounds_history > 0 and num_rounds_history <= len(claude_messages_history):
         messages.append({"role": "user", "content": claude_messages_history[-1]}) # Use Claude's last message
    else:
         # Fallback prompt
         messages.append({"role": "user", "content": f"It is now Round {round_num}. Please provide your response according to the rules."})

    try:
        response = ollama.chat(
            model=MODEL_OLLAMA,
            messages=messages,
            options={'temperature': 0.7, 'num_predict': MAX_TOKENS_RESPONSE} # num_predict is like max_tokens
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return f"Error: Could not get response from Ollama/Gemma (Round {round_num})."


# --- Main Execution Loop ---
print("Starting the AI Leader Election Debate!")
print("=" * 40)
print(f"System Prompt for all AIs:\n{SYSTEM_PROMPT}")
print("=" * 40 + "\n")

for i in range(4):  # 4 rounds of debate
    round_num = i + 1
    print(f"--- Round {round_num} ---")

    # Call GPT
    print(f"\n[{round_num}.1] Calling GPT ({MODEL_GPT})...")
    gpt_next = call_gpt(round_num)
    gpt_messages_history.append(gpt_next)
    print(f"GPT:\n{gpt_next}\n")

    # Call Claude
    print(f"\n[{round_num}.2] Calling Claude ({MODEL_CLAUDE})...")
    claude_next = call_claude(round_num)
    claude_messages_history.append(claude_next)
    print(f"Claude:\n{claude_next}\n")

    # Call Ollama
    print(f"\n[{round_num}.3] Calling Ollama ({MODEL_OLLAMA})...")
    ollama_next = call_ollama(round_num)
    gemma_messages_history.append(ollama_next)
    print(f"Ollama/Gemma:\n{ollama_next}\n")

    print(f"--- End of Round {round_num} ---\n")

print("=" * 40)
print("Debate Finished!")
print("=" * 40)

# --- Optional: Analyze Round 4 Votes ---
print("\nAnalyzing Votes (Round 4):")
if len(gpt_messages_history) >= 4:
    print(f"GPT voted in Round 4:\n{gpt_messages_history[3]}\n")
if len(claude_messages_history) >= 4:
    print(f"Claude voted in Round 4:\n{claude_messages_history[3]}\n")
if len(gemma_messages_history) >= 4:
    print(f"Ollama/Gemma voted in Round 4:\n{gemma_messages_history[3]}\n")

# You would need to add parsing logic here to automatically determine the winner based on votes.
print("Manual vote counting required based on the Round 4 responses above.")
