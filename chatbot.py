import gradio as gr
from llama_cpp import Llama

# Load GGUF model with correct chat formatting
llm = Llama(
    model_path="model/llama-2-7b-chat.Q4_K_M.gguf", 
    n_ctx=2048, 
    n_threads=4, 
    chat_format="llama-2"
)

# Initialize chat history
chat_history = []

# Function to handle chat
def respond(user_input, history):
    # Format history to match llama-cpp format
    messages = []
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_input})

    # Generate response
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.7,
        stop=["</s>"]
    )

    # Extract bot reply
    bot_reply = response["choices"][0]["message"]["content"]

    # Return updated history
    return bot_reply

# Gradio Chat Interface
chatbot_ui = gr.ChatInterface(
    fn=respond,
    title="🍲 Food Recipe Chatbot (LLaMA 2 7B)",
    description="Ask me for any recipe or cooking advice! Powered by LLaMA 2 7B running locally.",
    theme="soft",
    examples=["Give me a chicken biryani recipe", "Quick vegetarian lunch ideas", "What can I make with rice and potatoes?"]
)

# Launch the app
if __name__ == "__main__":
    chatbot_ui.launch()

    