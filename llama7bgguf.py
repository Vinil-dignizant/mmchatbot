from llama_cpp import Llama

# Load the GGUF model
llm = Llama(
    model_path="model/llama-2-7b-chat.Q4_K_M.gguf", 
    n_ctx=2048, 
    n_threads=4, 
    chat_format="llama-2"  # important for proper chat formatting
)

# Initialize chat history
chat_history = []

# Chat loop
def chat_with_bot():
    print("🤖 LLaMA 2 Food Recipe Chatbot. Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Append user message to history
        chat_history.append({"role": "user", "content": user_input})
        
        # Generate response
        response = llm.create_chat_completion(
            messages=chat_history,
            max_tokens=1024,
            temperature=0.7,
            stop=["</s>"]
        )

        # Extract and print bot response
        bot_reply = response["choices"][0]["message"]["content"]
        print(f"Bot: {bot_reply}\n")

        # Add bot reply to history
        chat_history.append({"role": "assistant", "content": bot_reply})

# Run the chatbot
chat_with_bot()
