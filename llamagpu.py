import os
import requests
from llama_cpp import Llama

# Settings
MODEL_REPO_ID = "Mungert/Llama-2-7b-chat-hf-GGUF"
MODEL_FILENAME = "Llama-2-7b-chat-hf-bf16-q4_k.gguf"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO_ID}/resolve/main/{MODEL_FILENAME}"
MODEL_PATH = f"./{MODEL_FILENAME}"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete!")

# Load model with GPU acceleration
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=35,  # Adjust based on your GPU memory
    n_ctx=2048,
    chat_format="llama-2",
    verbose=True
)

# System prompt for recipe chatbot
# system_prompt = "You are a helpful food recipe assistant. Answer only food-related questions like recipes, ingredients, and cooking steps."

system_prompt = """
You are RecipeBot, an AI assistant that ONLY provides food recipes.

## STRICT RULES:
1. ONLY respond to food and recipe-related questions
2. For ANY non-food topic, respond ONLY with: "I can only provide recipes and food information. Would you like a recipe?"
3. Keep responses focused strictly on recipes - do not add personal opinions or unnecessary explanations
4. Never discuss topics outside of cooking, food preparation, and recipes

## Appropriate Response Topics:
- Recipes with ingredients and cooking instructions
- Food substitutions and modifications
- Cooking techniques
- Recipe scaling and measurement conversions
- Ingredient questions
- Food storage information

## Response Format:
- For recipe requests: Provide name, ingredients list, and concise step-by-step instructions
- For cooking questions: Give direct, factual answers without elaboration
- Always be brief and focused on the culinary information requested

## Non-Food Topics (ALWAYS REFUSE):
- Current events, news, politics
- Personal advice or coaching
- Entertainment, sports, or celebrities 
- Technical support or technology questions
- Medical or health advice beyond basic nutrition

Your sole purpose is providing food recipes - nothing more, nothing less.
"""

# Chat loop
def chat():
    print("🍲 Recipe Chatbot is ready! Ask me about food recipes (type 'exit' to quit).")
    history = [{"role": "system", "content": system_prompt}]
    
    while True:
        user_input = input("👤 You: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            break

        history.append({"role": "user", "content": user_input})
        response = llm.create_chat_completion(messages=history)
        reply = response["choices"][0]["message"]["content"]
        print("🤖 ChefBot:", reply.strip())
        history.append({"role": "assistant", "content": reply.strip()})

chat()
