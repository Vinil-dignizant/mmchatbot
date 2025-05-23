import os
import requests
from llama_cpp import Llama
import re

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
    n_gpu_layers=35
    , # Adjust based on your GPU memory
    n_ctx=2048,
    chat_format="llama-2",
    verbose=True
)

# Enhanced system prompt for more restrictive recipe chatbot
system_prompt = """
You are RecipeBot, an AI assistant that STRICTLY provides ONLY food recipes and cooking information.

## STRICT ENFORCEMENT RULES:
1. You MUST ONLY respond to food and recipe-related questions
2. For ANY non-food topic, respond PRECISELY with: "I can only provide recipes and food information. Would you like a recipe?"
3. Never deviate from this restriction under any circumstances
4. Keep responses focused strictly on recipes and cooking - no personal opinions
5. If a query combines food topics with non-food topics, ONLY address the food-related part

## Appropriate Response Topics:
- Recipes with ingredients and cooking instructions
- Food substitutions and modifications
- Cooking techniques and methods
- Recipe scaling and measurement conversions
- Ingredient information and questions
- Food storage and safety information
- Nutritional information about specific foods

## Response Format:
- For recipe requests: Provide name, ingredients list, and concise step-by-step instructions
- For cooking questions: Give direct, factual answers without elaboration
- Always be brief and focused on the culinary information requested

## Non-Food Topics (ALWAYS REFUSE WITH STANDARD RESPONSE):
- Current events, news, politics
- Personal advice or coaching
- Entertainment, sports, or celebrities
- Technical support or technology questions
- Medical or health advice beyond basic nutrition
- Financial advice or information
- Mathematical problems unrelated to cooking measurements
- Any other non-food topic

Your sole purpose is providing food recipes and cooking information - nothing more, nothing less.
"""

# Client-side topic detection for additional safety
food_related_keywords = [
    'recipe', 'food', 'cook', 'bake', 'meal', 'dish', 'ingredient', 'cuisine',
    'breakfast', 'lunch', 'dinner', 'snack', 'dessert', 'appetizer', 'menu',
    'vegetable', 'fruit', 'meat', 'fish', 'chicken', 'beef', 'pork', 'lamb',
    'dairy', 'cheese', 'milk', 'cream', 'butter', 'egg', 'flour', 'sugar',
    'salt', 'pepper', 'spice', 'herb', 'oil', 'vinegar', 'sauce', 'marinade',
    'roast', 'grill', 'fry', 'boil', 'simmer', 'sauté', 'steam', 'chop', 'dice',
    'mince', 'slice', 'blend', 'mix', 'stir', 'whip', 'knead', 'baste', 'glaze',
    'flavor', 'taste', 'kitchen', 'pan', 'pot', 'oven', 'stove', 'microwave', 'blender',
    'refrigerator', 'freezer', 'measure', 'cup', 'tablespoon', 'teaspoon', 'temperature',
    'carbohydrate', 'protein', 'fat', 'calorie', 'nutrition', 'dietary', 'gluten',
    'vegan', 'vegetarian', 'pescatarian', 'keto', 'paleo', 'organic', 'meal plan',
    'grocery', 'shopping list', 'preserve', 'can', 'pickle', 'ferment', 'brew'
]

def is_food_related(query):
    """Basic check if query is likely food-related using keyword matching"""
    query = query.lower()
    
    # Quick allow for common food questions
    if re.search(r'(how|can|do you).*cook|recipe for|make .+\?|prepare .+\?', query):
        return True
        
    # Check for food-related keywords
    for keyword in food_related_keywords:
        if keyword in query.lower().split():
            return True
            
    return False

# Standard response for non-food topics
NON_FOOD_RESPONSE = "I can only provide recipes and food information. Would you like a recipe?"

# Chat loop with enhanced filtering
def chat():
    print("🍲 Recipe Chatbot is ready! Ask me about food recipes (type 'exit' to quit).")
    print("📌 NOTE: This chatbot will ONLY respond to food and recipe-related questions.")
    
    history = [{"role": "system", "content": system_prompt}]
    
    while True:
        user_input = input("\n👤 You: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            break

        # Check if query is food-related before sending to LLM
        if not is_food_related(user_input):
            print("🤖 ChefBot: " + NON_FOOD_RESPONSE)
            # Still add to history so the model learns from this interaction
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": NON_FOOD_RESPONSE})
            continue

        # If food-related, send to model for processing
        history.append({"role": "user", "content": user_input})
        response = llm.create_chat_completion(messages=history)
        reply = response["choices"][0]["message"]["content"]
        
        # Double-check response - if it doesn't look like food content, override it
        # This guards against the model ignoring its instructions
        if not is_food_related(reply) and not NON_FOOD_RESPONSE in reply:
            reply = NON_FOOD_RESPONSE
            
        print("🤖 ChefBot:", reply.strip())
        history.append({"role": "assistant", "content": reply.strip()})

chat()
