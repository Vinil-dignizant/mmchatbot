import os
import re
import time
import requests
from llama_cpp import Llama

# Settings
MODEL_REPO_ID = "TheBloke/Llama-2-7B-Chat-GGUF"
MODEL_FILENAME = "llama-2-7b-chat.Q4_K_M.gguf"  # Smaller quantization for easier loading
MODEL_URL = f"https://huggingface.co/{MODEL_REPO_ID}/resolve/main/{MODEL_FILENAME}"
MODEL_PATH = f"./{MODEL_FILENAME}"

def download_model():
    """Download the model if not present"""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model {MODEL_FILENAME}...")
        print(f"From: {MODEL_URL}")
        print("This may take a while depending on your internet connection.")
        
        try:
            response = requests.get(MODEL_URL, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB
            downloaded = 0
            
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Print progress
                        if total_size > 0:
                            percent = downloaded / total_size * 100
                            print(f"Progress: {downloaded / (1024 * 1024):.1f}MB / {total_size / (1024 * 1024):.1f}MB ({percent:.1f}%)", end='\r')
            
            print("\nDownload complete!")
            return True
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
    else:
        print(f"Model already exists at {MODEL_PATH}")
        return True

def initialize_model():
    """Initialize the Llama model with appropriate settings"""
    print("Initializing model...")
    
    try:
        # First try with GPU acceleration
        try:
            print("Attempting to load model with GPU acceleration...")
            llm = Llama(
                model_path=MODEL_PATH,
                n_gpu_layers=-1,  # Try to offload all layers
                n_ctx=2048,
                verbose=False,
                n_threads=4,
                n_batch=512,
                n_gpu=1,
                device='cuda',  # Use GPU
            )
            print("✅ Model loaded successfully with GPU acceleration!")
            return llm
        except Exception as e:
            print(f"GPU loading failed: {e}")
            print("Falling back to CPU...")
            
        # Fallback to CPU with more conservative settings
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            verbose=False,
            n_threads=4
        )
        print("✅ Model loaded successfully with CPU!")
        return llm
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please make sure the model file exists and is not corrupted.")
        return None

# Food topic detection
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
    'grocery', 'shopping list', 'preserve', 'can', 'pickle', 'ferment', 'brew',
    'snack', 'treat', 'appetizer', 'entree', 'side dish', 'dessert', 'sauce',
    'condiment', 'dip', 'spread', 'topping', 'garnish', 'syrup', 'jam', 'jelly',
    'preserve', 'compote', 'chutney', 'relish', 'salsa', 'pesto', 'hummus',
    'guacamole', 'salsa verde', 'tartar sauce', 'aioli', 'vinaigrette', 'dressing',
    'marinade', 'brine', 'rub', 'glaze', 'baste', 'sauté', 'stir-fry', 'braise',
    'roast', 'grill', 'barbecue', 'smoke', 'bake', 'broil', 'fry', 'deep fry',
    'pan fry', 'steam', 'poach', 'simmer', 'blanch', 'sear', 'caramelize',
    'deglaze', 'reduce', 'infuse', 'pickle', 'ferment', 'cure', 'dry',
    'salt', 'sugar', 'smoke', 'toast', 'brown', 'char', 'flambé', 'flambé',
    'sous vide', 'pressure cook', 'slow cook', 'air fry', 'microwave', 'steam',
    'bake', 'roast', 'grill', 'barbecue', 'stir-fry', 'sauté', 'braise',
    'simmer', 'poach', 'blanch', 'sear', 'caramelize', 'deglaze', 'reduce',
    'infuse', 'pickle', 'ferment', 'cure', 'dry', 'salt', 'sugar', 'smoke',
    'toast', 'brown', 'char', 'flambé', 'flambé', 'sous vide', 'pressure cook',
    'slow cook', 'air fry', 'microwave', 'steam', 'bake', 'roast', 'grill',
    'barbecue', 'stir-fry', 'sauté', 'braise', 'simmer', 'poach', 'blanch',
    'sear', 'caramelize', 'deglaze', 'reduce', 'infuse', 'pickle', 'ferment',
    'cure', 'dry', 'salt', 'sugar', 'smoke', 'toast', 'brown', 'char', 'flambé',
    'flambé', 'sous vide', 'pressure cook', 'slow cook', 'air fry', 'microwave',
    'steam', 'bake', 'roast', 'grill', 'barbecue', 'stir-fry', 'sauté', 'braise',
    'simmer', 'poach', 'blanch', 'sear', 'caramelize', 'deglaze', 'reduce',
    'infuse', 'pickle', 'ferment', 'cure', 'dry', 'salt', 'sugar', 'smoke',
    'seafood', 'poultry', 'pasta', 'rice', 'grain', 'legume', 'bean', 'lentil',
    'broth', 'stock', 'soup', 'stew', 'casserole', 'salad', 'sandwich', 'wrap',
    'smoothie', 'juice', 'beverage', 'cocktail', 'wine', 'beer', 'spirit',
    'balsamic', 'soy sauce', 'condiment', 'dressing', 'dip', 'garnish', 'seasoning',
    'braise', 'broil', 'poach', 'blanch', 'caramelize', 'deglaze', 'reduce', 'infuse',
    'marinate', 'tenderize', 'cure', 'smoke', 'toast', 'brown', 'char', 'flambé',
    'platter', 'bowl', 'skillet', 'griddle', 'baking sheet', 'casserole dish', 'ramekin',
    'food processor', 'mixer', 'colander', 'strainer', 'spatula', 'whisk', 'tongs',
    'appetit', 'culinary', 'gastronomy', 'gourmet', 'homemade', 'artisanal', 'fresh',
    'seasonal', 'farm-to-table', 'local', 'sustainable', 'free-range', 'grass-fed',
    'pastry', 'bread', 'dough', 'batter', 'frosting', 'icing', 'confection', 'chocolate',
    'honey', 'syrup', 'jam', 'jelly', 'preserve', 'compote', 'chutney', 'relish',
    'umami', 'bitter', 'sour', 'sweet', 'salty', 'spicy', 'tangy', 'zesty', 'savory',
    'meal prep', 'batch cooking', 'leftovers', 'doggy bag', 'takeout', 'delivery',
    'restaurant', 'bistro', 'café', 'diner', 'eatery', 'foodie', 'chef', 'sous chef',
    'diet', 'allergy', 'intolerance', 'lactose-free', 'nut-free', 'soy-free', 'plant-based',
    'macronutrient', 'micronutrient', 'vitamin', 'mineral', 'antioxidant', 'probiotics',
    'fermentation', 'pickling', 'canning', 'dehydrating', 'freezing', 'vacuum-sealing',
    'barbecue', 'rotisserie', 'slow cooker', 'pressure cooker', 'air fryer', 'sous vide',
    'mise en place', 'al dente', 'roux', 'mirepoix', 'sofrito', 'bouquet garni', 'reduction'
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

def chat(llm):
    """Run the chat loop with the model"""
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

    # Standard response for non-food topics
    NON_FOOD_RESPONSE = "I can only provide recipes and food information. Would you like a recipe?"

    print("\n🍲 Recipe Chatbot is ready! Ask me about food recipes (type 'exit' to quit).")
    print("📌 NOTE: This chatbot will ONLY respond to food and recipe-related questions.")
    
    history = [{"role": "system", "content": system_prompt}]
    
    # Add counters for token statistics
    total_tokens_generated = 0
    total_generation_time = 0
    interaction_count = 0
    
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
        
        try:
            # Start timing token generation
            start_time = time.time()
            
            # Get response from model
            response = llm.create_chat_completion(messages=history)
            
            # End timing
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Extract response and token count information
            reply = response["choices"][0]["message"]["content"]
            tokens_generated = response["usage"]["completion_tokens"]
            
            # Update statistics
            total_tokens_generated += tokens_generated
            total_generation_time += generation_time
            interaction_count += 1
            
            # Double-check response - if it doesn't look like food content, override it
            # This guards against the model ignoring its instructions
            if not is_food_related(reply) and NON_FOOD_RESPONSE not in reply:
                reply = NON_FOOD_RESPONSE
                
            print("🤖 ChefBot:", reply.strip())
            print(f"\n⏱️ Generation time: {generation_time:.2f} seconds")
            print(f"🔤 Tokens generated: {tokens_generated}")
            
            history.append({"role": "assistant", "content": reply.strip()})
        except Exception as e:
            print(f"Error generating response: {e}")
            print("Let's try again. If this keeps happening, try restarting the program.")

    # Print final statistics if there were any interactions
    if interaction_count > 0:
        print("\n" + "=" * 50)
        print("📊 Generation Statistics:")
        print(f"Total interactions: {interaction_count}")
        print(f"Total tokens generated: {total_tokens_generated}")
        print(f"Total generation time: {total_generation_time:.2f} seconds")
        print(f"Average tokens per response: {total_tokens_generated / interaction_count:.1f}")
        print(f"Average generation time: {total_generation_time / interaction_count:.2f} seconds")
        print(f"Average tokens per second: {total_tokens_generated / total_generation_time:.1f}")
        print("=" * 50)

def main():
    """Main function to orchestrate the program flow"""
    print("=" * 50)
    print("🍳 Llama-2 Recipe Chatbot 🍳")
    print("=" * 50)
    
    # Download model if needed
    if not download_model():
        print("Failed to download model. Please check your internet connection.")
        return
    
    # Initialize the model
    llm = initialize_model()
    if llm is None:
        return
    
    # Start the chat session
    chat(llm)

if __name__ == "__main__":
    main()