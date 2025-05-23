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

# Food topic detection - FIXED VERSION
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
    'seafood', 'poultry', 'pasta', 'rice', 'grain', 'legume', 'bean', 'lentil',
    'broth', 'stock', 'soup', 'stew', 'casserole', 'salad', 'sandwich', 'wrap',
    'smoothie', 'juice', 'beverage', 'cocktail', 'wine', 'beer', 'spirit',
    'balsamic', 'soy sauce', 'condiment', 'dressing', 'dip', 'garnish', 'seasoning',
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
    'mise en place', 'al dente', 'roux', 'mirepoix', 'sofrito', 'bouquet garni', 'reduction',
    # Added specific dish names
    'biriyani', 'biryani', 'pizza', 'pasta', 'curry', 'tacos', 'burgers', 'stir-fry',
    'pancakes', 'waffles', 'cookies', 'cake', 'pie', 'muffins', 'brownies', 'bread',
    'sandwich', 'salad', 'soup', 'stew', 'chili', 'rice', 'noodles', 'ramen',
    'sushi', 'tempura', 'risotto', 'paella', 'gumbo', 'jambalaya', 'lasagna',
    'carbonara', 'alfredo', 'marinara', 'pesto', 'bolognese', 'pad thai', 'fried rice',
    'dumplings', 'wontons', 'spring rolls', 'samosas', 'empanadas', 'quesadillas'
]

# Common food phrases and patterns
food_phrases = [
    'how to cook', 'how to make', 'recipe for', 'cooking', 'baking',
    'ingredients for', 'preparation of', 'steps to make', 'method for',
    'cooking time', 'baking time', 'serve with', 'goes well with',
    'substitute for', 'instead of', 'alternative to', 'replace with'
]

def is_food_related(query):
    """Enhanced check if query is likely food-related"""
    query_lower = query.lower().strip()
    
    # Handle simple responses like "yes", "no", "sure" in context
    simple_responses = ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'no', 'nope']
    if query_lower in simple_responses:
        return True  # Assume context is food-related
    
    # Check for food phrases first
    for phrase in food_phrases:
        if phrase in query_lower:
            return True
    
    # Quick patterns for recipe requests
    if re.search(r'(how|can|do you).*cook|recipe for|make .+|prepare .+|cook .+', query_lower):
        return True
    
    # Check for food-related keywords (word boundaries to avoid partial matches)
    words_in_query = re.findall(r'\b\w+\b', query_lower)
    for keyword in food_related_keywords:
        if keyword in words_in_query:
            return True
    
    # Check for partial matches in compound words or phrases
    for keyword in food_related_keywords:
        if keyword in query_lower:
            return True
            
    return False

def chat(llm):
    """Run the chat loop with the model"""
    # Enhanced system prompt for more restrictive recipe chatbot
    system_prompt = """You are RecipeBot, a helpful AI assistant that specializes in food recipes and cooking information.

IMPORTANT INSTRUCTIONS:
1. You should ONLY respond to food and recipe-related questions
2. For non-food topics, politely redirect: "I can only provide recipes and food information. Would you like a recipe?"
3. For food questions, provide helpful, detailed responses about recipes, cooking techniques, ingredients, and food preparation
4. When someone asks for a recipe, provide clear ingredients and step-by-step instructions
5. Be friendly and enthusiastic about food and cooking

You can help with:
- Recipes with ingredients and cooking instructions
- Food substitutions and modifications  
- Cooking techniques and methods
- Recipe scaling and measurement conversions
- Ingredient information and questions
- Food storage and safety information
- Nutritional information about specific foods

Stay focused on food and cooking topics only."""

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
            response = llm.create_chat_completion(
                messages=history,
                max_tokens=512,  # Allow longer responses for recipes
                temperature=0.7,  # Slightly more creative responses
                top_p=0.9
            )
            
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