import os
import re
import time
import requests
from llama_cpp import Llama

# Configuration
MODEL_REPO_ID = "TheBloke/Llama-2-7B-Chat-GGUF"
MODEL_FILENAME = "llama-2-7b-chat.Q4_K_M.gguf"
MODEL_PATH = f"./{MODEL_FILENAME}"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO_ID}/resolve/main/{MODEL_FILENAME}"

# GPU Optimization for CUDA 11.4 (7.9GB VRAM)
GPU_LAYERS = 14                # Conservative layer offloading
N_CTX = 1536                   # Context window size
N_BATCH = 48                   # Reduced batch size
MAX_TOKENS = 320               # Response length limit

def download_model():
    """Download model with progress tracking"""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading {MODEL_FILENAME}...")
        try:
            with requests.get(MODEL_URL, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                chunk_size = 1024 * 1024  # 1MB chunks
                downloaded = 0
                
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = downloaded / total_size * 100
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"🚚 Downloading: {mb_downloaded:.1f}MB/{mb_total:.1f}MB ({percent:.1f}%)", end='\r')
                print("\n✅ Download complete!")
                return True
        except Exception as e:
            print(f"🚨 Download failed: {e}")
            return False
    return True

def initialize_model():
    """Initialize model with enhanced memory management"""
    print("🚀 Initializing model...")
    try:
        # GPU Configuration
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=GPU_LAYERS,
            n_ctx=N_CTX,
            n_batch=N_BATCH,
            n_threads=6,
            offload_kqv=True,
            verbose=False,
            tensor_split=[0.8],  # Use 80% of available VRAM
            rope_freq_base=10000,
            flash_attn=False,
            main_gpu=0
        )
        print("🔥 Model loaded with GPU acceleration!")
        return llm
    except Exception as e:
        print(f"⚠️ GPU Error: {e}\n🔄 Falling back to CPU...")
        return Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=6,
            verbose=False
        )

# Enhanced Food Detection System
FOOD_PATTERNS = re.compile(
    r'(?i)\b(recipe|how to (make|cook|prepare)|ingredients?|'
    r'cooking|baking|grill|roast|fry|cuisine|meal|diet|'
    r'nutrition|calories|vegan|vegetarian|keto|gluten|'
    r'snack|breakfast|lunch|dinner|dessert|appetizer)\b'
)

FOOD_KEYWORDS = {
    'recipe', 'cook', 'bake', 'ingredient', 'meal', 'dish',
    'nutrition', 'diet', 'vegetarian', 'vegan', 'protein', 'carb',
    'fat', 'grill', 'roast', 'spice', 'sauce', 'oven', 'stove',
    'chef', 'menu', 'food', 'eat', 'dinner', 'lunch', 'breakfast',
    'snack', 'dessert', 'appetizer', 'beverage', 'smoothie', 'juice'
}

def is_food_related(query):
    """Enhanced food detection with phrase matching"""
    query = query.lower()
    return bool(FOOD_PATTERNS.search(query)) or any(kw in query for kw in FOOD_KEYWORDS)

def chat(llm):
    """Chat interface with memory safeguards"""
    SYSTEM_PROMPT = """You are ChefBot, a cooking expert. Follow these rules STRICTLY:
    1. Only respond to food/recipe questions
    2. For non-food queries: "I specialize in cooking and recipes"
    3. Recipe format:
       - **Name**: Bold title
       - ⏱️ Cooking Time
       - 📝 Ingredients (bullet points)
       - 🧑🍳 Instructions (numbered steps)
       - 💡 Pro Tip (optional)
    4. Keep responses under 300 tokens"""
    
    NON_FOOD_RESPONSE = "I specialize in cooking and recipes. What would you like to make today?"
    SAFETY_PROMPT = "Let's focus on cooking! What recipe can I help you with?"
    
    print("\n" + "="*50)
    print("🧑🍳 ChefBot 2.0 - CUDA Optimized".center(50))
    print("="*50)
    
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    token_count = 0

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ('exit', 'quit', 'bye'):
                break

            if not is_food_related(user_input):
                print(f"🤖 ChefBot: {NON_FOOD_RESPONSE}")
                continue

            # Manage conversation history
            history.append({"role": "user", "content": user_input})
            if len(history) > 5:  # Keep last 3 exchanges
                history = [history[0]] + history[-4:]

            # Generate response
            start_time = time.time()
            response = llm.create_chat_completion(
                messages=history,
                temperature=0.65,
                max_tokens=MAX_TOKENS,
                stop=["\n\n", "###"],
                repeat_penalty=1.15
            )
            
            reply = response["choices"][0]["message"]["content"]
            tokens_used = response["usage"]["completion_tokens"]
            token_count += tokens_used

            # Enforce food focus
            if not is_food_related(reply):
                reply = SAFETY_PROMPT
                history = [history[0]]  # Reset context

            print(f"🤖 ChefBot: {reply.strip()}")
            history.append({"role": "assistant", "content": reply.strip()})

            # Performance metrics
            delay = time.time() - start_time
            print(f"\n⏱️ {delay:.2f}s | 📝 {tokens_used} tokens | 💬 {token_count} total tokens")

        except Exception as e:
            print(f"⚠️ Error: {e}")
            if "KV cache" in str(e):
                print("🔄 Resetting conversation history...")
                history = [history[0]]
            time.sleep(1)  # Cooldown period

def main():
    print("\n" + "="*50)
    print("🌟 ChefBot - Professional Cooking Assistant".center(50))
    print("="*50)
    
    if not download_model():
        return
    
    llm = initialize_model()
    if llm:
        chat(llm)

if __name__ == "__main__":
    # Set environment variables for CUDA 11.4
    os.environ["GGML_CUDA_MAX_STREAMS"] = "4"
    os.environ["GGML_CUDA_MMQ_Y"] = "0"
    main()