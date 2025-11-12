import os
import sys
sys.path.append("c:\\Users\\friva\\Documents\\ATRbot-beta")
import logging
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.bot.chatbot import main as run_chatbot
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

# Load environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN2")  # Use TELEGRAM_TOKEN2 for Groq version
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN2 is not set in the environment variables.")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")        
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")
# No need to hardcode API keys here - they'll be set via Heroku config variables
# This ensures they're not exposed in the code repository

# Configure more visible logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("APP2")

def main():
    """
    Main entry point for the Telegram bot using the advanced chatbot.py implementation.
    This version includes:
    - Conversational memory
    - Better error handling
    - Evaluation metrics
    - Professional Telegram bot framework
    """
    try:
        # Verify required environment variables
        required_vars = ["TELEGRAM_TOKEN2", "GROQ_API_KEY"]  # Updated to check for TELEGRAM_TOKEN2
        for var in required_vars:
            if not os.getenv(var):
                raise ValueError(f"{var} is not set in the environment variables.")
        
        # Log the API keys being used (first few characters only for security)
        logger.info("🔑 Environment Variables:")
        logger.info(f"GROQ_API_KEY: {os.environ.get('GROQ_API_KEY', 'NOT_FOUND')[:20]}...")
        logger.info(f"TELEGRAM_TOKEN: {os.environ.get('TELEGRAM_TOKEN', 'NOT_FOUND')[:20]}...")
        
        # Test the Groq API connection before starting the bot
        try:
            logger.info("🔄 Testing Groq API connection...")
            llm = ChatGroq(
                groq_api_key=os.environ["GROQ_API_KEY"],
                model_name="llama-3.1-8b-instant",
                temperature=0.5,        # Lower temperature for more factual responses
                max_tokens=800,         # Enough for detailed explanations
                top_p=0.85,              # Slightly more focused sampling
                frequency_penalty=0.2,
                presence_penalty=0.1
            )
            start_time = time.time()
            response = llm.invoke("Test connection")
            end_time = time.time()
            logger.info(f"✅ Groq API connection successful! (Response time: {end_time-start_time:.2f}s)")
        except Exception as e:
            logger.error(f"❌ Groq API test failed: {e}")
            # Continue anyway as the chatbot has error handling
        
        logger.info("🚀 Starting ATRbot with advanced features...")
        logger.info("Features enabled: Fresh conversation memory, Error handling, Evaluation metrics")
        logger.info("Tip: Users can type /reset to clear conversation history at any time")
        
        # Run the chatbot
        run_chatbot()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise

if __name__ == "__main__":
    main()
