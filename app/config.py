import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
WHITELIST = [d.strip() for d in os.getenv("WHITELIST_DOMAINS","").split(",") if d.strip()]
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL", "4"))
APP_BRAND = os.getenv("APP_BRAND", "Agentra FactCheck")
