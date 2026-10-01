import os

from dotenv import load_dotenv

load_dotenv(override=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HAIKU_MODEL = os.getenv("INBOXHERO_MODEL", "claude-haiku-4-5-20251001")
SONNET_MODEL = os.getenv("SONNET_MODEL", "claude-sonnet-5")

TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")
# Optional. TypeSafe pricing is not read from the API; set these (USD per million tokens) to get a Jev cost.
JEV_PRICE_INPUT = float(os.getenv("JEV_PRICE_INPUT")) if os.getenv("JEV_PRICE_INPUT") else None
JEV_PRICE_OUTPUT = float(os.getenv("JEV_PRICE_OUTPUT")) if os.getenv("JEV_PRICE_OUTPUT") else None

REQUEST_DELAY_S = float(os.getenv("REQUEST_DELAY_S", "1.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))


def require_api_key():
    if not ANTHROPIC_API_KEY:
        raise SystemExit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ANTHROPIC_API_KEY


def require_typesafe_key():
    if not TYPESAFE_API_KEY:
        raise SystemExit("TYPESAFE_API_KEY is not set. Copy .env.example to .env and add your key.")
    return TYPESAFE_API_KEY
