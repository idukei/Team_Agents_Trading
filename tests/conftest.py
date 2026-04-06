import os
import sys
from pathlib import Path

# Ensure teamtrade/ is in sys.path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set minimal env vars so Settings can instantiate without real secrets
os.environ.setdefault("FIREWORKS_API_KEY", "test_fw_key")
os.environ.setdefault("ALPACA_API_KEY", "test_alpaca_key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test_alpaca_secret")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456789:test_token")
os.environ.setdefault("TELEGRAM_CHAT_ID", "123456789")
os.environ.setdefault("POSTGRES_DSN", "postgresql://tt:test@localhost:5432/teamtrade")
