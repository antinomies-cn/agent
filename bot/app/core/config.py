import os
from pathlib import Path
from dotenv import load_dotenv

# 统一从 .env.docker.example 加载环境变量，避免多入口读取不一致。
_ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = _ROOT_DIR / ".env.docker.example"
load_dotenv(dotenv_path=ENV_FILE_PATH)

IS_PROD = os.getenv("ENV", "dev") == "prod"
