import os
from pathlib import Path
from dotenv import load_dotenv

# 环境变量优先级：系统环境 > .env > .env.docker.example。
# 这样既支持本地/容器通过 .env 注入，也保留模板文件中的默认回退值。
_ROOT_DIR = Path(__file__).resolve().parents[2]
PRIMARY_ENV_FILE_PATH = _ROOT_DIR / ".env"
TEMPLATE_ENV_FILE_PATH = _ROOT_DIR / ".env.docker.example"

load_dotenv(dotenv_path=PRIMARY_ENV_FILE_PATH, override=False)
load_dotenv(dotenv_path=TEMPLATE_ENV_FILE_PATH, override=False)

IS_PROD = os.getenv("ENV", "dev") == "prod"
