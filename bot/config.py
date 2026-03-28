import os
from dotenv import load_dotenv

# 统一在配置模块加载环境变量，避免重复调用。
load_dotenv()

IS_PROD = os.getenv("ENV", "dev") == "prod"
