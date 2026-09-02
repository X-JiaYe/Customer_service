import sys
from pathlib import Path

# 让测试能 import 项目根目录下的 config / agent / tools / knowledge
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
