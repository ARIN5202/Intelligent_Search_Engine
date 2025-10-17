from pathlib import Path 
lines=Path('src/retrieval/manager.py').read_text(encoding='utf-8').splitlines()[82:90] 
print(lines) 
