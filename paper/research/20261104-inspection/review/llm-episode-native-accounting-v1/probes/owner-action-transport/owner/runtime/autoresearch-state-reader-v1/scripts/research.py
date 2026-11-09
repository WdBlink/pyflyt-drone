import json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[sys.argv.index('--packet')+1]).read_text())
print(json.dumps({'status':'refused' if p.get('fail') else 'synthetic-only','argv':sys.argv[1:]}))
raise SystemExit(2 if p.get('fail') else 0)
