#!/usr/bin/env python3
from pathlib import Path
import subprocess,tempfile,shutil,zipfile,json,datetime,sys
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'pass5-input/astra-prmers-aevum-pass5-prp-max-input'
FULL_REFERENCE=BASE.is_dir()
if not FULL_REFERENCE:BASE=ROOT/'scripts/pass5-baseline'
NOTES={'PASS5_RECOVERY.md','PASS5_WIP.patch','PASS5_STATUS.txt'}
def source(p):
 q=p.relative_to(BASE if p.is_relative_to(BASE) else ROOT)
 return not (any(x.startswith('build') or x in ('.git','__pycache__','.cache','node_modules') for x in q.parts) or p.suffix in ('.o','.d','.so','.a','.pyc','.zip','.log','.bin') or str(q) in ('third_party/aevum/src/bundle.cpp','third_party/aevum/src/version.inc') or p.name in NOTES)
old={str(p.relative_to(BASE)):p for p in BASE.rglob('*') if p.is_file() and source(p)}
new={str(p.relative_to(ROOT)):p for p in ROOT.rglob('*') if p.is_file() and source(p)}
if not FULL_REFERENCE:
 prior=ROOT/'PASS5_STATUS.txt'
 names={x[2:] for x in prior.read_text().splitlines() if x.startswith(('M ','A ','D '))} if prior.exists() else set()
 names.update(old)
 # Existing input documents also contain "pass5"; never infer that they are new.
 # New work outside the recorded scope can be supplied as relative-path arguments.
 names.update(sys.argv[1:])
 new={f:p for f,p in new.items() if f in names}
changed=[f for f in sorted(old.keys()|new.keys()) if f not in old or f not in new or old[f].read_bytes()!=new[f].read_bytes()]
with tempfile.TemporaryDirectory(prefix='pass5-diff-') as t:
 t=Path(t);(t/'a').mkdir();(t/'b').mkdir()
 for f in changed:
  for side,d in [('a',old),('b',new)]:
   if f in d:
    p=t/side/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(d[f],p)
 patch=subprocess.run(['git','diff','--no-index','--binary','--no-renames','a','b'],cwd=t,stdout=subprocess.PIPE).stdout
 lines=[]
 for line in patch.splitlines(keepends=True):
  if line.startswith((b'diff --git ',b'--- ',b'+++ ',b'Binary files ')):
   line=line.replace(b'a/a/',b'a/').replace(b'b/b/',b'b/').replace(b'a/b/',b'a/').replace(b'b/a/',b'b/')
  lines.append(line)
 (ROOT/'PASS5_WIP.patch').write_bytes(b''.join(lines))
 stat=subprocess.run(['git','diff','--no-index','--stat','a','b'],cwd=t,stdout=subprocess.PIPE).stdout.decode()
status='Current commit SHA: unavailable (current authoritative source worktree has no .git metadata).\n'
status+='git status --short: fatal: not a git repository\ngit diff --stat: unavailable without .git; equivalent binary-capable git --no-index source diff follows.\n'
status+='Reference: preserved pre-Pass-5 sources (full extracted input when available, otherwise scripts/pass5-baseline and prior status scope); current sources never overwritten.\n\n'+stat
status+='\nChanged files (M existing, A new, D deleted):\n'+''.join(('A' if f not in old else 'D' if f not in new else 'M')+' '+f+'\n' for f in changed)
status+='\nAll A entries above are relevant untracked Pass-5 source/scripts/notes; build artifacts excluded.\n'
(ROOT/'PASS5_STATUS.txt').write_text(status)
archive=ROOT.parent/'prmers-aevum-pass5-recovery-current.zip'
tmp=archive.with_suffix('.tmp')
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(set(changed)|NOTES):
  p=ROOT/f
  if p.is_file():z.write(p,f)
 # Preserve compact existing test evidence; these are not benchmark measurements.
 for name in ('pass5-delivery-tests.log','pass5-auto.log','pass5-syntax.log','pass5-continuation-tests.log'):
  p=Path('/tmp')/name
  if p.exists():z.writestr('recovery-evidence/'+name,p.read_bytes()[-50000:])
 assert z.testzip() is None
tmp.replace(archive)
print(json.dumps({'changed_files':len(changed),'patch_bytes':(ROOT/'PASS5_WIP.patch').stat().st_size,'archive':str(archive),'archive_bytes':archive.stat().st_size}))
