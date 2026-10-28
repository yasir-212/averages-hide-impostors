"""Re-derive every RAPID-v2 table row from the result files and check that the paper contains it verbatim."""
import sys as _sys, os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_R = next(p for p in [_HERE, *_HERE.parents] if (p / 'paths.py').exists())
for _p in (_R / 'sim', _R / 'rapid', _R, _HERE):
    _sys.path.insert(0, str(_p))
from paths import FIG_DIR, PAPER_TEX
import os, re, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_tables as G
tex = open(str(PAPER_TEX)).read()
ok = bad = 0
for line in G.main_table().splitlines():
    if '&' not in line: continue
    hit = line.strip() in tex; ok += hit; bad += not hit
    print(('OK       ' if hit else 'MISMATCH ') + line.split('&')[0].strip())
for path in (str(_HERE / 'tab_rapid_ablation.tex'), str(_HERE / 'tab_openset.tex')):
    if not os.path.exists(path): continue
    for line in open(path):
        if line.count('&') >= 8 and '\\\\' in line:
            hit = line.strip() in tex; ok += hit; bad += not hit
            print(('OK       ' if hit else 'MISMATCH ') + line.split('&')[0].strip())
print(f'{ok} rows verified, {bad} mismatches')
