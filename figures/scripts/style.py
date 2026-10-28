"""Shared figure style of the paper (validated categorical palette, fixed order per system)."""
import pathlib as _pl
_ROOT = next(p for p in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents] if (p / 'paths.py').exists())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6'
GRAY, INK, INK2, GRID = '#8a8984', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.4, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'legend.frameon': False, 'lines.linewidth': 1.8, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans',
    'axes.titlelocation': 'left'})
W = 6.3          # full text width of the preprint layout (inches)
OUT = str(_ROOT / 'figures') + '/'

# system key -> (label, colour, linestyle, marker); fixed order = legend order
SYS = {
    'RAPID':    ('RAPID (ours)', BLUE, '-', 'o'),
    'RF-eq':    ('SMoRFFI forest, equal data', GRAY, '-', 's'),
    'RF':       ('SMoRFFI forest, published', GRAY, ':', 's'),
    'MLP-eq':   ('MLP, equal data', ORANGE, '-', '^'),
    'MLP':      ('MLP, published', ORANGE, ':', '^'),
    'kNN-eq':   ('kNN, equal data', AQUA, '-', 'D'),
    'GBT-eq':   ('GBT, equal data', YELLOW, '-', 'v'),
    'CFeq':     ('CosFace prototypes, equal data', MAGENTA, '-', 'P'),
    'CFtrain':  ('CosFace prototypes, training data', VIOLET, ':', 'P'),
}

def style(k, **kw):
    lab, c, ls, mk = SYS[k]; d = dict(color=c, ls=ls, label=lab); d.update(kw); return d

def save(fig, name):
    fig.savefig(OUT + name, bbox_inches='tight', pad_inches=0.02); plt.close(fig); print('wrote', name)
