"""LaTeX rows of the component ablation (Table 10 of the manuscript) from ablation_v2.json."""
import json, pathlib, numpy as np
HERE = pathlib.Path(__file__).resolve().parent
rows = json.load(open(HERE / 'ablation_v2.json'))
names = ['v1: training-only embedding, mean prototype', '+ prototypes from recent enrolment', '+ embedding trained on the same enrolment data',
         '+ local prototypes (k nearest exemplars)', '+ identification margin delta (RAPID)']
labels = ['Training data, one mean prototype', '+ recent-enrolment prototypes', '+ embedding on enrolment data', '+ local prototypes ($k=3$)', '+ margin $\\beta$ (RAPID)']
for nm, lab in zip(names, labels):
    m = {k: np.mean([r[k] for r in rows if r['variant'] == nm]) for k in ('TAR', 'pooled', 'worst', 'ge50', 'devchosen', 'retry20', 'EER', 'AUC', 'TAR@FAR0.001')}
    print(f"{lab} & {100*m['TAR']:.1f}\\% & {100*m['pooled']:.2f}\\% & {100*m['worst']:.1f}\\% & {m['ge50']:.1f} & {100*m['devchosen']:.1f}\\% & "
          f"{100*m['retry20']:.1f}\\% & {100*m['EER']:.2f}\\% & {m['AUC']:.4f} & {100*m['TAR@FAR0.001']:.1f}\\% \\\\")
