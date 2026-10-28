"""Host-only cryptographic primitive timing, excludes capture, airtime, secure element."""
from pathlib import Path
from time import perf_counter_ns
import secrets,csv,statistics,sys
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
sys.path.insert(0,str(Path(__file__).parent))
from conditional_auth import Challenge,primary_tag,sign_secondary,encode_challenge
ROOT=Path(__file__).resolve().parents[1]
ch=Challenge(7,secrets.token_bytes(16));key=secrets.token_bytes(32);signer=Ed25519PrivateKey.generate();pk=signer.public_key();tag=primary_tag(key,ch);sig=sign_secondary(signer,ch)
ops={'primary_hmac_32B':lambda:primary_tag(key,ch),'verify_hmac':lambda:__import__('hmac').compare_digest(primary_tag(key,ch),tag),
     'sign_ed25519_host':lambda:sign_secondary(signer,ch),'verify_ed25519_host':lambda:pk.verify(sig,encode_challenge(ch))}
rows=[]
for name,f in ops.items():
 for _ in range(1000):f()
 timings=[]
 for _ in range(3000):
  start=perf_counter_ns();f();timings.append((perf_counter_ns()-start)/1e6)
 rows.append({'operation':name,'n':len(timings),'median_ms':statistics.median(timings),'p95_ms':sorted(timings)[2849], 'min_ms':min(timings),'max_ms':max(timings)})
print(rows)
with open(ROOT/'results/host_cryptography_times.csv','w',newline='') as io:
 w=csv.DictWriter(io,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
