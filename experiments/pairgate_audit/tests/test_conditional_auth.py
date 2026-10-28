import secrets,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from conditional_auth import *


def build(eligible={0}):
    primary={0:secrets.token_bytes(32),1:secrets.token_bytes(32)}
    signing={t:Ed25519PrivateKey.generate() for t in primary}
    verifier=ConditionalVerifier(primary,{t:k.public_key() for t,k in signing.items()},eligible)
    return verifier,primary,signing

def test_genuine_eligible_fast():
    v,keys,s=build();c=v.begin(0)
    assert v.verify(c,primary_tag(keys[0],c),True)=='ACCEPT_RF_FAST'

def test_primary_keyless_attacker():
    v,keys,s=build();c=v.begin(0)
    assert v.verify(c,primary_tag(keys[1],c),True)=='REJECT_BAD_PRIMARY'

def test_stolen_primary_and_rf_fails():
    v,keys,s=build();c=v.begin(0)
    assert v.verify(c,primary_tag(keys[0],c),False)=='REJECT_SECONDARY_REQUIRED'

def test_stolen_primary_and_rf_passes_is_not_protected():
    v,keys,s=build();c=v.begin(0)
    assert v.verify(c,primary_tag(keys[0],c),True)=='ACCEPT_RF_FAST'

def test_noneligible_needs_secondary():
    v,keys,s=build();c=v.begin(1)
    assert v.verify(c,primary_tag(keys[1],c),True)=='REJECT_SECONDARY_REQUIRED'

def test_independent_secondary_works():
    v,keys,s=build();c=v.begin(1)
    assert v.verify(c,primary_tag(keys[1],c),True,sign_secondary(s[1],c))=='ACCEPT_SECONDARY'

def test_invalid_secondary_key():
    v,keys,s=build();c=v.begin(1)
    assert v.verify(c,primary_tag(keys[1],c),True,sign_secondary(s[0],c))=='REJECT_INVALID_SECONDARY'

def test_replay_consumed():
    v,keys,s=build();c=v.begin(1); tag=primary_tag(keys[1],c); sig=sign_secondary(s[1],c)
    assert v.verify(c,tag,False,sig)=='ACCEPT_SECONDARY'
    assert v.verify(c,tag,False,sig)=='REJECT_REPLAY_OR_UNKNOWN_CHALLENGE'

def test_cross_target_signature_fail():
    v,keys,s=build();c=v.begin(1)
    modified=Challenge(0,c.nonce)
    assert v.verify(c,primary_tag(keys[1],c),False,sign_secondary(s[0],modified))=='REJECT_INVALID_SECONDARY'

def test_random_fake_sig_fails():
    v,keys,s=build();c=v.begin(1)
    assert v.verify(c,primary_tag(keys[1],c),False,secrets.token_bytes(64))=='REJECT_INVALID_SECONDARY'
