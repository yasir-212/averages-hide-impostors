"""Reference model of claim routing and two independent credentials.

This is a cryptographic protocol demonstration, NOT a radio implementation.
The primary secret can be compromised; the optional second key is assumed
independent (not available to that attacker). Fast-path PHY is a Boolean input
from an independently calibrated model. No on-air relay protection is claimed.
"""
from dataclasses import dataclass
import hmac, hashlib, secrets
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey,Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

DOMAIN=b'RFF-CLAIM-GATE-v1\x00'
@dataclass(frozen=True)
class Challenge:
    target: int
    nonce: bytes

def encode_challenge(ch: Challenge) -> bytes:
    if not (0 <= ch.target < (1<<16) and len(ch.nonce)==16):
        raise ValueError('bad challenge fields')
    return DOMAIN + ch.target.to_bytes(2,'big') + ch.nonce

def primary_tag(key: bytes, ch:Challenge) -> bytes:
    if len(key)<16:raise ValueError('primary key too short')
    return hmac.digest(key, encode_challenge(ch), 'sha256')

class ConditionalVerifier:
    def __init__(self, primary:dict[int,bytes], public_secondary:dict[int,Ed25519PublicKey], rf_eligible:set[int]):
        self.primary=dict(primary)
        self.secondary=dict(public_secondary)
        self.rf_eligible=set(rf_eligible)
        self.pending=set()

    def begin(self,target:int) -> Challenge:
        if target not in self.primary: raise KeyError('unknown identity')
        challenge=Challenge(target,secrets.token_bytes(16))
        while encode_challenge(challenge) in self.pending:
            challenge=Challenge(target,secrets.token_bytes(16))
        self.pending.add(encode_challenge(challenge))
        return challenge

    def verify(self,ch:Challenge,tag:bytes,rf_accept:bool,secondary_signature:bytes | None = None)->str:
        msg=encode_challenge(ch)
        if msg not in self.pending:
            return 'REJECT_REPLAY_OR_UNKNOWN_CHALLENGE'
        self.pending.remove(msg) # one attempt per challenge; no reusable key oracle
        k=self.primary.get(ch.target)
        if k is None or not hmac.compare_digest(primary_tag(k,ch),tag):
            return 'REJECT_BAD_PRIMARY'
        if ch.target in self.rf_eligible and bool(rf_accept):
            return 'ACCEPT_RF_FAST'
        if secondary_signature is None or ch.target not in self.secondary:
            return 'REJECT_SECONDARY_REQUIRED'
        try:self.secondary[ch.target].verify(secondary_signature,msg)
        except (InvalidSignature,ValueError,TypeError):return 'REJECT_INVALID_SECONDARY'
        return 'ACCEPT_SECONDARY'

def sign_secondary(key:Ed25519PrivateKey,ch:Challenge)->bytes:
    return key.sign(encode_challenge(ch))
