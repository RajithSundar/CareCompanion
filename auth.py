"""Login with demo accounts and JWT sessions.

* Three demo accounts, one per persona in demo_data.py. Passwords are hashed with
  scrypt at startup and compared in constant time; unknown usernames cost the same.
* Tokens are HS256 JWTs (PyJWT) with iss, aud, iat, nbf, exp and a random jti. The
  algorithm is pinned on decode, so "none" and algorithm-confusion tokens are rejected.
* The browser keeps the token in an HttpOnly, SameSite=Strict cookie (never readable by
  page scripts). API clients may send it as `Authorization: Bearer <token>` instead.
* Logout revokes the token's jti until it would have expired anyway.
* The signing key comes from CC_JWT_SECRET, or is generated once and kept in
  data/.jwt_secret (mode 600) so sessions survive restarts.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import jwt

ISSUER = "carecompanion"
AUDIENCE = "carecompanion-web"
ALGORITHM = "HS256"
TOKEN_TTL_S = int(os.getenv("CC_TOKEN_TTL_HOURS", "8")) * 3600
COOKIE = "cc_token"

DEMO_PASSWORD = os.getenv("CC_DEMO_PASSWORD", "demo1234")


@dataclass(frozen=True)
class Account:
    username: str
    pid: str           # persona id in demo_data.DEFAULT_APP_STATE
    name: str
    role: str

    @property
    def store_key(self) -> str:
        """Stable 32-hex key for this account's data file."""
        return hashlib.sha256(f"account:{self.username}".encode()).hexdigest()[:32]

    def public(self) -> Dict[str, str]:
        return {"username": self.username, "pid": self.pid.upper(), "name": self.name, "role": self.role}


ACCOUNTS: Dict[str, Account] = {
    a.username: a for a in (
        Account("alex", "p01", "Alex Rivera", "Half-Marathon Prep, Morning Runner"),
        Account("maya", "p07", "Maya Lin", "Tech Lead, Desk-Bound + Evening Yoga"),
        Account("jordan", "p12", "Jordan Patel", "Strength Athlete, Hypertrophy Focus"),
    )
}


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)


_SALTS = {u: secrets.token_bytes(16) for u in ACCOUNTS}
_HASHES = {u: _hash(DEMO_PASSWORD, _SALTS[u]) for u in ACCOUNTS}
_DUMMY_SALT = secrets.token_bytes(16)
_DUMMY_HASH = _hash(secrets.token_hex(16), _DUMMY_SALT)


def authenticate(username: str, password: str) -> Optional[Account]:
    user = username.strip().lower()
    salt, expected = (_SALTS[user], _HASHES[user]) if user in ACCOUNTS else (_DUMMY_SALT, _DUMMY_HASH)
    ok = hmac.compare_digest(_hash(password, salt), expected)
    return ACCOUNTS[user] if ok and user in ACCOUNTS else None


# --------------------------------------------------------------------------- signing key

def _load_secret() -> str:
    env = os.getenv("CC_JWT_SECRET", "").strip()
    if env:
        if len(env) < 32:
            raise RuntimeError("CC_JWT_SECRET must be at least 32 characters.")
        return env
    path = Path(os.getenv("CC_JWT_SECRET_FILE", str(Path(__file__).resolve().parent / "data" / ".jwt_secret")))
    try:
        value = path.read_text(encoding="utf-8").strip()
        if len(value) >= 32:
            return value
    except OSError:
        pass
    value = secrets.token_urlsafe(48)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(value)
    return value


_secret: Optional[str] = None


def secret() -> str:
    global _secret
    if _secret is None:
        _secret = _load_secret()
    return _secret


def set_secret(value: Optional[str]) -> None:
    """For tests: use a fixed key (or None to reload)."""
    global _secret
    _secret = value


# --------------------------------------------------------------------------- tokens

class _Revoked:
    def __init__(self) -> None:
        self._jtis: Dict[str, float] = {}
        self._lock = threading.Lock()

    def add(self, jti: str, exp: float) -> None:
        with self._lock:
            now = time.time()
            for k in [k for k, v in self._jtis.items() if v < now]:
                del self._jtis[k]
            self._jtis[jti] = exp

    def __contains__(self, jti: str) -> bool:
        with self._lock:
            return jti in self._jtis

    def clear(self) -> None:
        with self._lock:
            self._jtis.clear()


revoked = _Revoked()


def issue_token(account: Account) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER, "aud": AUDIENCE, "sub": account.username, "pid": account.pid,
        "iat": now, "nbf": now, "exp": now + TOKEN_TTL_S, "jti": secrets.token_hex(16),
    }
    return jwt.encode(claims, secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[Dict]:
    """The verified claims, or None for any invalid, expired, revoked or unknown-user token."""
    if not token or len(token) > 2048:
        return None
    try:
        claims = jwt.decode(
            token, secret(), algorithms=[ALGORITHM], audience=AUDIENCE, issuer=ISSUER,
            options={"require": ["exp", "iat", "nbf", "sub", "jti", "iss", "aud"]}, leeway=10,
        )
    except jwt.PyJWTError:
        return None
    account = ACCOUNTS.get(claims.get("sub"))
    if account is None or claims.get("pid") != account.pid or claims["jti"] in revoked:
        return None
    return claims


def account_for(token: Optional[str]) -> Optional[Account]:
    claims = decode_token(token or "")
    return ACCOUNTS[claims["sub"]] if claims else None


def revoke(token: Optional[str]) -> None:
    claims = decode_token(token or "")
    if claims:
        revoked.add(claims["jti"], float(claims["exp"]))


def safe_next(path: Optional[str]) -> str:
    """Only same-site paths, never //host or /\\host."""
    if not path or not path.startswith("/") or path.startswith("//") or "\\" in path or len(path) > 200:
        return "/"
    return path
