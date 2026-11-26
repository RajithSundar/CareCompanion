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
* Anyone can also sign up. Registered accounts live in data/users.json (mode 600):
  username, display name, a random scrypt salt and hash, and the PMData participant
  whose days the account started from. Writes are atomic and serialised with a lock.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
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
    pid: str           # persona id in demo_data.DEFAULT_APP_STATE, or u + 8 hex for a registered account
    name: str
    role: str
    source: str = ""   # registered accounts: the PMData participant their days came from

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
    rec = users.get(user) if user not in ACCOUNTS else None
    if user in ACCOUNTS:
        salt, expected = _SALTS[user], _HASHES[user]
    elif rec:
        salt, expected = bytes.fromhex(rec["salt"]), bytes.fromhex(rec["hash"])
    else:
        salt, expected = _DUMMY_SALT, _DUMMY_HASH
    ok = hmac.compare_digest(_hash(password, salt), expected)
    if not ok:
        return None
    return ACCOUNTS.get(user) or (_account(rec) if rec else None)


def get_account(username: Optional[str]) -> Optional[Account]:
    if not username:
        return None
    if username in ACCOUNTS:
        return ACCOUNTS[username]
    rec = users.get(username)
    return _account(rec) if rec else None


# --------------------------------------------------------------------------- sign-up

USERNAME_RE = re.compile(r"^[a-z0-9_]{3,20}$")
MIN_PASSWORD = 8
MAX_USERS = int(os.getenv("CC_MAX_USERS", "500"))
RESERVED = set(ACCOUNTS) | {"admin", "root", "system", "support", "coach", "demo"}


class SignupError(ValueError):
    pass


def _account(rec: Dict) -> Account:
    return Account(rec["username"], rec["pid"], rec["name"], rec.get("role") or "CareCompanion member", rec.get("source", ""))


class _Users:
    """Registered accounts, kept in one JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._cache: Optional[Dict[str, Dict]] = None

    def _load(self) -> Dict[str, Dict]:
        if self._cache is None:
            try:
                self._cache = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._cache = {}
        return self._cache

    def get(self, username: str) -> Optional[Dict]:
        with self._lock:
            return self._load().get(username)

    def count(self) -> int:
        with self._lock:
            return len(self._load())

    def add(self, rec: Dict) -> None:
        with self._lock:
            data = dict(self._load())
            if rec["username"] in data:
                raise SignupError("That username is taken.")
            if len(data) >= MAX_USERS:
                raise SignupError("Sign-ups are full right now. Use a demo account instead.")
            data[rec["username"]] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f".{self.path.name}.{secrets.token_hex(4)}")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            self._cache = data

    def reset(self, path: Optional[Path] = None) -> None:
        """For tests: forget the cache, optionally point at another file."""
        with self._lock:
            if path is not None:
                self.path = path
            self._cache = None


users = _Users(Path(os.getenv("CC_USERS_FILE", str(Path(__file__).resolve().parent / "data" / "users.json"))))


def register(username: str, password: str, name: str, source: str, role: str = "CareCompanion member") -> Account:
    """Create an account. Raises SignupError with a message fit to show the user."""
    user = username.strip().lower()
    name = " ".join(name.split())
    if not USERNAME_RE.match(user):
        raise SignupError("Usernames are 3 to 20 characters: lowercase letters, digits or _.")
    if user in RESERVED:
        raise SignupError("That username is taken.")
    if len(password) < MIN_PASSWORD:
        raise SignupError(f"Use a password of at least {MIN_PASSWORD} characters.")
    if password.lower() == user:
        raise SignupError("The password can't be the same as the username.")
    if not 1 <= len(name) <= 40:
        raise SignupError("Enter a name of up to 40 characters.")
    salt = secrets.token_bytes(16)
    rec = {"username": user, "name": name, "role": role, "pid": "u" + secrets.token_hex(4),
           "source": source, "salt": salt.hex(), "hash": _hash(password, salt).hex(), "created": int(time.time())}
    users.add(rec)
    return _account(rec)


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
    account = get_account(claims.get("sub"))
    if account is None or claims.get("pid") != account.pid or claims["jti"] in revoked:
        return None
    return claims


def account_for(token: Optional[str]) -> Optional[Account]:
    claims = decode_token(token or "")
    return get_account(claims["sub"]) if claims else None


def revoke(token: Optional[str]) -> None:
    claims = decode_token(token or "")
    if claims:
        revoked.add(claims["jti"], float(claims["exp"]))


def safe_next(path: Optional[str]) -> str:
    """Only same-site paths, never //host or /\\host."""
    if not path or not path.startswith("/") or path.startswith("//") or "\\" in path or len(path) > 200:
        return "/"
    return path
