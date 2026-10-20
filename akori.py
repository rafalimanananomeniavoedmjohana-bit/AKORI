import contextvars
import traceback
import secrets
import hashlib
import hmac
import inspect
import tempfile
import html
import shutil
import sys
import zipfile
from collections.abc import MutableMapping
import json
import os
from datetime import datetime
import random
import threading
import re
import time
import fitz  # PyMuPDF
from google import genai
from google.genai import types
from google.genai.errors import APIError
import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURATION CLIENT GEMINI & EMBEDDINGS ---
def _clean_key(key):
  """Retire espaces, retours à la ligne, guillemets et caractères invisibles collés avec la clé."""
  key = str(key or "")
  for ch in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u00a0"):
    key = key.replace(ch, "")
  return re.sub(r"\s+", "", key).strip("\"'“”‘’`")


def _find_gemini_key():
  """Clé Gemini : variable d'environnement, puis secret Colab, puis saisie masquée dans Colab."""
  key = _clean_key(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "")
  if key:
    return key
  try:
    from google.colab import userdata  # type: ignore
    key = _clean_key(userdata.get("GEMINI_API_KEY") or "")
  except Exception:
    key = ""
  if not key and "google.colab" in sys.modules and "ipykernel" in sys.modules:
    try:
      import getpass
      key = _clean_key(getpass.getpass("🔑 Clé API Gemini (https://aistudio.google.com/apikey) : "))
    except Exception:
      key = ""
  return key


GEMINI_API_KEY = _find_gemini_key()
GEMINI_KEY_OK = True
GEMINI_KEY_ERROR = ("Clé API Gemini invalide : elle contient des caractères non autorisés (accents, symboles). "
                    "Collez la clé exacte, qui commence par AIza, dans le secret GEMINI_API_KEY.")
if GEMINI_API_KEY and not GEMINI_API_KEY.isascii():
  GEMINI_KEY_OK = False        # un en-tête HTTP ne peut pas contenir d'accents (ex. « votre_clé » collé tel quel)
  GEMINI_API_KEY = ""
  print("⚠️ " + GEMINI_KEY_ERROR)
elif not GEMINI_API_KEY:
  print("⚠️ Aucune clé Gemini : ajoutez le secret Colab GEMINI_API_KEY (icône clé) ou la variable d'environnement. "
        "L'application démarre, mais les réponses de l'IA échoueront.")


def _require_gemini_key():
  if not GEMINI_KEY_OK:
    raise RuntimeError(GEMINI_KEY_ERROR)

GEMINI_HTTP_TIMEOUT_MS = int(os.getenv("GEMINI_HTTP_TIMEOUT_MS", "25000"))   # durée max d'un appel
GEMINI_BUDGET_S = float(os.getenv("GEMINI_BUDGET_S", "70"))                 # durée max d'une génération (tous essais)
_client_key = GEMINI_API_KEY or "missing-key"   # sans clé, l'appli démarre quand même (erreur claire à la 1re question)
try:
    client = genai.Client(api_key=_client_key, http_options=types.HttpOptions(timeout=GEMINI_HTTP_TIMEOUT_MS))
except Exception:
    client = genai.Client(api_key=_client_key)
MODEL_NAME = "gemini-3.5-flash-lite"
FALLBACK_MODEL_NAME = "gemini-3.1-flash-lite"  # Secours gratuit, si accessible au projet.
EXTRA_FALLBACK_MODELS = [
    m.strip() for m in os.getenv("GEMINI_EXTRA_FALLBACKS", "gemini-2.5-flash-lite,gemini-2.5-flash,gemini-2.0-flash").split(",") if m.strip()
]


_AVAILABLE_MODELS = None   # modèles réellement proposés par la clé (None = pas encore découverts)


def _version_key(name):
  nums = re.findall(r"\d+(?:\.\d+)?", name)
  return float(nums[0]) if nums else 0.0


def _discover_models():
  """Interroge Gemini pour connaître les modèles « flash » réellement disponibles avec cette clé.
  Les modèles retirés (404) sont alors écartés et les plus récents ajoutés aux modèles de secours."""
  global _AVAILABLE_MODELS
  if not GEMINI_KEY_OK or not GEMINI_API_KEY:
    return None
  try:
    found = []
    for m in client.models.list():
      name = str(getattr(m, "name", "") or "").replace("models/", "")
      actions = list(getattr(m, "supported_actions", None) or [])
      low = name.lower()
      if not low.startswith("gemini-") or "flash" not in low:
        continue
      if actions and "generateContent" not in actions:
        continue
      if any(x in low for x in ("embed", "image", "tts", "live", "audio", "robotics", "computer", "native", "aqa", "customtools")):
        continue
      found.append(name)
    if found:
      _AVAILABLE_MODELS = sorted(set(found), key=lambda n: (-_version_key(n), "lite" not in n.lower(), n))
      print(f"🤖 Modèles Gemini disponibles : {', '.join(_AVAILABLE_MODELS[:8])}")
    return _AVAILABLE_MODELS
  except Exception as e:
    print(f"⚠️ Liste des modèles Gemini indisponible : {type(e).__name__}")
    return None


def _models_chain():
  """Modèles configurés encore disponibles, puis les autres modèles « flash » découverts, sans doublon."""
  configured = [m for m in (MODEL_NAME, FALLBACK_MODEL_NAME, *EXTRA_FALLBACK_MODELS) if m]
  if _AVAILABLE_MODELS:
    chain = [m for m in configured if m in _AVAILABLE_MODELS]
    chain += [m for m in _AVAILABLE_MODELS if m not in chain][:6]
    if chain:
      return list(dict.fromkeys(chain))
  return list(dict.fromkeys(configured))

threading.Thread(target=lambda: _discover_models(), daemon=True).start()   # sans bloquer le démarrage
_MODEL_COOLDOWN = {}   # modèle -> instant jusqu'auquel on l'évite (surcharge récente) ; inf = modèle inconnu (404)
_MODEL_LOCK = threading.Lock()


def _models_ordered():
  """Modèles à essayer : d'abord ceux qui répondent bien, ceux récemment surchargés en dernier recours."""
  now = time.monotonic()
  with _MODEL_LOCK:
    healthy = [m for m in _models_chain() if _MODEL_COOLDOWN.get(m, 0) <= now]
    resting = sorted((m for m in _models_chain() if _MODEL_COOLDOWN.get(m, 0) > now), key=lambda m: _MODEL_COOLDOWN[m])
  return healthy + resting


def _mark_model(model, code=None):
  """code=None : le modèle a répondu. 404 : inconnu (écarté). Autre : surchargé, repos de 25 s."""
  with _MODEL_LOCK:
    if code is None:
      _MODEL_COOLDOWN.pop(model, None)
    elif code == 404:
      _MODEL_COOLDOWN[model] = time.monotonic() + 600   # modèle inconnu pour cette clé : on le réessaie dans 10 min
    else:
      _MODEL_COOLDOWN[model] = time.monotonic() + 25


GEMINI_MIN_REQUEST_INTERVAL = float(
    os.getenv("GEMINI_MIN_REQUEST_INTERVAL", "0.3")
)
_GEMINI_LAST_REQUEST_TIME = 0.0
_GEMINI_LOCK = threading.Lock()


def _wait_before_gemini_request():
  """Espace les requêtes sans bloquer les autres threads pendant l'attente."""
  global _GEMINI_LAST_REQUEST_TIME
  with _GEMINI_LOCK:
    now = time.monotonic()
    slot = max(now, _GEMINI_LAST_REQUEST_TIME + GEMINI_MIN_REQUEST_INTERVAL)
    _GEMINI_LAST_REQUEST_TIME = slot
  wait_time = slot - now
  if wait_time > 0:
    time.sleep(wait_time)


# Modèle d'embeddings multilingue (français + anglais). Surchargeable : AKORI_EMBED_MODEL.
EMBED_MODEL_NAME = os.getenv("AKORI_EMBED_MODEL", "BAAI/bge-m3")
LEGACY_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # modèle des anciens index (avant BGE-M3)
CHUNK_SIZE = int(os.getenv("AKORI_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("AKORI_CHUNK_OVERLAP", "80"))
RETRIEVAL_K = int(os.getenv("AKORI_RETRIEVAL_K", "4"))           # passages envoyés à Gemini


def _embed_device():
  try:
    import torch  # type: ignore
    return "cuda" if torch.cuda.is_available() else "cpu"
  except Exception:
    return "cpu"


EMBED_DEVICE = _embed_device()
# Seuil de pertinence : similarité cosinus minimale du meilleur passage. En dessous, AKORI répond que
# l'information n'est pas dans le cours au lieu de laisser Gemini improviser. À calibrer (voir calibrate_similarity).
MIN_SIMILARITY = float(os.getenv("AKORI_MIN_SIMILARITY", "0.30" if "bge-m3" in EMBED_MODEL_NAME.lower() else "0.18"))
embed_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": EMBED_DEVICE},
    # Vecteurs normalisés : la distance L2 de FAISS donne alors directement la similarité cosinus.
    encode_kwargs={"batch_size": 32 if EMBED_DEVICE == "cuda" else 16, "normalize_embeddings": True},
)
print(f"🔎 Embeddings : {EMBED_MODEL_NAME} ({EMBED_DEVICE}) · seuil de pertinence {MIN_SIMILARITY:.2f}")
if EMBED_DEVICE == "cpu" and "bge-m3" in EMBED_MODEL_NAME.lower():
  print("   ℹ️ Sans GPU, BGE-M3 est lent : activez un GPU dans Colab (Exécution > Modifier le type d'exécution) "
        "ou indexez vos PDF une fois avec `python akori.py --build`.")


def _warm_up_embeddings():
  """Charge le modèle d'embeddings en arrière-plan : la 1re question n'attend plus."""
  try:
    embed_model.embed_query("warm up")
  except Exception as e:  # pragma: no cover
    print(f"⚠️ Préchauffage des embeddings impossible : {e}")


threading.Thread(target=_warm_up_embeddings, daemon=True).start()


# --- Intentions simples traitées localement (sans appel Gemini) ---
def _normalize_query(query):
    return re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ ]+", " ", query.lower()).strip()


def _is_title_query(query):
    q = _normalize_query(query)
    return q in {
        "titre",
        "title",
        "nom du projet",
        "quel est le titre",
        "c est quoi le titre",
        "c est quoi le titre du projet",
    }


def _is_summary_query(query):
    q = _normalize_query(query)
    return q.startswith("resume") or q.startswith("résume") or "résumé du projet" in q


def _is_technology_query(query):
    q = _normalize_query(query)
    keys = ("technolog", "tech stack", "technique", "technologies")
    return any(k in q for k in keys)


def _is_overview_query(query):
    q = _normalize_query(query)
    keys = ("finalit", "objectif", "but du", "sujet", "de quoi parle", "de quoi traite", "presentation", "présentation",
            "purpose", "goal", "aim of", "what is this about", "overview")
    return any(k in q for k in keys)


def _is_global_query(query):
    return (
        _is_title_query(query)
        or _is_summary_query(query)
        or _is_technology_query(query)
        or _is_overview_query(query)
    )


def _extract_document_title(full_text):
    """
    Extrait une ligne de titre plausible depuis les premières lignes.
    Pour le document AKORI fourni, la première ligne est le titre utile.
    """
    if not full_text:
        return ""

    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in full_text.splitlines()
    ]
    lines = [line for line in lines if line]

    for line in lines[:12]:
        lower = line.lower()
        if len(line) <= 180 and not re.match(r"^(project vision|main objective|phase \d|week \d)", lower):
            return line

    return lines[0] if lines else ""


# --- 2. STOCKAGE PERSISTANT, COMPTES ET DONNÉES PAR UTILISATEUR ---
LOCAL_DATA_DIR = os.path.join(os.getcwd(), "akori_data_store")


def _migrate_local_to_drive(src, dst):
  """Premier passage sur Drive : on y copie les données locales existantes (rien n'est perdu)."""
  try:
    if os.path.isfile(os.path.join(src, "users.json")) and not os.path.exists(os.path.join(dst, "users.json")):
      shutil.copytree(src, dst, dirs_exist_ok=True)
      print(f"📦 Données locales copiées vers Google Drive : {dst}")
  except Exception as e:
    print(f"⚠️ Copie vers Google Drive impossible : {e}")


def _resolve_data_dir():
  """Dossier de données : Google Drive (persistant) si disponible, sinon dossier local (temporaire)."""
  explicit = os.getenv("AKORI_DATA_DIR")
  if explicit:
    return explicit, "/drive/" in explicit.replace("\\", "/").lower()
  drive_root = os.getenv("AKORI_DRIVE_ROOT", "/content/drive/MyDrive")
  in_colab = "google.colab" in sys.modules or os.path.isdir("/content")
  if in_colab and not os.getenv("AKORI_DRIVE_ROOT") and not os.path.isdir(drive_root):
    try:
      from google.colab import drive  # type: ignore
      print("📂 Connexion à Google Drive : autorisez l'accès quand Colab le demande…")
      drive.mount("/content/drive")
    except Exception as e:
      print(f"⚠️ Impossible de monter Google Drive automatiquement ({type(e).__name__}).\n"
            "   Dans une cellule Colab, exécutez d'abord :  from google.colab import drive; drive.mount('/content/drive')")
  if os.path.isdir(drive_root):
    target = os.path.join(drive_root, "AKORI")
    os.makedirs(target, exist_ok=True)
    _migrate_local_to_drive(LOCAL_DATA_DIR, target)
    print(f"✅ Données AKORI enregistrées dans Google Drive : {target}")
    return target, True
  print("⚠️ Google Drive indisponible : les données sont temporaires (perdues à la fermeture de Colab).")
  return LOCAL_DATA_DIR, False


DATA_DIR, STORAGE_ON_DRIVE = _resolve_data_dir()
os.makedirs(DATA_DIR, exist_ok=True)
try:
  _readme = os.path.join(DATA_DIR, "LISEZ-MOI.txt")
  if not os.path.exists(_readme):
    with open(_readme, "w", encoding="utf-8") as _f:
      _f.write("Dossier de données d'AKORI\n\n"
               "  users.json        comptes (mots de passe chiffrés)\n"
               "  users/<compte>/   cours, index, historiques, progression et PDF d'origine de chaque utilisateur\n"
               "  activity.log      journal d'activité\n\n"
               "Pour retrouver vos données : relancez AKORI avec Google Drive monté, ce dossier est relu automatiquement.\n")
except OSError:
  pass
USERS_DIR = os.path.join(DATA_DIR, "users")
LEGACY_DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")  # ancien stockage mono-utilisateur
USERS_FILE = os.path.join(DATA_DIR, "users.json")
TOKENS_FILE = os.path.join(DATA_DIR, "tokens.json")
ACTIVITY_LOG = os.path.join(DATA_DIR, "activity.log")
os.makedirs(USERS_DIR, exist_ok=True)

APP_STARTED_AT = time.time()
TOKEN_TTL_DAYS = 30
MAX_PDF_MB = float(os.getenv("AKORI_MAX_PDF_MB", "40"))

_CUR_USER = contextvars.ContextVar("akori_user", default=None)  # utilisateur de la requête en cours
_STORE_LOCK = threading.RLock()
SESSIONS = {}   # session_hash -> nom d'utilisateur
_USER_DBS = {}  # nom d'utilisateur -> {nom de fichier: données du cours}
_LOGIN_FAILS = {}  # nom d'utilisateur -> (échecs, verrouillé jusqu'à)


# ---- fichiers JSON (écriture atomique) et journal d'activité ----
def _read_json(path, default):
  try:
    with open(path, "r", encoding="utf-8") as f:
      return json.load(f)
  except (OSError, ValueError):
    return default


def _write_json(path, data):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = f"{path}.{os.getpid()}.tmp"
  with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
  os.replace(tmp, path)


def _log(event, user=None, detail="", **measures):
  """Journal d'activité (une ligne JSON par événement) pour le suivi et la maintenance."""
  rec = {
      "t": datetime.now().isoformat(timespec="seconds"),
      "event": event,
      "user": user if user is not None else (_CUR_USER.get() or ""),
      "detail": str(detail)[:300],
  }
  if measures:
    rec["m"] = measures   # mesures (temps de recherche, de 1er mot, total…) relues par la console admin
  try:
    with _STORE_LOCK, open(ACTIVITY_LOG, "a", encoding="utf-8") as f:
      f.write(json.dumps(rec, ensure_ascii=False) + "\n")
  except OSError:
    pass


def _read_log(limit=300):
  try:
    with open(ACTIVITY_LOG, "r", encoding="utf-8") as f:
      lines = f.readlines()[-limit:]
  except OSError:
    return []
  out = []
  for line in lines:
    try:
      out.append(json.loads(line))
    except ValueError:
      continue
  return out[::-1]


# ---- comptes ----
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")
MIN_PASSWORD = 8


def _hash_password(password, salt=None):
  salt = salt or secrets.token_hex(16)
  digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 200_000).hex()
  return salt, digest


def _users():
  return _read_json(USERS_FILE, {})


def _create_user(identifier, password, role="user", allow_username=False, onboarded=None):
  """Retourne (ok, message). Les messages sont en français : l'interface les traduit."""
  ident = str(identifier or "").strip()
  password = str(password or "")
  if not (EMAIL_RE.match(ident) or (allow_username and USERNAME_RE.match(ident))) or len(ident) > 120:
    return False, "⚠️ Adresse e-mail invalide."
  if len(password) < MIN_PASSWORD:
    return False, f"⚠️ Mot de passe trop court ({MIN_PASSWORD} caractères minimum)."
  key = ident.lower()
  with _STORE_LOCK:
    users = _users()
    if key in users:
      return False, "⚠️ Un compte existe déjà avec cet e-mail."
    salt, digest = _hash_password(password)
    users[key] = {
        "username": ident, "salt": salt, "hash": digest, "role": role, "active": True,
        "created_at": datetime.now().isoformat(timespec="seconds"), "last_login": "", "logins": 0,
        "name": "", "level": "", "goal": "", "lang": "",
        "onboarded": (role == "admin") if onboarded is None else bool(onboarded),
    }
    _write_json(USERS_FILE, users)
  os.makedirs(_user_docs_dir(key), exist_ok=True)
  _log("register", key, f"rôle={role}")
  return True, "✅ Compte créé."


def _profile(key):
  u = _users().get(key) or {}
  return {"name": u.get("name", ""), "level": u.get("level", ""), "goal": u.get("goal", ""),
          "lang": u.get("lang", ""), "onboarded": u.get("onboarded", True)}   # comptes anciens : déjà prêts


def _save_profile(key, **fields):
  with _STORE_LOCK:
    users = _users()
    if key in users:
      users[key].update(fields)
      _write_json(USERS_FILE, users)


def _display_name(key):
  if not key:
    return ""
  name = _profile(key)["name"]
  if name:
    return name
  ident = (_users().get(key) or {}).get("username", key)
  return str(ident).split("@")[0]


def _authenticate(username, password):
  """Retourne (clé_utilisateur | None, message)."""
  key = str(username or "").strip().lower()
  fails, locked_until = _LOGIN_FAILS.get(key, (0, 0))
  if locked_until > time.time():
    return None, "⏳ Trop de tentatives. Réessayez dans une minute."
  user = _users().get(key)
  ok = False
  if user:
    _, digest = _hash_password(str(password or ""), user["salt"])
    ok = hmac.compare_digest(digest, user["hash"])
  if not ok:
    fails += 1
    _LOGIN_FAILS[key] = (fails, time.time() + 60 if fails >= 5 else 0)
    _log("login_fail", key)
    return None, "⚠️ Identifiants incorrects."
  if not user.get("active", True):
    _log("login_blocked", key)
    return None, "⛔ Ce compte est désactivé. Contactez l'administrateur."
  _LOGIN_FAILS.pop(key, None)
  with _STORE_LOCK:
    users = _users()
    users[key]["last_login"] = datetime.now().isoformat(timespec="seconds")
    users[key]["logins"] = int(users[key].get("logins", 0)) + 1
    _write_json(USERS_FILE, users)
  _log("login", key)
  return key, ""


def _user_role(key):
  return (_users().get(key) or {}).get("role", "user")


def _is_admin(key=None):
  key = key or _CUR_USER.get()
  return bool(key) and _user_role(key) == "admin"


# ---- jetons de reconnexion automatique (« rester connecté ») ----
def _token_hash(token):
  return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _issue_token(key, days=TOKEN_TTL_DAYS):
  token = secrets.token_urlsafe(32)
  with _STORE_LOCK:
    tokens = _read_json(TOKENS_FILE, {})
    now = time.time()
    tokens = {h: t for h, t in tokens.items() if t.get("exp", 0) > now}
    tokens[_token_hash(token)] = {"user": key, "exp": now + days * 86400}
    _write_json(TOKENS_FILE, tokens)
  return token


def _user_from_token(token):
  if not token:
    return None
  rec = _read_json(TOKENS_FILE, {}).get(_token_hash(str(token)))
  if not rec or rec.get("exp", 0) < time.time():
    return None
  user = _users().get(rec["user"])
  return rec["user"] if user and user.get("active", True) else None


def _revoke_token(token):
  if not token:
    return
  with _STORE_LOCK:
    tokens = _read_json(TOKENS_FILE, {})
    if tokens.pop(_token_hash(str(token)), None) is not None:
      _write_json(TOKENS_FILE, tokens)


def _revoke_user_tokens(key):
  with _STORE_LOCK:
    tokens = _read_json(TOKENS_FILE, {})
    kept = {h: t for h, t in tokens.items() if t.get("user") != key}
    if len(kept) != len(tokens):
      _write_json(TOKENS_FILE, kept)


# ---- données par utilisateur ----
def _user_docs_dir(key):
  return os.path.join(USERS_DIR, key, "documents")


DEFAULT_ADMIN_PASSWORD = "admin123"   # identifiants par défaut : admin / admin123 (à changer dans la console admin)


def _bootstrap_admin():
  """Crée le compte administrateur au premier lancement et migre l'ancien stockage."""
  users = _users()
  # Accès admin le plus simple : créez un compte normal, puis lancez avec AKORI_ADMIN_USER=<votre e-mail> :
  # ce compte devient administrateur (aucun mot de passe supplémentaire).
  wanted = (os.getenv("AKORI_ADMIN_USER") or "").strip().lower()
  if wanted in users and users[wanted].get("role") != "admin":
    with _STORE_LOCK:
      users[wanted]["role"] = "admin"
      _write_json(USERS_FILE, users)
    _log("promote_admin", user=wanted, detail="via AKORI_ADMIN_USER")
    print(f"🛡  « {wanted} » est maintenant administrateur.")
  # Ancien compte admin créé avec un mot de passe aléatoire (fichier ADMIN_INITIAL_PASSWORD.txt encore là) :
  # il repasse aux identifiants par défaut admin / admin123.
  legacy_note = os.path.join(DATA_DIR, "ADMIN_INITIAL_PASSWORD.txt")
  if "admin" in users and users["admin"].get("role") == "admin" and os.path.exists(legacy_note):
    with _STORE_LOCK:
      users = _users()
      users["admin"]["salt"], users["admin"]["hash"] = _hash_password(DEFAULT_ADMIN_PASSWORD)
      users["admin"]["active"] = True
      _write_json(USERS_FILE, users)
    with open(legacy_note, "w", encoding="utf-8") as f:
      f.write(f"Compte administrateur AKORI\nIdentifiant : admin\nMot de passe : {DEFAULT_ADMIN_PASSWORD}\n")
    print(f"🛡  Compte administrateur : admin / {DEFAULT_ADMIN_PASSWORD}")
  if not any(u.get("role") == "admin" for u in users.values()):
    name = os.getenv("AKORI_ADMIN_USER", "admin")
    password = os.getenv("AKORI_ADMIN_PASSWORD") or DEFAULT_ADMIN_PASSWORD
    generated = True
    ok, msg = _create_user(name, password, role="admin", allow_username=True, onboarded=True)
    if ok and generated:
      path = os.path.join(DATA_DIR, "ADMIN_INITIAL_PASSWORD.txt")
      with open(path, "w", encoding="utf-8") as f:
        f.write(f"Compte administrateur AKORI\nIdentifiant : {name}\nMot de passe : {password}\n"
                "Changez-le dès la première connexion, puis supprimez ce fichier.\n")
      print("=" * 64)
      print(f"🛡  AKORI — compte administrateur créé : {name} / {password}")
      print(f"    (aussi enregistré dans {path})")
      print("=" * 64)
    users = _users()
  admin_key = next((k for k, u in users.items() if u.get("role") == "admin"), None)
  # Mot de passe admin oublié : AKORI_RESET_ADMIN=1 + AKORI_ADMIN_PASSWORD=<nouveau> au lancement.
  new_password = os.getenv("AKORI_ADMIN_PASSWORD")
  if admin_key and new_password and os.getenv("AKORI_RESET_ADMIN") == "1" and len(new_password) >= MIN_PASSWORD:
    with _STORE_LOCK:
      users = _users()
      users[admin_key]["salt"], users[admin_key]["hash"] = _hash_password(new_password)
      users[admin_key]["active"] = True
      _write_json(USERS_FILE, users)
    _revoke_user_tokens(admin_key)
    _log("admin_password_reset", user=admin_key, detail="via AKORI_RESET_ADMIN")
    print(f"🛡  Mot de passe du compte administrateur « {admin_key} » réinitialisé.")
  # Anciens cours (stockage unique) -> rattachés à l'administrateur, rien n'est perdu.
  if admin_key and os.path.isdir(LEGACY_DOCUMENTS_DIR) and os.listdir(LEGACY_DOCUMENTS_DIR):
    target = _user_docs_dir(admin_key)
    os.makedirs(target, exist_ok=True)
    for entry in os.listdir(LEGACY_DOCUMENTS_DIR):
      dst = os.path.join(target, entry)
      if not os.path.exists(dst):
        shutil.move(os.path.join(LEGACY_DOCUMENTS_DIR, entry), dst)
    print(f"📦 Anciens documents migrés vers le compte « {admin_key} ».")


def _user_pdfs_dir(key):
  return os.path.join(USERS_DIR, key, "pdfs")


def _doc_id_from_filename(filename):
  return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]


def _require_user():
  key = _CUR_USER.get()
  if not key:
    raise PermissionError("Connexion requise.")
  return key


def _doc_folder(filename):
  return os.path.join(_user_docs_dir(_require_user()), _doc_id_from_filename(filename))


def _doc_metadata_path(filename):
  return os.path.join(_doc_folder(filename), "metadata.json")


def _doc_index_path(filename):
  return os.path.join(_doc_folder(filename), "faiss_index")


def clean_latex_artifacts(text: str) -> str:
    """Nettoie ou convertit les artefacts LaTeX bruts pour un affichage lisible."""
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    return text


def _doc_metadata(filename, data):
  return {
      "filename": filename,
      "title": data.get("title", ""),
      "chunks_count": int(data.get("chunks_count", 0)),
      "file_size": int(data.get("file_size", 0)),
      "content_hash": data.get("content_hash", ""),
      "added_at": data.get("added_at", ""),
      "embed_model": data.get("embed_model", EMBED_MODEL_NAME),
      "ocr_pages": int(data.get("ocr_pages", 0) or 0),
      "pins": data.get("pins", []),
      "history": data.get("history", []),
      "progress": data.get("progress", {"flashcards": {}, "quiz_attempts": []}),
      # état des sections de révision de ce PDF (restauré quand on y revient)
      "flashcards": data.get("flashcards", []),
      "flash_index": int(data.get("flash_index", 0)),
      "quiz": data.get("quiz", {}),
      "summary_cache": data.get("summary_cache", {}),
  }


def _save_document(filename):
  """Sauvegarde complète (index FAISS + métadonnées)."""
  data = documents_db.get(filename)
  if not data:
    return
  os.makedirs(_doc_folder(filename), exist_ok=True)
  data["vector_db"].save_local(_doc_index_path(filename))
  _write_json(_doc_metadata_path(filename), _doc_metadata(filename, data))


def _save_history(filename):
  """Sauvegarde uniquement les métadonnées JSON, sans réécrire l'index FAISS."""
  data = documents_db.get(filename)
  if not data:
    return
  os.makedirs(_doc_folder(filename), exist_ok=True)
  _write_json(_doc_metadata_path(filename), _doc_metadata(filename, data))


class _LazyVectorStore:
  """Index FAISS d'un cours chargé seulement à la première utilisation (recherche, résumé, quiz…).

  Si l'index a été calculé avec un ancien modèle d'embeddings, `rebuild` le recalcule (une seule fois)
  à partir du PDF d'origine conservé dans le dossier de l'utilisateur."""

  def __init__(self, path, rebuild=None):
    self._path = path
    self._rebuild = rebuild
    self._vs = None
    self._lock = threading.Lock()

  def _load(self):
    with self._lock:
      if self._vs is None:
        if self._rebuild is not None:
          self._vs = self._rebuild()
          self._rebuild = None
        else:
          self._vs = FAISS.load_local(self._path, embed_model, allow_dangerous_deserialization=True)
      return self._vs

  def __getattr__(self, name):
    return getattr(self._load(), name)


def _load_user_documents(key):
  docs = {}
  root = _user_docs_dir(key)
  if not os.path.isdir(root):
    return docs
  for entry in sorted(os.listdir(root)):
    folder = os.path.join(root, entry)
    metadata_path = os.path.join(folder, "metadata.json")
    index_path = os.path.join(folder, "faiss_index")
    if not os.path.isdir(folder) or not os.path.exists(metadata_path) or not os.path.exists(index_path):
      continue
    try:
      metadata = _read_json(metadata_path, None)
      filename = (metadata or {}).get("filename")
      if not filename:
        continue
      model_used = metadata.get("embed_model") or LEGACY_EMBED_MODEL
      doc_entry = {}
      rebuild = None
      if model_used != EMBED_MODEL_NAME:
        rebuild = (lambda k=key, f=filename, p=index_path, e=doc_entry: _rebuild_index_for(k, f, p, e))
      vector_db = _LazyVectorStore(index_path, rebuild)
      doc_entry.update({
          "embed_model": model_used,
          "ocr_pages": int(metadata.get("ocr_pages", 0) or 0),
          "pins": metadata.get("pins", []),
          "vector_db": vector_db,
          "history": metadata.get("history", []),
          "chunks_count": int(metadata.get("chunks_count", 0)),
          "full_text": "",
          "title": metadata.get("title", ""),
          "file_size": int(metadata.get("file_size", 0)),
          "content_hash": metadata.get("content_hash", ""),
          "added_at": metadata.get("added_at", ""),
          "progress": metadata.get("progress", {"flashcards": {}, "quiz_attempts": []}),
          "flashcards": metadata.get("flashcards", []),
          "flash_index": int(metadata.get("flash_index", 0)),
          "quiz": metadata.get("quiz", {}),
          "summary_cache": metadata.get("summary_cache", {}),
      })
      docs[filename] = doc_entry
    except Exception as e:
      print(f"⚠️ Impossible de restaurer '{entry}' ({key}) : {e}")
  return dict(sorted(docs.items(), key=lambda kv: kv[1].get("added_at", "")))


def _db_for(key):
  if not key:
    return {}
  with _STORE_LOCK:
    if key not in _USER_DBS:
      _USER_DBS[key] = _load_user_documents(key)
    return _USER_DBS[key]


class _UserDocs(MutableMapping):
  """Dictionnaire des cours de l'utilisateur de la requête en cours (isolation entre comptes)."""

  def _d(self, write=False):
    key = _CUR_USER.get()
    if write and not key:
      raise PermissionError("Connexion requise.")
    return _db_for(key)

  def __getitem__(self, k):
    return self._d()[k]

  def __setitem__(self, k, v):
    self._d(True)[k] = v

  def __delitem__(self, k):
    del self._d(True)[k]

  def __iter__(self):
    return iter(list(self._d()))

  def __len__(self):
    return len(self._d())


documents_db = _UserDocs()


def _dir_size(path):
  total = 0
  for dp, _, files in os.walk(path):
    for f in files:
      try:
        total += os.path.getsize(os.path.join(dp, f))
      except OSError:
        pass
  return total


def _delete_user_data(key):
  _USER_DBS.pop(key, None)
  shutil.rmtree(os.path.join(USERS_DIR, key), ignore_errors=True)


_bootstrap_admin()


# --- Progression réelle, persistante et calculée uniquement à partir des interactions ---
def _default_progress():
  return {"flashcards": {}, "quiz_attempts": []}


def _progress_for(filename):
  data = documents_db.get(filename)
  if not data:
    return _default_progress()
  progress = data.setdefault("progress", _default_progress())
  progress.setdefault("flashcards", {})
  progress.setdefault("quiz_attempts", [])
  return progress


def _save_progress(filename):
  _save_history(filename)


def _course_progress_percent(filename):
  if not filename or filename not in documents_db:
    return 0
  progress = _progress_for(filename)
  flash = progress.get("flashcards", {})
  known = sum(1 for v in flash.values() if v == "known")
  flash_total = len(flash)
  attempts = progress.get("quiz_attempts", [])
  quiz_pct = (sum(float(a.get("score", 0)) / max(1, int(a.get("total", 1))) for a in attempts) / len(attempts) * 100) if attempts else 0
  components = []
  if flash_total:
    components.append((known / flash_total) * 100)
  if attempts:
    components.append(quiz_pct)
  return round(sum(components) / len(components)) if components else 0


def _progress_topics(filename):
  attempts = _progress_for(filename).get("quiz_attempts", [])
  topics = {}
  for attempt in attempts:
    for item in attempt.get("items", []):
      topic = str(item.get("topic") or "Général").strip() or "Général"
      bucket = topics.setdefault(topic, {"correct": 0, "total": 0})
      bucket["total"] += 1
      bucket["correct"] += 1 if item.get("correct") else 0
  return topics


def _global_progress_stats():
  if not documents_db:
    return {"percent": 0, "courses": 0, "started": 0, "flashcards": 0, "known": 0, "quizzes": 0, "quiz_avg": 0}
  values = [_course_progress_percent(name) for name in documents_db]
  quiz_points = quiz_total = all_flash = all_known = all_quizzes = 0
  for name in documents_db:
    prog = _progress_for(name)
    flash = prog.get("flashcards", {})
    all_flash += len(flash)
    all_known += sum(1 for state in flash.values() if state == "known")
    attempts = prog.get("quiz_attempts", [])
    all_quizzes += len(attempts)
    for attempt in attempts:
      quiz_points += int(attempt.get("score", 0))
      quiz_total += int(attempt.get("total", 0))
  return {"percent": round(sum(values)/len(values)) if values else 0, "courses": len(documents_db), "started": sum(1 for v in values if v > 0), "flashcards": all_flash, "known": all_known, "quizzes": all_quizzes, "quiz_avg": round((quiz_points/quiz_total)*100) if quiz_total else 0}

def global_progress_html(active_doc=None):
  if not documents_db:
    return """<div class='progress-empty'><b>Votre progression commencera ici.</b><br>Importez un cours pour créer votre premier suivi. Tant qu'aucun cours n'est chargé, la progression reste à 0 %.</div>"""
  st = _global_progress_stats(); rows=[]
  for name,data in documents_db.items():
    cp=_course_progress_percent(name); active=" active-progress-course" if name==active_doc else ""
    rows.append(f"<div class='global-course-row{active}'><div class='global-course-main'><span class='global-course-dot'></span><div><b>{_escape_html(data.get('title') or name)}</b><small>{_escape_html(name)}</small></div></div><div class='global-course-bar'><div style='width:{cp}%;'></div></div><strong>{cp}%</strong></div>")
  pct=st['percent']
  return f"""<div class='global-progress-hero'><div class='global-progress-ring' style='--progress:{pct}%;'><span>{pct}%</span></div><div class='global-progress-copy'><div class='eyebrow'>VUE D'ENSEMBLE</div><h2>Progression globale</h2><p>Synthèse de tous vos cours réellement importés et de vos interactions de révision.</p></div></div><div class='global-metrics'><div><b>{st['courses']}</b><span>Cours suivis</span></div><div><b>{st['started']}</b><span>Cours commencés</span></div><div><b>{st['known']}/{st['flashcards']}</b><span>Flashcards maîtrisées</span></div><div><b>{st['quiz_avg']}%</b><span>Moyenne des quiz</span></div></div><div class='global-course-list'><div class='global-list-title'>Progression par cours</div>{''.join(rows)}</div><div class='global-detail-hint'>Le détail pédagogique reste lié au cours sélectionné : points forts, points faibles, flashcards et résultats des quiz.</div>"""

def progress_detail_html(active_doc=None):
  if not documents_db:
    return """<div class='progress-empty'><b>Aucune progression à afficher.</b><br>Importez un cours puis utilisez les flashcards ou terminez un quiz pour commencer à construire votre progression.</div>"""
  if not active_doc or active_doc not in documents_db:
    active_doc = next(iter(documents_db))
  data = documents_db[active_doc]
  progress = _progress_for(active_doc)
  pct = _course_progress_percent(active_doc)
  flash = progress.get("flashcards", {})
  known = sum(1 for v in flash.values() if v == "known")
  review = sum(1 for v in flash.values() if v == "review")
  attempts = progress.get("quiz_attempts", [])
  total_questions = sum(int(a.get("total", 0)) for a in attempts)
  total_correct = sum(int(a.get("score", 0)) for a in attempts)
  quiz_avg = round((total_correct / total_questions) * 100) if total_questions else 0
  topics = _progress_topics(active_doc)
  strengths = [(t, round(v["correct"] / v["total"] * 100)) for t,v in topics.items() if v["total"] and v["correct"] / v["total"] >= .75]
  weaknesses = [(t, round(v["correct"] / v["total"] * 100)) for t,v in topics.items() if v["total"] and v["correct"] / v["total"] < .60]
  if not topics:
    strengths_html = "<div class='progress-note'>Pas encore assez de réponses de quiz pour identifier un point fort.</div>"
    weaknesses_html = "<div class='progress-note'>Pas encore assez de réponses de quiz pour identifier un point faible.</div>"
  else:
    strengths_html = "".join(f"<div class='strength-row'><span>{_escape_html(t)}</span><b>{score}%</b></div>" for t,score in sorted(strengths,key=lambda x:-x[1])) or "<div class='progress-note'>Aucun point fort clairement établi pour le moment.</div>"
    weaknesses_html = "".join(f"<div class='weakness-row'><span>{_escape_html(t)}</span><b>{score}%</b></div>" for t,score in sorted(weaknesses,key=lambda x:x[1])) or "<div class='progress-note'>Aucun point faible clairement établi pour le moment.</div>"
  return f"""
  <div class='progress-hero'>
    <div class='progress-ring'><span>{pct}%</span></div>
    <div><div class='progress-title'>Progression réelle · {_escape_html(data.get('title') or active_doc)}</div>
    <div class='progress-sub'>Calculée uniquement à partir des interactions enregistrées sur ce cours.</div></div>
  </div>
  <div class='progress-metrics'>
    <div><b>{known}</b><span>Flashcards maîtrisées</span></div>
    <div><b>{review}</b><span>À revoir</span></div>
    <div><b>{quiz_avg}%</b><span>Moyenne des quiz</span></div>
    <div><b>{len(attempts)}</b><span>Quiz terminés</span></div>
  </div>
  <div class='progress-columns'>
    <div class='progress-box'><h3>✦ Points forts</h3>{strengths_html}</div>
    <div class='progress-box'><h3>↗ Points faibles</h3>{weaknesses_html}</div>
  </div>
  """

# --- 3. INDEXATION & UTILITAIRES ---
def _file_sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for block in iter(lambda: f.read(1024 * 1024), b""):
      h.update(block)
  return h.hexdigest()


def _looks_like_heading(line):
  """Heuristique : une ligne courte, sans ponctuation finale, qui ressemble à un titre de section."""
  s = line.strip()
  if not (3 <= len(s) <= 90) or s[-1] in ".,;:" or not re.search(r"[A-Za-zÀ-ÿ]", s):
    return False
  if sum(c.isdigit() for c in s) > len(s) // 2 or len(s.split()) > 12:
    return False
  return s[0].isupper() or s[0].isdigit()


def _page_headings(text):
  marks, pos = [], 0
  for line in text.splitlines(keepends=True):
    if _looks_like_heading(line):
      marks.append((pos, line.strip()))
    pos += len(line)
  return marks


# ---- OCR : pages scannées (sans couche de texte) ------------------------------------------------
OCR_ENGINE = os.getenv("AKORI_OCR", "auto").strip().lower()      # auto | gemini | tesseract | off
OCR_MAX_PAGES = int(os.getenv("AKORI_OCR_MAX_PAGES", "80"))      # pages lues par import (au-delà : ignorées)
OCR_DPI = int(os.getenv("AKORI_OCR_DPI", "150"))
OCR_MIN_CHARS = 20                                              # moins de caractères + une image = page scannée


def _tesseract_available():
  return shutil.which("tesseract") is not None


def _ocr_engine_name():
  """Moteur réellement utilisé : Tesseract s'il est installé (local, gratuit), sinon Gemini Vision."""
  if OCR_ENGINE == "off":
    return "off"
  if OCR_ENGINE == "tesseract" or (OCR_ENGINE == "auto" and _tesseract_available()):
    return "tesseract"
  return "gemini"


def _ocr_label():
  engine = _ocr_engine_name()
  return {"off": "désactivé", "tesseract": "Tesseract (local)", "gemini": "Gemini Vision"}[engine]


def _page_has_images(page):
  try:
    return bool(page.get_images())
  except Exception:
    return True


def _ocr_tesseract(page):
  for lang in ("fra+eng", "eng"):
    try:
      textpage = page.get_textpage_ocr(language=lang, dpi=OCR_DPI, full=True)
      return page.get_text(textpage=textpage)
    except Exception:
      continue
  return ""


_OCR_PROMPT = ("Transcris fidèlement tout le texte de cette page de cours scannée, dans l'ordre de lecture. "
               "Conserve les titres, les listes et les formules. N'ajoute aucun commentaire. "
               "Si la page ne contient aucun texte, réponds uniquement : VIDE")


def _ocr_gemini(png_bytes):
  """Transcription d'une page (image PNG) par Gemini Vision, avec les mêmes modèles de secours que le chat."""
  if not GEMINI_KEY_OK:
    return ""
  last_error, deadline = None, time.monotonic() + 60
  for round_no in range(2):
    for model_target in _models_ordered():
      if time.monotonic() > deadline:
        return ""
      try:
        _wait_before_gemini_request()
        response = client.models.generate_content(
            model=model_target,
            contents=[types.Part.from_bytes(data=png_bytes, mime_type="image/png"), _OCR_PROMPT],
            config=types.GenerateContentConfig(max_output_tokens=3000),
        )
        _mark_model(model_target)
        text = (response.text or "").strip()
        return "" if text.upper().rstrip(".") == "VIDE" else text
      except APIError as e:
        last_error = e
        if e.code in (404, 429, 500, 503, 504):
          _mark_model(model_target, e.code)
          continue
        print(f"⚠️ OCR : erreur Gemini {e.code} ({e}).")
        return ""
      except Exception as e:
        last_error = e
        _mark_model(model_target, "net")
    time.sleep(1.5)
  print(f"⚠️ OCR impossible pour une page : {last_error}")
  return ""


def _ocr_pages(doc, numbers, progress=None):
  """Texte lu par OCR pour les pages demandées -> {numéro de page: texte}."""
  results, total = {}, len(numbers)
  if _ocr_engine_name() == "tesseract":
    for done, number in enumerate(numbers, start=1):
      results[number] = _ocr_tesseract(doc[number - 1])
      if progress:
        progress(done, total)
    return results
  images = {number: doc[number - 1].get_pixmap(dpi=OCR_DPI).tobytes("png") for number in numbers}   # fitz : thread principal
  from concurrent.futures import ThreadPoolExecutor, as_completed
  with ThreadPoolExecutor(max_workers=3) as pool:
    futures = {pool.submit(_ocr_gemini, png): number for number, png in images.items()}
    for done, future in enumerate(as_completed(futures), start=1):
      try:
        results[futures[future]] = future.result()
      except Exception:
        results[futures[future]] = ""
      if progress:
        progress(done, total)
  return results


def _run_progress(work, interval=0.5):
  """Exécute work(callback) en arrière-plan ; produit ("status", texte) pendant la lecture OCR, retourne le résultat."""
  state, box = {"msg": None}, {}

  def callback(done, total):
    state["msg"] = f"⏳ Lecture OCR des pages scannées {done}/{total}…"

  def runner():
    try:
      box["value"] = work(callback)
    except BaseException as exc:
      box["error"] = exc

  thread = threading.Thread(target=runner, daemon=True)
  thread.start()
  last = None
  while thread.is_alive():
    thread.join(interval)
    if state["msg"] and state["msg"] != last and thread.is_alive():
      last = state["msg"]
      yield ("status", last)
  if "error" in box:
    raise box["error"]
  return box["value"]


def _extract_chunks(file_path, progress=None, info=None):
  """PDF -> (fragments, métadonnées {page, section}, texte complet). Les fragments ne chevauchent pas deux pages.
  Les pages scannées (image sans texte) sont lues par OCR ; info["ocr_pages"] dit combien."""
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True)
  texts, metas, current = [], [], ""
  with fitz.open(file_path) as doc:
    pages_text, scanned = [], []
    for number, page in enumerate(doc, start=1):
      page_text = page.get_text()
      pages_text.append(page_text)
      if OCR_ENGINE != "off" and len(page_text.strip()) < OCR_MIN_CHARS and _page_has_images(page):
        scanned.append(number)
    if scanned:
      read = _ocr_pages(doc, scanned[:OCR_MAX_PAGES], progress)
      for number, text in read.items():
        if text.strip():
          pages_text[number - 1] = text
      if info is not None:
        info["ocr_pages"] = sum(1 for t in read.values() if t.strip())
        info["ocr_skipped"] = max(0, len(scanned) - OCR_MAX_PAGES)
    for number, page_text in enumerate(pages_text, start=1):
      if not page_text.strip():
        continue
      marks = _page_headings(page_text)
      for piece in splitter.create_documents([page_text]):
        start = max(0, int((piece.metadata or {}).get("start_index", 0) or 0))
        section = current
        for pos, head in marks:
          if pos <= start:
            section = head
          else:
            break
        texts.append(piece.page_content)
        metas.append({"page": number, "section": section})
      if marks:
        current = marks[-1][1]
  return texts, metas, "".join(pages_text)


def _embed_texts(texts, progress=None, batch=64):
  vectors = []
  for i in range(0, len(texts), batch):
    vectors.extend(embed_model.embed_documents(texts[i:i + batch]))
    if progress:
      progress(min(i + batch, len(texts)), len(texts))
  return vectors


def _rebuild_index_for(key, filename, index_path, entry):
  """Recalcule l'index d'un ancien cours avec le modèle d'embeddings actuel (PDF d'origine requis)."""
  pdf = os.path.join(_user_pdfs_dir(key), filename)
  if not os.path.isfile(pdf):
    raise RuntimeError("Ce cours a été indexé avec un ancien modèle de recherche : réimportez le PDF pour le mettre à jour.")
  print(f"♻️ Réindexation de « {filename} » avec {EMBED_MODEL_NAME}…")
  texts, metas, _ = _extract_chunks(pdf)
  if not texts:
    raise RuntimeError("Ce cours ne contient plus de texte exploitable : réimportez le PDF.")
  vs = FAISS.from_embeddings(list(zip(texts, _embed_texts(texts))), embed_model, metadatas=metas)
  vs.save_local(index_path)
  entry["embed_model"] = EMBED_MODEL_NAME
  entry["chunks_count"] = len(texts)
  meta_path = os.path.join(os.path.dirname(index_path), "metadata.json")
  meta = _read_json(meta_path, {}) or {}
  meta["embed_model"] = EMBED_MODEL_NAME
  meta["chunks_count"] = len(texts)
  _write_json(meta_path, meta)
  _log("reindex", user=key, detail=f"{filename} · {len(texts)} fragments")
  return vs


def indexer_pdf_stream(file_path):
  """Importe un PDF par étapes. Produit ("status", texte) puis ("done", nom_du_fichier | None, texte)."""
  if not file_path:
    yield ("done", None, "⚠️ Aucun fichier sélectionné.")
    return

  filename = os.path.basename(file_path)
  try:
    size = os.path.getsize(file_path)
    if size > MAX_PDF_MB * 1024 * 1024:
      yield ("done", None, f"⚠️ Fichier trop volumineux (maximum {int(MAX_PDF_MB)} Mo).")
      return

    content_hash = _file_sha256(file_path)
    previous = documents_db.get(filename)
    if previous and previous.get("content_hash") == content_hash and previous.get("embed_model", EMBED_MODEL_NAME) == EMBED_MODEL_NAME:
      # Même fichier déjà indexé : on réutilise tout, c'est instantané.
      yield ("done", filename, f"✅ '{filename}' est déjà indexé : réutilisé instantanément.")
      return

    yield ("status", "⏳ Extraction du texte…")
    ocr_info = {}
    chunks, metas, texte_complet = yield from _run_progress(lambda cb: _extract_chunks(file_path, cb, ocr_info))
    if not texte_complet.strip() or not chunks:
      if ocr_info.get("ocr_pages") == 0:
        yield ("done", None, "⚠️ Aucun texte n'a pu être lu dans ce PDF scanné (OCR sans résultat).")
      else:
        yield ("done", None, "⚠️ Aucun texte exploitable dans ce PDF (document scanné ?).")
      return
    ocr_pages = int(ocr_info.get("ocr_pages") or 0)

    total = len(chunks)
    yield ("status", f"⏳ Indexation 0/{total}…")

    vectors = []
    batch = 64
    for i in range(0, total, batch):
      vectors.extend(embed_model.embed_documents(chunks[i:i + batch]))
      yield ("status", f"⏳ Indexation {min(i + batch, total)}/{total}…")
    vector_db = FAISS.from_embeddings(list(zip(chunks, vectors)), embed_model, metadatas=metas)

    same_name = previous is not None
    # Même PDF réindexé (changement de modèle) : on garde l'historique, la progression et les révisions.
    keep = previous if (previous and previous.get("content_hash") == content_hash) else {}
    documents_db[filename] = {
        "vector_db": vector_db,
        "embed_model": EMBED_MODEL_NAME,
        "ocr_pages": ocr_pages,
        "pins": list(keep.get("pins", [])),
        "history": list(keep.get("history", [])),
        "progress": keep.get("progress") or _default_progress(),
        "flashcards": keep.get("flashcards", []),
        "flash_index": keep.get("flash_index", 0),
        "quiz": keep.get("quiz", {}),
        "summary_cache": keep.get("summary_cache", {}),
        "chunks_count": total,
        "full_text": "",
        "title": _extract_document_title(texte_complet),
        "file_size": size,
        "content_hash": content_hash,
        "added_at": (previous or {}).get("added_at") if same_name and (previous or {}).get("added_at") else datetime.now().isoformat(timespec="seconds"),
    }
    yield ("status", "⏳ Enregistrement…")
    _save_document(filename)
    try:  # copie du PDF d'origine : on le retrouve tel quel dans Google Drive
      os.makedirs(_user_pdfs_dir(_require_user()), exist_ok=True)
      shutil.copy2(file_path, os.path.join(_user_pdfs_dir(_require_user()), filename))
    except Exception as e:
      print(f"⚠️ Copie du PDF d'origine impossible : {e}")
    _log("upload", detail=f"{filename} · {total} fragments · {size // 1024} Ko")
    if ocr_pages:
      yield ("done", filename, f"✅ '{filename}' indexé avec succès ({total} fragments, OCR sur {ocr_pages} pages).")
    else:
      yield ("done", filename, f"✅ '{filename}' indexé avec succès ({total} fragments).")
  except PermissionError:
    yield ("done", None, "⚠️ Session expirée : rechargez la page pour vous reconnecter.")
  except Exception as e:
    _log("upload_error", detail=f"{filename} · {e}")
    yield ("done", None, f"❌ Erreur lors de l'indexation : {str(e)}")


UI_CHAT_LIMIT = 30      # messages envoyés au navigateur (le reste reste archivé)
UI_HISTORY_LIMIT = 80   # messages affichés dans l'onglet Historique
HISTORY_KEEP = 300      # messages conservés par cours


def _fmt_ms(ms):
  ms = float(ms or 0)
  return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def _source_html(src):
  page = src.get("page")
  head = f"Page {int(page)}" if page else "Extrait"
  section = str(src.get("section") or "").strip()
  meta = f"{head} · {_escape_html(section)}" if section else head
  quote = _escape_html(str(src.get("excerpt") or "")).replace("$", "&#36;")
  return (f"<li><b>[{int(src.get('n', 0))}]</b> <span class='ak-src-meta'>{meta}</span>"
          f"<div class='ak-src-quote'>{quote}</div></li>")


def _message_actions_html(index, item):
  """Barre d'actions sous une réponse : épingler en chapitre, lire à voix haute, voir les sources."""
  sources = item.get("sources") or []
  buttons = [
      f"<button type='button' class='ak-act ak-a-pin ak-i-{index}'>📌 <span>Épingler comme chapitre</span></button>",
      "<button type='button' class='ak-act ak-a-speak'>🔊 <span>Lire à voix haute</span></button>",
  ]
  if sources:
    buttons.append(f"<button type='button' class='ak-act ak-a-sources'>📎 <span>Sources ({len(sources)})</span></button>")
  timing = item.get("timing") or {}
  if timing.get("total"):
    buttons.append(f"<span class='ak-timing' title='Temps de réponse'>⏱ {_fmt_ms(timing['total'])}</span>")
  html_out = "<div class='ak-msg-actions'>" + "".join(buttons) + "</div>"
  if sources:
    html_out += "<ol class='ak-src-list' hidden>" + "".join(_source_html(x) for x in sources) + "</ol>"
  return "\n\n" + html_out


def _chat_history_for_ui(doc_name):
  data = documents_db.get(doc_name, {})
  raw = data.get("history", [])
  start = max(0, len(raw) - UI_CHAT_LIMIT)
  result = []
  for offset, item in enumerate(raw[start:]):
    if isinstance(item, dict) and item.get("role") and item.get("content"):
      content = str(item["content"])
      index = start + offset
      streaming = bool(item.get("streaming")) and index == len(raw) - 1
      if item["role"] == "assistant" and not streaming and not content.startswith(("⏳", "⚠️", "ℹ️")):
        content += _message_actions_html(index, item)
      result.append({"role": item["role"], "content": content})
  return result


def _global_docs(doc_data):
  """Passages répartis sur tout le document (titre, résumé, technologies, flashcards, quiz)."""
  vector_db = doc_data["vector_db"]
  all_docs = [
      doc for doc in vector_db.docstore._dict.values()
      if getattr(doc, "page_content", "").strip()
  ]
  if not all_docs:
    return []

  MAX_GLOBAL_CHUNKS = 12
  if len(all_docs) <= MAX_GLOBAL_CHUNKS:
    return all_docs

  step = (len(all_docs) - 1) / float(MAX_GLOBAL_CHUNKS - 1)
  indices = [round(i * step) for i in range(MAX_GLOBAL_CHUNKS)]
  return [all_docs[i] for i in indices if i < len(all_docs)]


def preparer_contexte_global(doc_data):
  return [doc.page_content.strip() for doc in _global_docs(doc_data)]


# --- 4. PIPELINE CHAT & GEMINI ---
TOKEN_SESSION_LIMIT = 50000
session_usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}


def _estimate_tokens(text_value):
  return max(1, len(str(text_value or "")) // 4)


_EN_WORDS = {"the", "what", "is", "are", "how", "why", "does", "do", "explain", "define", "of", "in", "and", "to", "which",
             "when", "where", "who", "can", "give", "list", "between", "difference", "about", "this", "that", "with", "for"}
_FR_WORDS = {"le", "la", "les", "des", "du", "de", "une", "un", "est", "sont", "quoi", "quel", "quelle", "quels", "comment",
             "pourquoi", "explique", "expliquer", "définir", "définis", "entre", "et", "que", "qui", "dans", "pour", "sur",
             "ce", "cette", "ces", "quand", "où", "peux", "donne", "liste", "différence"}
_SMALLTALK = {"bonjour", "salut", "coucou", "bonsoir", "merci", "merci beaucoup", "ok", "d accord", "hello", "hi", "hey",
              "thanks", "thank you", "good morning"}


def _guess_lang(text):
  words = re.findall(r"[a-zàâçéèêëîïôûùüÿ']+", str(text).lower())
  en = sum(1 for w in words if w in _EN_WORDS)
  fr = sum(1 for w in words if w in _FR_WORDS)
  return "en" if en > fr else "fr"


def _is_smalltalk(text):
  return _normalize_query(text) in _SMALLTALK


def _source_label(src):
  page = src.get("page")
  head = f"page {int(page)}" if page else "extrait"
  return f"{head} — {src['section']}" if src.get("section") else head


def _source_entry(n, doc, similarity=None):
  meta = getattr(doc, "metadata", None) or {}
  excerpt = re.sub(r"\s+", " ", str(getattr(doc, "page_content", ""))).strip()
  if len(excerpt) > 260:
    excerpt = excerpt[:257].rstrip() + "…"
  return {"n": n, "page": meta.get("page"), "section": meta.get("section") or "", "excerpt": excerpt,
          "similarity": round(similarity, 3) if similarity is not None else None}


def _gemini_fail_text(errors):
  """Message clair selon la cause réelle (quota, modèle introuvable, réseau, surcharge) + détail technique."""
  codes = [c for _, c in errors]
  tally = ", ".join(f"{c}×{codes.count(c)}" for c in dict.fromkeys(codes))
  if codes and all(c == 429 for c in codes):
    return f"Quota Gemini atteint : la limite de requêtes de votre clé est épuisée (par minute ou par jour). Attendez une minute ou utilisez une autre clé. ({tally})"
  if codes and all(c == 404 for c in codes):
    return f"Aucun modèle Gemini n'est accessible avec cette clé (404). Vérifiez GEMINI_API_KEY et les noms de modèles. ({tally})"
  if codes and all(c == "net" for c in codes):
    return f"Connexion à Gemini impossible (réseau ou délai dépassé). Réessayez dans un instant. ({tally})"
  return f"Gemini est très sollicité en ce moment. Réessayez dans une dizaine de secondes. ({tally})" if tally else \
         "Gemini est très sollicité en ce moment. Réessayez dans une dizaine de secondes."


def _stream_gemini(user_prompt, system_instruction, throttle=False):
  """Flux de texte Gemini. Tant qu'aucun mot n'est reçu : on passe aussitôt au modèle de secours suivant,
  puis on refait un tour complet (jusqu'à 3) après une courte pause, dans la limite du délai global.
  Un quota épuisé (429) ou un modèle inconnu (404) n'est pas réessayé en boucle."""
  _require_gemini_key()
  errors = []
  deadline = time.monotonic() + GEMINI_BUDGET_S
  for round_no in range(3):
    round_errors = []
    for model_target in _models_ordered():
      if time.monotonic() > deadline:
        break
      started = False
      try:
        if throttle:
          _wait_before_gemini_request()
        stream = client.models.generate_content_stream(
            model=model_target,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=1200,
            ),
        )
        for chunk in stream:
          text = getattr(chunk, "text", "") or ""
          if text:
            if not started:
              _mark_model(model_target)
            started = True
            yield text
        return
      except APIError as e:
        if started:
          raise
        if e.code in (404, 429, 500, 503, 504):
          errors.append((model_target, e.code)); round_errors.append(e.code)
          print(f"⚠️ {model_target} : erreur {e.code}. Modèle suivant.")
          _mark_model(model_target, e.code)
          continue
        raise
      except (PermissionError, GeneratorExit, KeyboardInterrupt):
        raise
      except UnicodeError:
        raise RuntimeError(GEMINI_KEY_ERROR)
      except Exception as e:  # délai dépassé, coupure réseau…
        if started:
          raise
        print(f"⚠️ {model_target} : {type(e).__name__} ({e}). Modèle suivant.")
        errors.append((model_target, "net")); round_errors.append("net")
        _mark_model(model_target, "net")
    if time.monotonic() > deadline or (round_errors and all(c in (429, 404) for c in round_errors)):
      break
    time.sleep(min(1.2 * (round_no + 1), 3.0) + random.uniform(0.0, 0.4))
  if time.monotonic() > deadline and not errors:
    _log("gemini_timeout")
    raise RuntimeError("Gemini met trop de temps à répondre. Réessayez dans un instant.")
  text = _gemini_fail_text(errors)
  _log("gemini_overloaded", detail=text[-120:])
  if any(c == 404 for _, c in errors):
    threading.Thread(target=_discover_models, daemon=True).start()   # les noms de modèles ont peut-être changé
  raise RuntimeError(text)


def _ui_chat_user_message(message_utilisateur, history, doc_selectionne):
  """Étape instantanée : affiche la question sans attendre FAISS/Gemini."""
  message_str = str(message_utilisateur or "").strip()
  history = list(history or [])

  if not message_str:
    return "", history, ""

  if not doc_selectionne or doc_selectionne not in documents_db:
    history.append({
        "role": "user",
        "content": message_str,
    })
    history.append({
        "role": "assistant",
        "content": "⚠️ Veuillez d'abord sélectionner un cours.",
    })
    return "", history, ""

  historique = documents_db[doc_selectionne]["history"]
  historique.append({"role": "user", "content": message_str})
  historique.append({"role": "assistant", "content": "⏳ Génération…"})
  if len(historique) > HISTORY_KEEP:
    del historique[:-HISTORY_KEEP]

  # IMPORTANT : aucune sauvegarde FAISS ici.
  return "", _chat_history_for_ui(doc_selectionne), message_str


def repondre_akori_chat_stream(message_str, doc_selectionne):
  """Étape lourde : retrieval + Gemini streaming, sans réécrire FAISS."""
  if not message_str:
    yield _chat_history_for_ui(doc_selectionne), ""
    return

  if not doc_selectionne or doc_selectionne not in documents_db:
    yield _chat_history_for_ui(doc_selectionne), ""
    return

  doc_data = documents_db[doc_selectionne]
  vector_db = doc_data["vector_db"]
  historique = doc_data["history"]

  # Le placeholder a déjà été créé par _ui_chat_user_message().
  if not historique or historique[-1].get("role") != "assistant":
    historique.append({"role": "assistant", "content": "⏳ Génération…"})

  try:
    t_start = time.monotonic()
    item = historique[-1]
    item["streaming"] = True
    item.pop("sources", None)
    item.pop("timing", None)
    answer_lang = _guess_lang(message_str)

    def finish_local(text):
      item["content"] = text
      item.pop("streaming", None)
      _save_history(doc_selectionne)

    # Requête locale : pas d'appel Gemini pour une question de titre simple.
    if _is_title_query(message_str):
      title = doc_data.get("title", "").strip()
      english = _normalize_query(message_str) in {"title", "the title", "what is the title", "what s the title"}
      if title:
        finish_local(f"The title of the document is: **{title}**" if english else f"Le titre du document est : **{title}**")
      else:
        finish_local("⚠️ The document title is unavailable." if english else "⚠️ Le titre du document est indisponible.")
      yield _chat_history_for_ui(doc_selectionne), ""
      return

    if _is_smalltalk(message_str):
      finish_local("ℹ️ Hello! Ask me a question about this course and I will answer from its passages."
                   if answer_lang == "en" else
                   "ℹ️ Bonjour ! Posez-moi une question sur ce cours : je réponds à partir de ses passages.")
      yield _chat_history_for_ui(doc_selectionne), ""
      return

    lazy = doc_data["vector_db"]
    if isinstance(lazy, _LazyVectorStore) and lazy._rebuild is not None:
      # Ancien index : recalculé une seule fois, avec signes de vie pour que la connexion reste ouverte.
      item["content"] = "⏳ Mise à jour de l'index du cours (une seule fois)…"
      yield _chat_history_for_ui(doc_selectionne), ""
      yield from _with_heartbeat(lazy._load, lambda sec: (_chat_history_for_ui(doc_selectionne), ""))
      item["content"] = "⏳ Génération…"

    best_similarity = None
    if _is_global_query(message_str):
      retrieved = [(doc, None) for doc in _global_docs(doc_data)]
    else:
      scored = vector_db.similarity_search_with_score(message_str, k=RETRIEVAL_K)
      retrieved = [(doc, max(0.0, min(1.0, 1.0 - float(dist) / 2.0))) for doc, dist in scored]
      best_similarity = max((sim for _, sim in retrieved), default=None)
    t_search = time.monotonic() - t_start

    if not retrieved:
      finish_local("⚠️ Aucune information pertinente n'a pu être extraite du document.")
      yield _chat_history_for_ui(doc_selectionne), ""
      return

    # Garde-fou anti-hallucination : si même le meilleur passage est trop éloigné de la question,
    # on le dit au lieu de laisser Gemini répondre avec ses connaissances générales.
    has_previous = len([m for m in historique[:-2] if m.get("content")]) > 0
    too_short = len(message_str.split()) < 3 or (has_previous and len(message_str.split()) < 4)
    if best_similarity is not None and best_similarity < MIN_SIMILARITY and not too_short:
      finish_local("ℹ️ The requested information is not in the provided course document."
                   if answer_lang == "en" else
                   "ℹ️ L'information demandée ne figure pas dans le document de cours fourni.")
      _log("chat_refused", detail=f"{doc_selectionne} · sim {best_similarity:.2f} < {MIN_SIMILARITY:.2f}",
           search_ms=round(t_search * 1000), similarity=round(best_similarity, 3))
      yield _chat_history_for_ui(doc_selectionne), ""
      return

    candidates = []
    for n, (doc, sim) in enumerate(retrieved, start=1):
      entry = _source_entry(n, doc, sim)
      candidates.append(entry)
    contexte_brut = "\n\n".join(
        f"[{c['n']}] ({_source_label(c)})\n{doc.page_content.strip()}"
        for c, (doc, _) in zip(candidates, retrieved)
    )
    system_instruction = (
        "Tu es l'assistant académique AKORI. Réponds de façon claire et précise "
        "en t'appuyant STRICTEMENT sur le contexte fourni. "
        "Si une information n'est pas présente dans le contexte, indique-le "
        "plutôt que de l'inventer. "
        "Les passages du contexte sont numérotés [1], [2]… : quand tu t'appuies sur un passage, "
        "cite son numéro entre crochets à la fin de la phrase concernée, par exemple [1]. "
        "Pour les formules, utilise un format lisible compatible Markdown/LaTeX. "
        "LANGUE : réponds TOUJOURS dans la langue utilisée par l'utilisateur dans sa "
        "dernière question (français, English, etc.), même si le contexte du cours est "
        "rédigé dans une autre langue."
    )
    # Les deux derniers échanges permettent de comprendre les questions de suivi.
    recent = [
        m for m in historique[:-2]
        if m.get("content") and not str(m["content"]).startswith(("⏳", "⚠️"))
    ][-4:]
    recent_txt = "\n".join(
        f"{'Utilisateur' if m['role'] == 'user' else 'AKORI'} : {str(m['content'])[:600]}" for m in recent
    )
    user_prompt = (
        f"CONTEXTE:\n{contexte_brut}\n\n"
        + (f"ÉCHANGES PRÉCÉDENTS:\n{recent_txt}\n\n" if recent_txt else "")
        + f"DEMANDE UTILISATEUR:\n{message_str}"
    )
    input_tokens_est = _estimate_tokens(user_prompt)
    session_usage["input_tokens"] += input_tokens_est

    # Streaming visuel lissé : Gemini envoie des fragments de tailles variables.
    # On regroupe les fragments très courts avant de rafraîchir l'interface afin
    # d'éviter un effet de clignotement tout en gardant une impression naturelle.
    reponse = ""
    buffer = ""
    t_first = None
    last_ui_update = time.monotonic()
    MIN_STREAM_CHARS = 10
    MAX_STREAM_DELAY = 0.05

    for texte_chunk in _stream_gemini(user_prompt, system_instruction):
      if t_first is None:
        t_first = time.monotonic() - t_start
      reponse += texte_chunk
      buffer += texte_chunk
      now = time.monotonic()

      if len(buffer) >= MIN_STREAM_CHARS or (now - last_ui_update) >= MAX_STREAM_DELAY:
        item["content"] = reponse
        buffer = ""
        last_ui_update = now
        # L'historique précédent reste affiché ; seul le dernier message évolue.
        yield _chat_history_for_ui(doc_selectionne), ""

    t_total = time.monotonic() - t_start
    # Sources : les passages cités par la réponse ([1], [2]…), sinon tous les passages utilisés.
    cited = {int(n) for n in re.findall(r"\[(\d{1,2})\]", reponse)}
    shown = [c for c in candidates if c["n"] in cited] or candidates
    item["sources"] = shown
    item["timing"] = {"search": round(t_search * 1000), "first": round((t_first or t_total) * 1000),
                      "total": round(t_total * 1000)}
    item["content"] = reponse or "…"
    item.pop("streaming", None)
    session_usage["output_tokens"] += _estimate_tokens(reponse)
    session_usage["requests"] += 1
    _save_history(doc_selectionne)
    _log("chat", detail=f"{doc_selectionne} · {len(reponse)} car. · {_fmt_ms(t_total * 1000)}",
         search_ms=item["timing"]["search"], ttft_ms=item["timing"]["first"], total_ms=item["timing"]["total"],
         similarity=round(best_similarity, 3) if best_similarity is not None else None)

    # Dernier état garanti après la fin du flux.
    yield _chat_history_for_ui(doc_selectionne), ""

  except Exception as e:
    _log("error", detail=f"chat · {e}")
    last = historique[-1] if historique else None
    if isinstance(last, dict):
      last.pop("streaming", None)
      last["content"] = f"⚠️ {str(e)}"
    _save_history(doc_selectionne)
    yield _chat_history_for_ui(doc_selectionne), ""


# --- 5. NOUVELLE INTERFACE AKORI ---
# Interface orientée "AI Study Assistant" : cours -> révision -> flashcards/quiz -> assistant -> progression.

TOKEN_SESSION_LIMIT = 50000


LANG_CHOICES = ["Français", "English"]


def _lang_code(label):
    return "en" if "English" in str(label or "") else "fr"


def _with_lang(system_instruction, lang):
    """Ajoute la langue de réponse choisie dans les paramètres (résumé, flashcards, quiz)."""
    if _lang_code(lang) == "en":
        return system_instruction + " IMPORTANT: write the whole answer in English."
    return system_instruction


def _escape_html(value):
    import html
    return html.escape(str(value or ""))


def _format_file_size(size_bytes):
    try:
        size = float(size_bytes or 0)
    except Exception:
        size = 0
    units = ["B", "KB", "MB", "GB"]
    unit = units[0]
    for candidate in units:
        unit = candidate
        if size < 1024 or candidate == units[-1]:
            break
        size /= 1024
    if unit == "B":
        return f"{int(size)} B"
    return f"{size:.1f} {unit}"


def _format_added_date(data, filename=None):
    raw = data.get("added_at", "")
    if raw:
        try:
            return datetime.fromisoformat(raw).strftime("%d/%m/%Y")
        except Exception:
            pass
    if filename:
        try:
            return datetime.fromtimestamp(os.path.getmtime(_doc_metadata_path(filename))).strftime("%d/%m/%Y")
        except Exception:
            pass
    return "Date inconnue"


def _course_card_html(name, data, active=False):
    title = data.get("title") or os.path.splitext(name)[0]
    chunks = int(data.get("chunks_count", 0) or 0)
    size = _format_file_size(data.get("file_size", 0))
    added = _format_added_date(data, name)
    active_cls = " active-course" if active else ""
    return f"""
    <div class='folder-card{active_cls}'>
      <div class='folder-card-top'>
        <div class='pdf-icon'>PDF</div>
        <div class='folder-card-name' title='{_escape_html(name)}'>{_escape_html(name)}</div>
        <span class='folder-status'>Indexé</span>
      </div>
      <div class='folder-card-size'>PDF · {size}</div>
      <div class='folder-card-meta'>Indexé · {chunks} fragments</div>
      <div class='folder-card-date'>Ajouté le {added}</div>
      <div class='folder-card-title'>{_escape_html(title)}</div>
    </div>
    """


def documents_html_v12(active_doc=None):
    if not documents_db:
        return "<div class='empty-state'>📁 Aucun document pour le moment.<br><span>Ajoutez votre premier PDF pour commencer.</span></div>"
    return "<div class='folder-grid'>" + "".join(_course_card_html(n, d, n == active_doc) for n, d in documents_db.items()) + "</div>"


def documents_html_v16(active_doc=None):
    if not documents_db:
        return "<div class='empty-state'>📁 Aucun document pour le moment.<br><span>Ajoutez votre premier PDF pour commencer.</span></div>"
    return "<div class='folder-grid'>" + "".join(
        _course_card_html(name, data, name == active_doc)
        for name, data in documents_db.items()
    ) + "</div>"


def dashboard_html(active_doc=None):
    total_docs = len(documents_db)
    total_chunks = sum(int(d.get("chunks_count", 0)) for d in documents_db.values())
    active = documents_db.get(active_doc, {}) if active_doc else {}
    active_title = active.get("title", "Aucun cours sélectionné")
    _who = _display_name(_CUR_USER.get())
    greeting = f"Bonjour {_escape_html(_who)} 👋" if _who else "Bonjour 👋"
    cards = "".join(_course_card_html(n, d, n == active_doc) for n, d in documents_db.items())
    if not cards:
        cards = "<div class='empty-state'>📚 Aucun cours pour le moment.<br><span>Importez votre premier PDF dans 'Mes dossiers' pour commencer.</span></div>"

    return f"""
    <div class='hero'>
      <div>
        <div class='eyebrow'>ASSISTANT KNOWLEDGE ORGANIZED TO REVISE INTELLIGENTLY</div>
        <h1>{greeting}<br><span>Prêt à booster votre révision ?</span></h1>
        <p class='home-slogan'>AKORI transforme vos cours PDF en un espace de révision intelligent : résumé, flashcards roulette, quiz et assistant RAG.</p>
      </div>
      <div class='hero-orb'>✦</div>
    </div>
    <div class='guide-box'>
      <h3>💡 Guide de révision AKORI &amp; mode roulette flashcards</h3>
      <div class='guide-grid'>
        <div class='guide-item'><b>1. Chargez vos dossiers</b><span>Glissez vos PDF dans Mes dossiers pour activer l'indexation FAISS.</span></div>
        <div class='guide-item'><b>2. Lancement roulette</b><span>Générez au moins 7 flashcards. Les questions défilent automatiquement en boucle.</span></div>
        <div class='guide-item'><b>3. Clic &amp; réponse effacée</b><span>Cliquez sur la carte pour stopper/relancer. La réponse s'efface à chaque relance.</span></div>
      </div>
    </div>
    <div class='stat-row'>
      <div class='mini-stat'><b>{total_docs}</b><span>Cours importés</span></div>
      <div class='mini-stat'><b>{total_chunks}</b><span>Fragments indexés</span></div>
      <div class='mini-stat'><b>{len(active.get('history', [])) // 2 if active else 0}</b><span>Échanges du cours</span></div>
      <div class='mini-stat'><b>RAG</b><span>Recherche active</span></div>
    </div>
    <div class='section-head'><div><h2>Réviser un cours en un clic</h2><p>Générez les outils principaux à partir du cours actif.</p></div></div>
    <div class='revision-banner'>
      <div><strong>{_escape_html(active_title)}</strong><br><span>Résumé · Flashcards · Quiz / QCM · Assistant IA</span></div>
      <span class='revision-chip'>Cours actif</span>
    </div>
    <div class='section-head'><div><h2>Mes cours</h2><p>Vos supports de révision indexés localement.</p></div></div>
    <div class='folder-grid'>{cards}</div>
    """

def course_overview_html(active_doc=None):
    if not active_doc or active_doc not in documents_db:
        return "<div class='empty-state'>Sélectionnez un cours pour afficher son espace de révision.</div>"
    data = documents_db[active_doc]
    progress = _course_progress_percent(active_doc)
    return f"""
    <div class='course-overview'>
      <div class='course-overview-title'>{_escape_html(data.get('title') or active_doc)}</div>
      <div class='course-overview-file'>{_escape_html(active_doc)}</div>
      <div class='course-overview-meta'>{int(data.get('chunks_count',0))} fragments indexés · Progression {progress}% · RAG local</div>
      <div class='course-overview-actions'>Résumé · Flashcards · Quiz / QCM · Assistant IA</div>
    </div>
    """

def documents_html_v12_legacy():
    if not documents_db:
        return "<div class='empty-state'>Aucun document chargé.</div>"
    return "<div class='course-grid'>" + "".join(_course_card_html(n, d) for n, d in documents_db.items()) + "</div>"


def stats_html_v12(active_doc=None):
    return progress_detail_html(active_doc)


def _extract_json_from_text(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r'\{.*\}', text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _gemini_structured(prompt, system_instruction, max_tokens=900):
    """Génération JSON résiliente : modèle de secours aussitôt en cas de surcharge (503/500) ou de limite (429),
    jusqu'à 3 tours complets avec pause croissante. L'erreur n'est montrée qu'une fois tout épuisé."""
    _require_gemini_key()
    errors = []
    deadline = time.monotonic() + GEMINI_BUDGET_S
    for round_no in range(3):
        round_errors = []
        for model_target in _models_ordered():
            if time.monotonic() > deadline:
                break
            try:
                _wait_before_gemini_request()
                response = client.models.generate_content(
                    model=model_target,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json",
                    ),
                )
                _mark_model(model_target)
                return (response.text or "").strip()
            except APIError as e:
                if e.code in (404, 429, 500, 503, 504):
                    errors.append((model_target, e.code)); round_errors.append(e.code)
                    _mark_model(model_target, e.code)
                    print(f"⚠️ {model_target} indisponible ({e.code}). Modèle suivant.")
                    continue
                raise
            except (PermissionError, KeyboardInterrupt):
                raise
            except UnicodeError:
                raise RuntimeError(GEMINI_KEY_ERROR)
            except Exception as e:  # délai dépassé, coupure réseau… : modèle suivant
                print(f"⚠️ {model_target} : {type(e).__name__} ({e}). Modèle suivant.")
                errors.append((model_target, "net")); round_errors.append("net")
                _mark_model(model_target, "net")
        if time.monotonic() > deadline or (round_errors and all(c in (429, 404) for c in round_errors)):
            break
        time.sleep(min(1.5 * (round_no + 1), 4.0) + random.uniform(0.1, 0.5))
    if time.monotonic() > deadline and not errors:
        _log("gemini_timeout")
        raise RuntimeError("Gemini met trop de temps à répondre. Réessayez dans un instant.")
    text = _gemini_fail_text(errors)
    _log("gemini_overloaded", detail=text[-120:])
    raise RuntimeError(text)


def generate_flashcards_v12(doc_name, count=6):
    if not doc_name or doc_name not in documents_db:
        return [], "⚠️ Sélectionnez d'abord un cours."
    context = "\n\n".join(preparer_contexte_global(documents_db[doc_name]))
    prompt = f"""CONTEXTE DU COURS:\n{context}\n\nCrée exactement {count} flashcards de révision basées uniquement sur ce contexte.\nRetourne uniquement un JSON valide de la forme: {{\"cards\":[{{\"question\":\"...\",\"answer\":\"...\"}}]}}"""
    try:
        data = _extract_json_from_text(_gemini_structured(prompt, "Tu es AKORI, assistant académique. Utilise uniquement le contexte fourni. N'invente aucune information."))
        cards = data.get("cards", []) if isinstance(data, dict) else []
        cards = [c for c in cards if c.get("question") and c.get("answer")][:count]
        if not cards:
            return [], "⚠️ Impossible de générer les flashcards à partir du contexte."
        return cards, f"✅ {len(cards)} flashcards générées à partir du cours actif."
    except Exception as e:
        message = str(e)
        if "503" in message or "surchargé" in message or "indisponible" in message:
            return [], "⚠️ Gemini est temporairement indisponible. Réessayez dans quelques secondes."
        if "429" in message or "Quota" in message or "limite" in message:
            return [], "⚠️ La limite temporaire de Gemini a été atteinte. Attendez un peu puis réessayez."
        return [], f"⚠️ Génération impossible : {message}"


# ============================================================
# FLASHCARDS — mode roulette
# ============================================================
def normalize_flashcards(cards):
    if not isinstance(cards, list):
        return []
    return [
        {"question": str(c.get("question", "")).strip(), "answer": str(c.get("answer", "")).strip()}
        for c in cards if isinstance(c, dict) and str(c.get("question", "")).strip()
    ]


def flashcard_escape(value):
    return html.escape(str(value or ""), quote=True)


def save_flashcard_result(doc_name, question, result):
    """Enregistre le résultat dans la progression réelle du cours."""
    if not doc_name or doc_name not in documents_db:
        return
    _progress_for(doc_name)["flashcards"][question] = "review" if result == "review" else "known"
    _save_progress(doc_name)


def get_flashcard_results(doc_name, cards):
    if not doc_name or doc_name not in documents_db:
        return {}
    marks = _progress_for(doc_name)["flashcards"]
    return {str(i): marks.get(c["question"]) for i, c in enumerate(cards) if marks.get(c["question"])}


def flashcard_statistics(cards, results):
    total = len(cards)
    known = sum(1 for i in range(total) if results.get(str(i)) == "known")
    review = sum(1 for i in range(total) if results.get(str(i)) == "review")
    return {"total": total, "known": known, "review": review,
            "mastery": round(known / total * 100) if total else 0}


_FC_TEXT = {
    "fr": {"card": "🎯 CARTE", "running": "🎰 Roulette active", "stop_hint": "Cliquez sur la carte pour l'arrêter.",
           "stopped": "⏸ Roulette arrêtée", "show_hint": "Cliquez sur le bouton pour afficher la réponse.",
           "next_hint": "Passez à la carte suivante.", "answer": "💡 Réponse :",
           "click_btn": "Cliquez sur le bouton « Afficher la réponse »."},
    "en": {"card": "🎯 CARD", "running": "🎰 Roulette running", "stop_hint": "Click the card to stop it.",
           "stopped": "⏸ Roulette stopped", "show_hint": "Click the button to show the answer.",
           "next_hint": "Go to the next card.", "answer": "💡 Answer:",
           "click_btn": "Click the “Show the answer” button."},
}


def flashcard_view(cards, index=0, revealed=False, doc_name=None, results=None, lang=None):
    cards = normalize_flashcards(cards)
    t = _FC_TEXT[_lang_code(lang)]
    if not cards:
        return "<div class='empty-study'>🧠<br><b>Aucune flashcard générée.</b><br>Choisissez un cours puis cliquez sur « Générer les flashcards ».</div>"

    try:
        index = int(index)
    except Exception:
        index = 0
    index = max(0, min(index, len(cards) - 1))

    current = cards[index]
    question = html.escape(current["question"], quote=False)
    answer = html.escape(current["answer"], quote=False)
    total = len(cards)
    cards_json = json.dumps(cards, ensure_ascii=True).replace("</", "<\\/")

    if revealed:
        answer_html = f"<div style='margin-top:20px;padding:15px;background:#eef2ff;border-radius:12px;border-left:4px solid #4f46e5;color:#1e293b;'><b>{t['answer']}</b><br><br>{answer}</div>"
    else:
        answer_html = f"<div style='margin-top:20px;text-align:center;color:#94a3b8;'>{t['click_btn']}</div>"

    html_body = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body {{ font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:10px;overflow:hidden; }}
.card {{ cursor:pointer;position:relative;min-height:280px;padding:40px;border-radius:24px;background:#fff;border:1px solid #e2e8f0;box-shadow:0 10px 30px rgba(15,23,42,.07);display:flex;flex-direction:column;justify-content:center;align-items:center;transition:box-shadow .2s; }}
.card:hover {{ box-shadow:0 15px 40px rgba(15,23,42,.12); }}
.badge {{ position:absolute;top:20px;left:20px;color:#6366f1;font-size:12px;font-weight:800; }}
.status {{ position:absolute;top:20px;right:20px;padding:5px 10px;border-radius:20px;background:#fef3c7;color:#b45309;font-size:12px;font-weight:700; }}
.q-text {{ text-align:center;color:#172033;font-size:24px;font-weight:bold;line-height:1.4;margin:20px 0; }}
.hint {{ margin-top:20px;color:#94a3b8;font-size:12px;text-align:center; }}
</style></head><body>
<div class="card" onclick="toggle()">
  <div class="badge">{t['card']} {index + 1} / {total}</div>
  <div id="status" class="status">{t['running']}</div>
  <div id="question" class="q-text">{question}</div>
  <div style="width:100%;">{answer_html}</div>
  <div id="hint" class="hint">{t['stop_hint']}</div>
</div>
<script>
const cards = {cards_json};
const realIndex = {index};
const isRevealed = {'true' if revealed else 'false'};
let timer = null;
let running = !isRevealed;
const qEl = document.getElementById("question");
const statusEl = document.getElementById("status");
const hintEl = document.getElementById("hint");
function start() {{
  if (isRevealed || cards.length <= 1) return;
  running = true;
  statusEl.innerText = {json.dumps(t["running"], ensure_ascii=False)};
  statusEl.style.background = "#fef3c7"; statusEl.style.color = "#b45309";
  hintEl.innerText = {json.dumps(t["stop_hint"], ensure_ascii=False)};
  timer = setInterval(() => {{ qEl.innerText = cards[Math.floor(Math.random() * cards.length)].question; }}, 120);
}}
function stop() {{
  running = false;
  if (timer) clearInterval(timer);
  statusEl.innerText = {json.dumps(t["stopped"], ensure_ascii=False)};
  statusEl.style.background = "#dcfce7"; statusEl.style.color = "#15803d";
  hintEl.innerText = isRevealed ? {json.dumps(t["next_hint"], ensure_ascii=False)} : {json.dumps(t["show_hint"], ensure_ascii=False)};
  qEl.innerText = cards[realIndex].question;
}}
window.toggle = function() {{ if (isRevealed) return; running ? stop() : start(); }};
running ? start() : stop();
</script></body></html>"""
    safe_html = html.escape(html_body, quote=True)
    return f'<iframe srcdoc="{safe_html}" style="width:100%;height:450px;border:none;overflow:hidden;background:transparent;"></iframe>'


MIN_FLASHCARDS = 7
GENERATE_LABEL = "✦ Générer les flashcards"
REGENERATE_LABEL = "🔄 Régénérer (7 minimum)"


def _with_heartbeat(work, tick, interval=3.0):
    """Exécute `work()` en arrière-plan et émet `tick(secondes)` régulièrement.

    Le flux reste ainsi actif pendant une génération longue (le tunnel ne coupe plus la connexion
    inactive) et l'utilisateur voit que quelque chose se passe.
    """
    box = {}

    def runner():
        try:
            box["value"] = work()
        except BaseException as exc:  # remonté dans le flux principal
            box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    started = time.monotonic()
    while thread.is_alive():
        thread.join(interval)
        if thread.is_alive():
            yield tick(int(time.monotonic() - started))
    if "error" in box:
        raise box["error"]
    return box["value"]


def _flash_cards_from_context(context, lang=None):
    """Une série de flashcards (au moins MIN_FLASHCARDS) à partir du contexte. Retourne (cartes, erreur)."""
    cards, last_error = [], None
    for _attempt in range(2):
        try:
            prompt = f"""CONTEXTE DU COURS:\n{context}\n\n
Crée {MIN_FLASHCARDS + 1} flashcards (questions/réponses courtes) pour réviser les concepts clés de ce cours.
Il en faut au moins {MIN_FLASHCARDS}, avec des questions toutes différentes.
Retourne UNIQUEMENT un JSON valide avec la structure exacte suivante :
{{"flashcards": [{{"question": "...", "answer": "..."}}]}}"""
            raw = _gemini_structured(
                prompt,
                _with_lang("Tu es AKORI, assistant académique. Base-toi uniquement sur le contexte fourni et n'invente rien.", lang),
                max_tokens=2500,
            )
            data = _extract_json_from_text(raw)
            found = []
            if isinstance(data, dict):
                found = data.get("flashcards") or data.get("cards") or []
            cards = normalize_flashcards(found)
            if len(cards) >= MIN_FLASHCARDS:
                break
        except Exception as e:
            last_error = e
            break
    return cards, last_error


def _summary_text(context, lang=None):
    """Résumé à partir du contexte du cours (appel Gemini, sans cache). Utilisé par le bouton Résumé et par le Build."""
    prompt = f"CONTEXTE DU COURS:\n{context}\n\nDEMANDE: Fais un résumé synthétique des points clés principaux de ce document."
    response = _gemini_structured(prompt, _with_lang("Tu es AKORI. Résume uniquement les informations présentes dans le contexte fourni. Retourne un JSON {\"summary\":\"...\"}.", lang))
    data = _extract_json_from_text(response)
    return data.get("summary", response) if isinstance(data, dict) else response


def _flash_generate_fail(message):
    return [], 0, False, flashcard_view([], 0, False), message, gr.update(visible=False), gr.update(value=GENERATE_LABEL)


def flashcard_generate_handler(doc_name, lang=None):
    """Génère une NOUVELLE série (7 flashcards minimum) à chaque appel."""
    if not doc_name or doc_name not in documents_db:
        yield _flash_generate_fail("⚠️ Aucun document sélectionné.")
        return

    document = documents_db[doc_name]
    context = "\n\n".join(preparer_contexte_global(document))
    if not context.strip():
        yield _flash_generate_fail("❌ Document vide.")
        return

    cards, last_error = yield from _with_heartbeat(
        lambda: _flash_cards_from_context(context, lang), lambda sec: (gr.update(),) * 4 + (f"⏳ Génération des flashcards… ({sec} s)", gr.update(), gr.update()))

    if last_error is not None and not cards:
        yield _flash_generate_fail(f"❌ Erreur : {last_error}")
        return
    if len(cards) < MIN_FLASHCARDS:
        yield _flash_generate_fail(
            f"⚠️ Seulement {len(cards)} flashcard(s) obtenue(s) : il en faut au moins {MIN_FLASHCARDS}. Relancez la génération.")
        return

    document["flashcards"] = cards
    document["flash_index"] = 0
    _save_history(doc_name)
    _log("flashcards", detail=f"{doc_name} · {len(cards)}")
    view = flashcard_view(cards, 0, False, doc_name, get_flashcard_results(doc_name, cards), lang)
    yield (cards, 0, False, view, f"✅ {len(cards)} flashcards prêtes !",
           gr.update(visible=True), gr.update(value=GENERATE_LABEL))


def flashcard_reveal_handler(cards, index, doc_name, lang=None):
    cards = normalize_flashcards(cards)
    return True, flashcard_view(cards, index, True, doc_name, get_flashcard_results(doc_name, cards), lang)


def flashcard_end_game_view(doc_name, total, known_count, review_count):
    mastery = round(known_count / total * 100) if total else 0
    doc_safe = flashcard_escape(doc_name or "Cours actuel")
    return f"""
    <div class='quiz-result'>
      <div style="font-size:48px;margin-bottom:12px;">🎉</div>
      <div class='quiz-result-kicker'>SESSION TERMINÉE</div>
      <p>{doc_safe}</p>
      <div class='quiz-stats'>
        <div><b>{mastery}%</b><span>Maîtrise</span></div>
        <div><b>{known_count}</b><span>Maîtrisées</span></div>
        <div><b>{review_count}</b><span>À revoir</span></div>
      </div>
      <p style="margin-top:18px;">Cliquez sur « 🔄 Régénérer (7 minimum) » pour obtenir une nouvelle série.</p>
    </div>
    """


def flashcard_mark_handler(cards, index, mark_type, doc_name, lang=None):
    cards = normalize_flashcards(cards)
    if not cards:
        return 0, False, flashcard_view([], 0, False), gr.update(visible=False), gr.update()
    index = int(index)
    if index >= len(cards):
        # Session déjà terminée : rien d'autre à faire que régénérer.
        return index, False, gr.update(), gr.update(visible=False), gr.update(value=REGENERATE_LABEL)
    save_flashcard_result(doc_name, cards[index]["question"], mark_type)
    next_index = index + 1
    if doc_name in documents_db:
        documents_db[doc_name]["flash_index"] = next_index
        _save_history(doc_name)
    results = get_flashcard_results(doc_name, cards)
    if next_index >= len(cards):
        st = flashcard_statistics(cards, results)
        return (next_index, False,
                flashcard_end_game_view(doc_name, st["total"], st["known"], st["review"]),
                gr.update(visible=False), gr.update(value=REGENERATE_LABEL))
    return (next_index, False, flashcard_view(cards, next_index, False, doc_name, results, lang),
            gr.update(visible=True), gr.update())


# ============================================================
# QUIZ — une question à la fois, réponses cliquables
# ============================================================
def _generate_quiz_from_context(context, count=5, lang=None):
    prompt = f"""CONTEXTE DU COURS:\n{context}\n\nCrée exactement {count} questions QCM de révision. Une seule bonne réponse par question.\nRetourne uniquement un JSON valide: {{\"questions\":[{{\"question\":\"...\",\"options\":[\"...\",\"...\",\"...\",\"...\"],\"answer\":0,\"explanation\":\"...\",\"topic\":\"notion abordée (2-3 mots)\"}}]}}\nanswer est l'index 0-3 de la bonne option."""
    try:
        data = _extract_json_from_text(_gemini_structured(prompt, _with_lang("Tu es AKORI, assistant académique. Base-toi uniquement sur le contexte fourni et n'invente rien.", lang)))
        questions = data.get("questions", []) if isinstance(data, dict) else []
        valid = []
        for q in questions:
            if isinstance(q, dict) and q.get("question") and isinstance(q.get("options"), list) and len(q["options"]) == 4 and q.get("answer") in [0, 1, 2, 3]:
                q["topic"] = str(q.get("topic") or "Général").strip() or "Général"
                valid.append(q)
        valid = valid[:count]
        if not valid:
            return [], "⚠️ Impossible de générer le quiz."
        return valid, f"✅ Quiz de {len(valid)} questions généré."
    except Exception as e:
        return [], f"⚠️ {e}"


def generate_quiz_v12(doc_name, count=5, lang=None):
    if not doc_name or doc_name not in documents_db:
        return [], "⚠️ Sélectionnez d'abord un cours."
    return _generate_quiz_from_context("\n\n".join(preparer_contexte_global(documents_db[doc_name])), count, lang)


NEXT_LABEL = "Question suivante →"
FINISH_LABEL = "Voir le résultat 🎯"


def _quiz_score(questions, answers):
    answers = answers or []
    return sum(
        1 for i, q in enumerate(questions)
        if i < len(answers) and answers[i] is not None and int(answers[i]) == int(q["answer"])
    )


def _quiz_radio(questions, index, visible=True):
    """Radio dont les choix sont les vrais textes des réponses."""
    if not questions or index >= len(questions):
        return gr.update(choices=[], value=None, visible=False)
    options = questions[index]["options"]
    return gr.update(
        choices=[(f"{chr(65 + i)}.  {opt}", i) for i, opt in enumerate(options)],
        value=None, visible=visible, interactive=True, label="Choisissez votre réponse",
    )


def quiz_view(questions, index=0, validated=False, answers=None):
    if not questions:
        return "<div class='empty-study'>📝<br><b>Aucun quiz généré.</b><br>Choisissez un cours puis cliquez sur « Générer le quiz ».</div>"
    total = len(questions)
    index = max(0, min(int(index), total - 1))
    q = questions[index]
    answers = answers or []
    selected = answers[index] if index < len(answers) else None
    score = _quiz_score(questions, answers)
    done = index + (1 if validated else 0)
    pct = round(done / total * 100)

    body = ""
    if validated:
        rows = ""
        for i, option in enumerate(q["options"]):
            cls = "option"
            if i == int(q["answer"]):
                cls += " correct"
            elif selected == i:
                cls += " wrong"
            mark = "✓" if i == int(q["answer"]) else ("✗" if selected == i else chr(65 + i))
            rows += f"<div class='{cls}'><span>{mark}</span>{_escape_html(option)}</div>"
        ok = selected is not None and int(selected) == int(q["answer"])
        verdict = "<div class='quiz-verdict ok'>✅ Bonne réponse !</div>" if ok else "<div class='quiz-verdict ko'>❌ Réponse incorrecte</div>"
        body = f"{verdict}{rows}<div class='quiz-explanation'><b>Explication :</b> {_escape_html(q.get('explanation', ''))}</div>"

    return f"""
    <div class='quiz-topline'><span>Question {index + 1} / {total}</span><span>Score : {score}</span></div>
    <div class='quiz-bar'><div style='width:{pct}%;'></div></div>
    <div class='quiz-card'><div class='card-label'>QUIZ / QCM · {_escape_html(q.get('topic', 'Général'))}</div><h2>{_escape_html(q['question'])}</h2>{body}</div>
    """


def _remember_quiz(doc_name, questions, index, answers, validated):
    """Conserve l'état du quiz dans le cours (restauré au retour sur ce PDF)."""
    if doc_name in documents_db:
        documents_db[doc_name]["quiz"] = {
            "questions": questions, "index": int(index), "answers": list(answers or []), "validated": bool(validated),
        }
        _save_history(doc_name)


def quiz_generate_handler(doc_name, lang=None):
    if not doc_name or doc_name not in documents_db:
        qs, status = [], "⚠️ Sélectionnez d'abord un cours."
    else:
        context = "\n\n".join(preparer_contexte_global(documents_db[doc_name]))
        qs, status = yield from _with_heartbeat(
            lambda: _generate_quiz_from_context(context, 5, lang),
            lambda sec: (gr.update(),) * 4 + (f"⏳ Génération du quiz… ({sec} s)", gr.update(), gr.update(), gr.update()))
    if qs:
        _log("quiz", detail=f"{doc_name} · {len(qs)}")
        _remember_quiz(doc_name, qs, 0, [None] * len(qs), False)
    yield (qs, 0, False, quiz_view(qs, 0, False, []), status,
           _quiz_radio(qs, 0), [None] * len(qs), gr.update(value=NEXT_LABEL))


def quiz_restart_handler(questions, doc_name=None):
    questions = questions or []
    _remember_quiz(doc_name, questions, 0, [None] * len(questions), False)
    return (questions, 0, False, quiz_view(questions, 0, False, []), "🔁 Quiz relancé.",
            _quiz_radio(questions, 0), [None] * len(questions), gr.update(value=NEXT_LABEL))


def quiz_validate_handler(questions, index, choice, validated, answers, doc_name=None):
    if not questions or int(index) >= len(questions):
        return validated, gr.update(), "⚠️ Générez d'abord un quiz.", answers, gr.update(), gr.update()
    if validated:
        return validated, gr.update(), "ℹ️ Réponse déjà validée : passez à la suite.", answers, gr.update(), gr.update()
    if choice is None:
        return False, gr.update(), "⚠️ Sélectionnez une réponse avant de valider.", answers, gr.update(), gr.update()
    idx = int(index)
    answers = list(answers or [None] * len(questions))
    answers.extend([None] * (len(questions) - len(answers)))
    answers[idx] = int(choice)
    ok = int(choice) == int(questions[idx]["answer"])
    status = "✅ Bonne réponse." if ok else "❌ Réponse incorrecte. Consultez l'explication."
    last = idx == len(questions) - 1
    _remember_quiz(doc_name, questions, idx, answers, True)
    return (True, quiz_view(questions, idx, True, answers), status, answers,
            gr.update(visible=False), gr.update(value=FINISH_LABEL if last else NEXT_LABEL))


def quiz_result_view(questions, answers):
    total = len(questions)
    score = _quiz_score(questions, answers)
    pct = round(score / total * 100) if total else 0
    wrong = ""
    for i, q in enumerate(questions):
        a = answers[i] if i < len(answers) else None
        if a is None or int(a) != int(q["answer"]):
            wrong += (f"<div class='quiz-miss'><b>{_escape_html(q['question'])}</b>"
                      f"<span>Bonne réponse : {_escape_html(q['options'][int(q['answer'])])}</span></div>")
    review = f"<div class='quiz-miss-list'><div class='global-list-title'>À revoir</div>{wrong}</div>" if wrong else "<p>🏆 Sans faute, bravo !</p>"
    return f"""
    <div class='quiz-result'>
      <div class='quiz-result-kicker'>QUIZ TERMINÉ</div>
      <div class='quiz-score'>{score} / {total}</div>
      <div class='quiz-score-percent'>{pct}% de bonnes réponses</div>
      <p>Votre résultat a été enregistré dans la progression de ce cours.</p>
    </div>{review}
    """


def quiz_next_handler(questions, index, validated, answers, doc_name):
    if not questions or int(index) >= len(questions):
        return index, validated, gr.update(), gr.update(), gr.update(), gr.update()
    index = int(index)
    if not validated:
        return index, False, gr.update(), "⚠️ Validez d'abord cette réponse.", gr.update(), gr.update()
    if index < len(questions) - 1:
        new_index = index + 1
        _remember_quiz(doc_name, questions, new_index, answers, False)
        return (new_index, False, quiz_view(questions, new_index, False, answers), "",
                _quiz_radio(questions, new_index), gr.update(value=NEXT_LABEL))
    answers = list(answers or [])
    score = _quiz_score(questions, answers)
    items = []
    for i, q in enumerate(questions):
        a = answers[i] if i < len(answers) else None
        items.append({"topic": q.get("topic", "Général"), "correct": a is not None and int(a) == int(q["answer"])})
    _progress_for(doc_name)["quiz_attempts"].append(
        {"score": score, "total": len(questions), "items": items, "timestamp": datetime.now().isoformat(timespec="seconds")})
    _save_progress(doc_name)
    # index = len(questions) marque le quiz comme terminé (pas de double enregistrement).
    _remember_quiz(doc_name, questions, len(questions), answers, True)
    return (len(questions), True, quiz_result_view(questions, answers), f"🎯 Quiz terminé : {score}/{len(questions)}.",
            gr.update(choices=[], value=None, visible=False), gr.update(value=NEXT_LABEL))


SUMMARY_PLACEHOLDER = "Sélectionnez un cours puis lancez la génération."


def _revision_views(doc_name, lang=None):
    """État des sections Flashcards / Quiz / Résumé du PDF choisi (vierges pour un PDF sans révision)."""
    data = (documents_db.get(doc_name) if doc_name else None) or {}

    # --- flashcards
    cards = normalize_flashcards(data.get("flashcards", []))
    if cards:
        idx = int(data.get("flash_index", 0))
        results = get_flashcard_results(doc_name, cards)
        if idx >= len(cards):
            st = flashcard_statistics(cards, results)
            flash = (cards, idx, False, flashcard_end_game_view(doc_name, st["total"], st["known"], st["review"]), "",
                     gr.update(visible=False), gr.update(value=REGENERATE_LABEL))
        else:
            flash = (cards, idx, False, flashcard_view(cards, idx, False, doc_name, results, lang), "",
                     gr.update(visible=True), gr.update(value=GENERATE_LABEL))
    else:
        flash = ([], 0, False, flashcard_view([], 0, False), "", gr.update(visible=False), gr.update(value=GENERATE_LABEL))

    # --- quiz
    quiz = data.get("quiz") or {}
    qs = quiz.get("questions") or []
    hidden_radio = gr.update(choices=[], value=None, visible=False)
    if qs:
        n = len(qs)
        idx = int(quiz.get("index", 0))
        answers = list(quiz.get("answers") or [None] * n)
        validated = bool(quiz.get("validated"))
        if idx >= n:
            qz = (qs, idx, True, quiz_result_view(qs, answers), "", hidden_radio, answers, gr.update(value=NEXT_LABEL))
        else:
            finish = validated and idx == n - 1
            qz = (qs, idx, validated, quiz_view(qs, idx, validated, answers), "",
                  hidden_radio if validated else _quiz_radio(qs, idx), answers,
                  gr.update(value=FINISH_LABEL if finish else NEXT_LABEL))
    else:
        qz = ([], 0, False, quiz_view([], 0), "", hidden_radio, [], gr.update(value=NEXT_LABEL))

    # --- résumé
    summary = (data.get("summary_cache") or {}).get(_lang_code(lang)) or SUMMARY_PLACEHOLDER
    return (*flash, *qz, summary)


def history_view(doc_name):
    if not doc_name or doc_name not in documents_db:
        return "<div class='empty-study'>Aucun historique pour le moment.</div>"
    history = documents_db[doc_name].get("history", [])
    blocks = []
    for item in history[-UI_HISTORY_LIMIT:]:
        role = item.get("role", "")
        raw_content = str(item.get("content", ""))
        content = _escape_html(raw_content[:2000] + ("…" if len(raw_content) > 2000 else ""))
        cls = "history-user" if role == "user" else "history-ai"
        label = "Vous" if role == "user" else "AKORI"
        blocks.append(f"<div class='{cls}'><b>{label}</b><div>{content}</div></div>")
    return "".join(blocks) or "<div class='empty-study'>Aucune conversation enregistrée.</div>"


# Logo de l'application (PNG 160x160 intégré : un seul fichier à lancer)
LOGO_DATA_URI = "data:image/png;base64,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"


def _logo_favicon_path():
    """Écrit le logo dans un fichier temporaire pour l'icône de l'onglet du navigateur."""
    try:
        import base64
        path = os.path.join(tempfile.gettempdir(), "akori_logo.png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(LOGO_DATA_URI.split(",", 1)[1]))
        return path
    except Exception:
        return None


def _md_lite(text):
    """Mise en forme minimale (gras, listes, paragraphes) pour afficher une réponse hors du chat."""
    out, in_list = [], False
    for raw in str(text or "").splitlines():
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _escape_html(raw.rstrip()))
        bullet = re.match(r"^\s*[-*•]\s+(.*)$", line)
        if bullet:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{bullet.group(1)}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.strip():
            out.append(f"<p>{line}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


def pins_view(doc_name):
    if not doc_name or doc_name not in documents_db:
        return "<div class='empty-study'>📌<br><b>Aucun cours sélectionné</b></div>"
    pins = documents_db[doc_name].get("pins", [])
    if not pins:
        return ("<div class='empty-study'>📌<br><b>Aucun chapitre épinglé pour ce cours.</b><br>"
                "Dans l'assistant, cliquez sur « Épingler comme chapitre » sous une réponse utile.</div>")
    cards = []
    for number, pin in enumerate(pins, start=1):
        sources = pin.get("sources") or []
        src_html = ("<ol class='ak-src-list pin-sources'>" + "".join(_source_html(x) for x in sources) + "</ol>") if sources else ""
        cards.append(
            f"<div class='pin-card'><div class='pin-head'><span class='pin-num'>Chapitre {number}</span>"
            f"<h3>{_escape_html(pin.get('title') or 'Chapitre')}</h3></div>"
            f"<div class='pin-body'>{_md_lite(pin.get('content', ''))}</div>{src_html}"
            "<div class='ak-msg-actions'>"
            "<button type='button' class='ak-act ak-a-speak'>🔊 <span>Lire à voix haute</span></button>"
            f"<button type='button' class='ak-act ak-a-unpin ak-id-{_escape_html(pin.get('id', ''))}'>✕ <span>Retirer</span></button>"
            "</div></div>"
        )
    return "<div class='pin-list'>" + "".join(cards) + "</div>"


def pin_message(payload, doc_name):
    """Épingle une réponse de l'assistant comme chapitre du cours actif."""
    if doc_name and doc_name in documents_db:
        doc = documents_db[doc_name]
        history = doc.get("history", [])
        try:
            index = int(str(payload).split(":")[0])
            item = history[index]
        except (ValueError, IndexError):
            return pins_view(doc_name)
        if item.get("role") == "assistant" and item.get("content") and not str(item["content"]).startswith(("⏳", "⚠️", "ℹ️")):
            pins = doc.setdefault("pins", [])
            if not any(p.get("content") == item["content"] for p in pins):
                question = ""
                if index > 0 and history[index - 1].get("role") == "user":
                    question = str(history[index - 1].get("content", "")).strip()
                pins.append({
                    "id": secrets.token_hex(4),
                    "title": (question[:90] + ("…" if len(question) > 90 else "")) or "Chapitre",
                    "question": question,
                    "content": item["content"],
                    "sources": list(item.get("sources") or []),
                    "pinned_at": datetime.now().isoformat(timespec="seconds"),
                })
                _save_history(doc_name)
                _log("pin", detail=doc_name)
    return pins_view(doc_name)


def unpin_chapter(payload, doc_name):
    if doc_name and doc_name in documents_db:
        doc = documents_db[doc_name]
        before = len(doc.get("pins", []))
        doc["pins"] = [p for p in doc.get("pins", []) if p.get("id") != str(payload).strip()]
        if len(doc["pins"]) != before:
            _save_history(doc_name)
    return pins_view(doc_name)


custom_css = r"""
:root {
  --ak-blue:#4f6df5;
  --ak-blue-dark:#3f59d8;
  --ak-purple:#8b6cf6;
  --ak-bg:#f6f8fc;
  --ak-card:#ffffff;
  --ak-text:#182033;
  --ak-muted:#69758b;
  --ak-soft:#eef2ff;
  --ak-line:#e4e9f2;
}

html, body { background:var(--ak-bg)!important; }
body, .gradio-container {
  background:var(--ak-bg)!important;
  color:var(--ak-text)!important;
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif!important;
}
.gradio-container { max-width:1540px!important; margin:auto!important; padding:18px 20px!important; }
.gradio-container * { box-sizing:border-box; }

/* ===== Sidebar ===== */
/* Sidebar fixe et menu Gradio secondaire masqué */
#component-\d+-\$headless\$dropdown, #main-tabs > .overflow, #main-tabs .overflow, #main-tabs button[aria-label="More"], #main-tabs .tab-nav .overflow {display:none!important;}
#sidebar {
  background:#fff!important;
  border:1px solid var(--ak-line)!important;
  border-radius:22px!important;
  padding:20px 16px!important;
  position:sticky!important;top:14px!important;height:calc(100vh - 28px)!important;overflow-y:auto!important;
  min-height:calc(100vh - 36px)!important;
  box-shadow:0 8px 30px rgba(42,55,90,.06)!important;
}
#sidebar .brand { display:flex;gap:11px;align-items:center;margin:2px 4px 28px; }
#sidebar .brand-mark { width:42px;height:42px;border-radius:13px;background:linear-gradient(135deg,#4768f5,#8e6cf4);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:21px;box-shadow:0 7px 16px rgba(79,109,245,.2); }
#sidebar .brand-name { color:#182033!important;font-size:20px;font-weight:850;line-height:1; }
#sidebar .brand-sub { color:#7b8497!important;font-size:9px;line-height:1.25;margin-top:5px; }
#sidebar .nav-title { color:#9aa4b7!important;font-size:10px;font-weight:800;letter-spacing:.1em;margin:4px 5px 2px; }

/* Gradio 6: elem_classes is attached to the component itself, not an inner button. */
#sidebar .navbtn {
  width:100%!important;
  min-height:43px!important;
  margin:4px 0!important;
  border:1px solid transparent!important;
  border-radius:11px!important;
  background:transparent!important;
  color:#566278!important;
  box-shadow:none!important;
  font-size:13px!important;
  font-weight:650!important;
  text-align:left!important;
  justify-content:flex-start!important;
}
#sidebar .navbtn:hover,
#sidebar .navbtn:focus {
  background:#f1f4ff!important;
  color:var(--ak-blue-dark)!important;
  border-color:#e3e8ff!important;
}
#sidebar .navbtn:active {
  transform:translateX(1px);
}
#sidebar .navbtn:focus-visible { outline:2px solid #cbd4ff!important; }
#sidebar .gr-button { color:inherit!important; }

/* Import button */
#sidebar .primary {
  background:linear-gradient(135deg,#4f6df5,#6d78ef)!important;
  color:#fff!important;
  border:0!important;
  border-radius:11px!important;
  box-shadow:0 8px 16px rgba(79,109,245,.18)!important;
}

/* ===== Main area ===== */
#main-column { min-width:0!important; }
#topbar {
  background:#fff!important;
  border:1px solid var(--ak-line)!important;
  border-radius:16px!important;
  padding:9px 12px!important;
  margin-bottom:14px!important;
  box-shadow:0 5px 18px rgba(42,55,90,.035)!important;
}
#topbar h3, #topbar .prose { color:#27324a!important; }
#topbar label { color:#66728a!important; }
#topbar .wrap { border-color:var(--ak-line)!important; }

/* ===== Navigation : sidebar uniquement ===== */
/* Gradio peut déplacer le bouton de débordement (« … ») dans plusieurs
   conteneurs selon la largeur de la fenêtre. On masque toute la barre d'onglets
   et ses contrôles, car la navigation AKORI est portée par le sidebar. */
#main-tabs > .tab-nav,
#main-tabs .tab-nav,
#main-tabs [role="tablist"],
#main-tabs .overflow,
#main-tabs .tab-nav .overflow,
#main-tabs button[aria-label*="more" i],
#main-tabs button[aria-label*="plus" i] {
  display:none!important;
  visibility:hidden!important;
  height:0!important;
  min-height:0!important;
  overflow:hidden!important;
  margin:0!important;
  padding:0!important;
}

#main-tabs {
  border:0!important;
  background:transparent!important;
}
#main-tabs > .tabitem {
  border:0!important;
}

/* Sidebar persistante pendant le défilement de la page. */
#sidebar {
  position:sticky!important;
  top:18px!important;
  height:calc(100vh - 36px)!important;
  max-height:calc(100vh - 36px)!important;
  overflow-y:auto!important;
  overflow-x:hidden!important;
  scrollbar-width:thin!important;
}

/* Transition légère lors d'un changement de page. */
#main-tabs .tabitem {
  animation:akoriPageIn .15s ease-out;
  transform-origin:top center;
}
@keyframes akoriPageIn {
  from { opacity:0; transform:translateY(3px); }
  to { opacity:1; transform:translateY(0); }
}
#main-tabs.akori-switching {
  animation:akoriShellIn .12s ease-out;
}
@keyframes akoriShellIn {
  from { opacity:.72; transform:translateY(3px); }
  to { opacity:1; transform:translateY(0); }
}

/* General Gradio text contrast */
.gr-markdown, .prose, .gradio-container label, .gradio-container .wrap, .gradio-container .block {
  color:var(--ak-text)!important;
}
.gradio-container h1,.gradio-container h2,.gradio-container h3,.gradio-container h4,
.gradio-container strong { color:var(--ak-text)!important; }
.gradio-container p { color:#68748a!important; }

/* ===== Dashboard ===== */
.hero {
  background:linear-gradient(120deg,#eef3ff 0%,#faf7ff 100%)!important;
  border:1px solid #dfe6fb!important;
  border-radius:22px!important;
  padding:30px 32px!important;
  display:flex!important;
  justify-content:space-between!important;
  align-items:center!important;
  min-height:190px!important;
  box-shadow:0 8px 25px rgba(61,82,150,.045)!important;
}
.eyebrow { color:#6678bf!important;font-size:10px;font-weight:850;letter-spacing:.08em; }
.hero h1 { color:#25304a!important;font-size:34px!important;margin:10px 0 8px!important;line-height:1.1; }
.hero h1 span { color:#5069df!important; }
.hero p { color:#69758b!important;max-width:650px; }
.hero-orb { width:96px;height:96px;border-radius:28px;background:linear-gradient(135deg,#657ef7,#a687f7);color:#fff;display:flex;align-items:center;justify-content:center;font-size:48px;box-shadow:0 15px 35px rgba(83,104,230,.22); }

.stat-row { display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0 24px; }
.mini-stat { background:#fff!important;border:1px solid var(--ak-line)!important;border-radius:15px!important;padding:16px!important;box-shadow:0 4px 14px rgba(42,55,90,.025); }
.mini-stat b { color:#1e2940!important;display:block;font-size:23px; }
.mini-stat span { color:#778298!important;font-size:12px; }
.section-head { margin:22px 0 10px; }
.section-head h2 { color:#202a40!important;margin:0;font-size:20px; }
.section-head p { color:#778298!important;margin:4px 0;font-size:13px; }
.revision-banner { background:#fff!important;border:1px solid var(--ak-line)!important;border-radius:17px!important;padding:18px 20px!important;display:flex;justify-content:space-between;align-items:center;box-shadow:0 5px 18px rgba(40,55,90,.04); }
.revision-banner strong { color:#27324a!important;font-size:17px; }
.revision-banner span { color:#778298!important;font-size:12px; }
.revision-chip { background:#eef2ff!important;color:#5067df!important;padding:7px 10px;border-radius:20px; }

.course-grid { display:grid;grid-template-columns:repeat(3,1fr);gap:12px; }
.course-card { background:#fff!important;border:1px solid var(--ak-line)!important;border-radius:16px!important;padding:15px!important;display:flex;gap:12px;align-items:flex-start;min-height:105px; }
.course-card.active-course { border-color:#9cadff!important;box-shadow:0 8px 22px rgba(78,101,221,.1); }
.course-icon { width:38px;height:38px;border-radius:11px;background:#edf2ff;color:#5067e5;display:flex;align-items:center;justify-content:center;font-size:19px;flex:none; }
.course-title { color:#25304a!important;font-weight:750;font-size:14px; }
.course-file,.course-meta { color:#7a8599!important;font-size:11px;margin-top:4px; }
.course-badge { margin-left:auto;font-size:10px;background:#f4f6fa;padding:5px 8px;border-radius:10px;color:#758096; }

/* ===== Mes dossiers : cartes PDF ===== */
.folder-grid { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px; }
.folder-card { background:#fff!important;border:1px solid #dfe5ef!important;border-radius:15px!important;padding:14px!important;min-height:132px;box-shadow:0 4px 14px rgba(42,55,90,.035);transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
.folder-card:hover { transform:translateY(-2px);box-shadow:0 10px 24px rgba(42,55,90,.08);border-color:#cbd5f8!important; }
.folder-card.active-course { border-color:#8ea0f5!important;box-shadow:0 0 0 2px #eef1ff,0 10px 24px rgba(78,101,221,.08); }
.folder-card-top { display:flex;align-items:center;gap:9px;min-width:0; }
.pdf-icon { width:36px;height:40px;border-radius:8px;background:#fff1f1;color:#e24d4d;border:1px solid #ffd8d8;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:850;flex:none; }
.folder-card-name { color:#26324b!important;font-size:13px;font-weight:750;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1; }
.folder-status { background:#e8f8ef;color:#168451;border-radius:9px;padding:4px 7px;font-size:9px;font-weight:750;flex:none; }
.folder-card-size,.folder-card-meta,.folder-card-date { color:#778298!important;font-size:10px;margin-top:7px; }
.folder-card-meta { color:#3d956b!important;font-weight:650; }
.folder-card-date { margin-top:5px; }
.folder-card-title { color:#a0a8b8!important;font-size:10px;margin-top:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
#add-document-tile { min-height:132px!important;height:100%!important; }
#add-document-tile .wrap { min-height:132px!important;height:100%!important;border:1.5px dashed #b9c5e8!important;border-radius:15px!important;background:#fbfcff!important; }
#add-document-tile .wrap:hover { border-color:#7288eb!important;background:#f5f7ff!important; }
#add-document-tile .upload-button, #add-document-tile button { color:#5268d8!important;font-weight:750!important; }
#upload-status { margin-top:8px!important;color:#66728a!important;font-size:12px!important; }

/* Mes dossiers : la zone d'import est une vraie carte d'ajout, claire et cohérente. */
#add-document-tile {
  background:#ffffff!important;
  border:1.5px dashed #c9d2e3!important;
  border-radius:18px!important;
  min-height:142px!important;
  overflow:hidden!important;
  box-shadow:0 8px 22px rgba(36,50,75,.06)!important;
}
#add-document-tile .wrap,
#add-document-tile .file-preview,
#add-document-tile .upload-container {
  background:#ffffff!important;
  color:#26324b!important;
}
#add-document-tile label,
#add-document-tile span,
#add-document-tile p { color:#26324b!important; }
#add-document-tile button {
  background:#eef1ff!important;
  color:#4f63d8!important;
  border:1px solid #d7dcff!important;
}
#add-document-tile .file-preview-file-name { color:#26324b!important; }
#add-document-tile .file-preview-remove { color:#66728a!important; }

@media(max-width:1100px){.folder-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media(max-width:800px){.folder-grid{grid-template-columns:1fr;}}

.empty-state,.empty-study { text-align:center;padding:45px 20px;color:#758096!important;background:#fff!important;border:1px dashed #d9dfeb!important;border-radius:18px; }
.empty-state span {font-size:12px}.note{font-size:11px;color:#8993a7!important;}

/* ===== Study cards ===== */
.study-progress { color:#758096!important;font-size:12px;margin-bottom:10px; }
.flashcard,.quiz-card { background:#fff!important;border:1px solid var(--ak-line)!important;border-radius:22px!important;padding:34px!important;min-height:300px;box-shadow:0 10px 28px rgba(42,55,90,.05); }
.card-label { color:#6879dc!important;font-size:10px;font-weight:850;letter-spacing:.08em; }
.flashcard h2,.quiz-card h2 { color:#202a40!important;font-size:25px;line-height:1.25;max-width:800px;margin:30px 0; }
.card-answer { padding:20px;border-radius:14px;line-height:1.6; }
.answer-hidden { background:#f6f8fc;color:#8993a7!important; }
.answer-visible { background:#eff4ff;color:#25345d!important; }
.option { color:#334057!important;display:flex;gap:12px;align-items:center;padding:14px;border:1px solid #e4e9f2;border-radius:12px;margin:9px 0;cursor:pointer; }
.option span { width:27px;height:27px;border-radius:50%;background:#f1f3f8;color:#526078!important;display:flex;align-items:center;justify-content:center;font-weight:750; }
.option.selected { border-color:#6c82ed;background:#f1f4ff; }
.option.correct { border-color:#48b98a;background:#eefaf5; }
.option.wrong { border-color:#ed7f8b;background:#fff1f3; }
.quiz-explanation { margin-top:18px;padding:14px;background:#f8f9fc;border-radius:12px;color:#5f687b!important;font-size:13px; }


.global-progress-hero{display:flex;align-items:center;gap:24px;background:linear-gradient(135deg,#fff,#f7f8ff);border:1px solid var(--ak-line);border-radius:22px;padding:26px;margin-bottom:14px}.global-progress-ring{width:108px;height:108px;border-radius:50%;background:conic-gradient(#586ff0 var(--progress,0%),#e9edf5 0);display:flex;align-items:center;justify-content:center;position:relative;flex:none}.global-progress-ring:before{content:"";position:absolute;width:82px;height:82px;background:#fff;border-radius:50%}.global-progress-ring span{position:relative;font-size:22px;font-weight:850;color:#26324b!important}.global-progress-copy h2{margin:3px 0 6px;color:#26324b!important}.global-progress-copy p{margin:0;color:#758096!important;font-size:13px;max-width:650px;line-height:1.5}.global-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}.global-metrics>div{background:#fff;border:1px solid var(--ak-line);border-radius:16px;padding:16px}.global-metrics b{display:block;font-size:23px;color:#26324b}.global-metrics span{font-size:12px;color:#778298}.global-course-list{background:#fff;border:1px solid var(--ak-line);border-radius:18px;padding:18px}.global-list-title{font-weight:800;color:#26324b;margin-bottom:12px}.global-course-row{display:grid;grid-template-columns:minmax(220px,1fr) minmax(180px,2fr) 55px;gap:14px;align-items:center;padding:13px 8px;border-bottom:1px solid #edf0f5}.global-course-row:last-child{border-bottom:0}.global-course-row.active-progress-course{background:#f7f8ff;border-radius:12px}.global-course-main{display:flex;align-items:center;gap:10px;min-width:0}.global-course-main b{display:block;color:#344057;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.global-course-main small{display:block;color:#929caf;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.global-course-dot{width:9px;height:9px;border-radius:50%;background:#5b72ed;flex:none}.global-course-bar{height:9px;background:#edf0f6;border-radius:999px;overflow:hidden}.global-course-bar div{height:100%;background:linear-gradient(90deg,#566ff0,#7b7df1);border-radius:999px}.global-course-row>strong{text-align:right;color:#5364cf;font-size:13px}.global-detail-hint{margin-top:12px;padding:13px 15px;background:#f7f9fc;border-radius:12px;color:#778298;font-size:12px;line-height:1.5}.home-slogan{font-size:15px!important;line-height:1.75!important;color:#344057!important;font-weight:600!important;max-width:850px;margin-top:18px!important}.home-intro{color:#758096!important;max-width:780px}
@media(max-width:900px){.global-metrics{grid-template-columns:1fr 1fr}.global-course-row{grid-template-columns:1fr}.global-course-row>strong{text-align:left}}

/* ===== Real progress ===== */
.progress-hero{display:flex;align-items:center;gap:20px;background:#fff;border:1px solid var(--ak-line);border-radius:20px;padding:22px;margin-bottom:14px;}
.progress-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}.progress-metrics>div{background:#fff;border:1px solid var(--ak-line);border-radius:16px;padding:16px}.progress-metrics b{display:block;font-size:24px;color:#26324b}.progress-metrics span{font-size:12px;color:#778298}.progress-columns{display:grid;grid-template-columns:1fr 1fr;gap:14px}.progress-box{background:#fff;border:1px solid var(--ak-line);border-radius:18px;padding:18px}.progress-box h3{margin:0 0 12px;color:#26324b}.strength-row,.weakness-row{display:flex;justify-content:space-between;padding:12px 0;border-bottom:1px solid #edf0f5;color:#4a566d}.strength-row b{color:#22966b}.weakness-row b{color:#d95b67}.progress-note,.progress-empty{padding:14px;color:#7b879b;background:#f7f9fc;border-radius:12px}.course-overview{background:#fff;border:1px solid var(--ak-line);border-radius:18px;padding:22px}.course-overview-title{font-size:24px;font-weight:800;color:#26324b}.course-overview-file{margin-top:5px;color:#7c879a}.course-overview-meta{margin-top:14px;color:#69768c;font-size:13px}.course-overview-actions{margin-top:12px;color:#5364cf;font-weight:650}
.quiz-result{background:#fff;border:1px solid #dfe5ef;border-radius:22px;padding:38px;text-align:center;box-shadow:0 10px 28px rgba(42,55,90,.05)}.quiz-result-kicker{font-size:11px;letter-spacing:.1em;font-weight:800;color:#6879dc}.quiz-score{font-size:58px;font-weight:850;color:#26324b;margin-top:10px}.quiz-score-percent{font-size:20px;color:#5364cf;font-weight:750}.quiz-result p{color:#778298}
@media(max-width:900px){.progress-metrics{grid-template-columns:1fr 1fr}.progress-columns{grid-template-columns:1fr}}

/* ===== Progress / history ===== */
.progress-panel { display:flex;align-items:center;gap:16px;background:#fff;border:1px solid var(--ak-line);border-radius:18px;padding:20px; }
.progress-ring { width:74px;height:74px;border-radius:50%;background:conic-gradient(#5c75ef var(--progress,0%),#e9edf5 0);display:flex;align-items:center;justify-content:center;position:relative; }
.progress-ring:before { content:"";position:absolute;width:56px;height:56px;background:#fff;border-radius:50%; }
.progress-ring span { position:relative;font-weight:850;color:#26324b!important; }
.progress-title { color:#27324a!important;font-weight:750; }.progress-sub{font-size:12px;color:#778298!important;margin-top:4px;}
.progress-list{margin-top:12px;background:#fff;border:1px solid var(--ak-line);border-radius:16px;padding:4px 16px;}.progress-list div{display:flex;justify-content:space-between;padding:13px 0;border-bottom:1px solid #edf0f5;font-size:13px;color:#465269!important}.progress-list div:last-child{border:0}
.history-user,.history-ai { padding:13px 16px;border-radius:14px;margin:8px 0;line-height:1.5;font-size:13px; }
.history-user { background:#eef2ff;margin-left:15%;color:#34405a!important; }.history-ai { background:#fff;border:1px solid var(--ak-line);margin-right:10%;color:#34405a!important; }
.history-user b,.history-ai b { font-size:11px;color:#586ee1!important;display:block;margin-bottom:5px; }

/* ===== Controls ===== */
.gradio-container .gr-button {
  border-radius:11px!important;
  border:1px solid #dfe5ef!important;
  color:#40506a!important;
  background:#fff!important;
  box-shadow:none!important;
  font-weight:650!important;
}
.gradio-container .gr-button:hover { border-color:#bdc9f5!important;background:#f5f7ff!important;color:#415bd3!important; }
.gradio-container .gr-button.primary,
.gradio-container button.primary,
.gradio-container .primary { background:linear-gradient(135deg,#4f6df5,#6d78ef)!important;color:#fff!important;border:0!important;box-shadow:0 7px 16px rgba(79,109,245,.18)!important; }
.gradio-container .gr-button.primary:hover,
.gradio-container button.primary:hover { background:#4662e8!important;color:#fff!important; }
.gradio-container input,.gradio-container textarea,.gradio-container select { color:#26324b!important;background:#fff!important; }
.gradio-container input::placeholder,.gradio-container textarea::placeholder { color:#9aa4b7!important; }

/* Dropdown / file picker */
.gradio-container .wrap, .gradio-container .input-container { border-color:#dfe5ef!important;background:#fff!important; }
.gradio-container .wrap:focus-within { border-color:#9dadf3!important;box-shadow:0 0 0 2px #eef1ff!important; }

/* Tab content and titles */
.main-tabs .tabitem { background:transparent!important;color:var(--ak-text)!important;padding-top:2px!important; }
.main-tabs .tabitem > .prose:first-child { margin-top:0!important; }


/* ===== Gradio 6 hard visual reset for AKORI ===== */

/* ===== V12.2: force a readable light academic UI ===== */
:root {
  --ak-bg: #f7f9fd !important;
  --ak-text: #1f2a44 !important;
  --ak-blue-dark: #3f5ed8 !important;
  --ak-line: #dfe5ef !important;
}

/* Sidebar and navigation */
#sidebar,
#sidebar .gr-column,
#sidebar .block,
#sidebar .form {
  background: #ffffff !important;
  color: #24314a !important;
}

#sidebar .navbtn,
#sidebar .navbtn.gr-button,
#sidebar button,
#sidebar .gr-button {
  background: #ffffff !important;
  background-image: none !important;
  color: #263550 !important;
  -webkit-text-fill-color: #263550 !important;
  border: 1px solid transparent !important;
  box-shadow: none !important;
  opacity: 1 !important;
}

#sidebar .navbtn *,
#sidebar button *,
#sidebar .gr-button * {
  color: #263550 !important;
  -webkit-text-fill-color: #263550 !important;
  opacity: 1 !important;
}

#sidebar .navbtn:hover,
#sidebar .navbtn.gr-button:hover,
#sidebar button:hover {
  background: #eef2ff !important;
  color: #3f5ed8 !important;
  -webkit-text-fill-color: #3f5ed8 !important;
  border-color: #d7defa !important;
}

#sidebar .navbtn:active,
#sidebar .navbtn:focus,
#sidebar .navbtn:focus-visible {
  background: #e9efff !important;
  color: #3f5ed8 !important;
  -webkit-text-fill-color: #3f5ed8 !important;
}

/* Import button remains primary */
#sidebar .primary,
#sidebar .primary *,
#sidebar button.primary {
  background: linear-gradient(135deg, #4f6df5, #707cf0) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  border: 0 !important;
}

/* Other buttons stay light */
.gradio-container .gr-button:not(.primary),
.gradio-container button:not(.primary) {
  background: #ffffff !important;
  color: #34415b !important;
  -webkit-text-fill-color: #34415b !important;
  border-color: #dfe5ef !important;
}

/* Unicode/special-character friendly font */
html, body, .gradio-container,
#sidebar, #main-column,
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  font-family: "Noto Sans", "DejaVu Sans", "Segoe UI", Arial, sans-serif !important;
}

/* Math/KaTeX readability */
.katex,
.katex-display,
.katex * {
  color: #1f2a44 !important;
}
.katex-display {
  overflow-x: auto !important;
  overflow-y: hidden !important;
  padding: 8px 0 !important;
}

/* Chatbot: light and high contrast */
#assistant-chatbot,
#assistant-chatbot > *,
#assistant-chatbot .wrap,
#assistant-chatbot [data-testid="chatbot"] {
  background: #ffffff !important;
  color: #263550 !important;
}

#assistant-chatbot .message,
#assistant-chatbot .message-wrap,
#assistant-chatbot .bubble-wrap,
#assistant-chatbot .message-bubble,
#assistant-chatbot .prose,
#assistant-chatbot .prose * {
  color: #263550 !important;
  -webkit-text-fill-color: #263550 !important;
}

#assistant-chatbot [data-testid="user"],
#assistant-chatbot .message.user {
  background: #eef2ff !important;
  border: 1px solid #dce4ff !important;
  color: #263550 !important;
}

#assistant-chatbot [data-testid="bot"],
#assistant-chatbot .message.bot {
  background: #ffffff !important;
  border: 1px solid #e2e7f0 !important;
  color: #263550 !important;
}

#assistant-chatbot .prose code {
  background: #f2f5f9 !important;
  color: #34415b !important;
}

#main-tabs .tab-nav, #main-tabs > .tab-nav, #main-tabs [role="tablist"] {
  display:none!important;
}
#main-tabs .tabitem {
  border:0!important;
  background:transparent!important;
  color:#24314a!important;
}

/* Active-course selector */
#topbar .gr-dropdown,
#topbar .gr-dropdown .wrap,
#topbar .gr-dropdown input,
#topbar .gr-dropdown button {
  background:#fff!important;
  color:#26324b!important;
  border-color:#dfe5ef!important;
}
#topbar .gr-dropdown label {
  color:#66728a!important;
}

/* Assistant chat: force the light AKORI surface instead of Gradio's dark
   default bubble theme. */
#assistant-chatbot {
  background:#ffffff!important;
  border:1px solid #dfe5ef!important;
  border-radius:18px!important;
  box-shadow:0 8px 25px rgba(42,55,90,.05)!important;
  overflow:hidden!important;
}
#assistant-chatbot > div,
#assistant-chatbot .wrap,
#assistant-chatbot .chatbot,
#assistant-chatbot [data-testid="chatbot"] {
  background:#ffffff!important;
}
#assistant-chatbot .message,
#assistant-chatbot .message-wrap,
#assistant-chatbot .bubble-wrap,
#assistant-chatbot .message-bubble,
#assistant-chatbot .prose {
  color:#293650!important;
}
#assistant-chatbot .message.user,
#assistant-chatbot [data-testid="user"] {
  background:#eef2ff!important;
  color:#293650!important;
  border:1px solid #dfe5ff!important;
}
#assistant-chatbot .message.bot,
#assistant-chatbot [data-testid="bot"] {
  background:#ffffff!important;
  color:#293650!important;
  border:1px solid #e4e9f2!important;
}
#assistant-chatbot p,
#assistant-chatbot li,
#assistant-chatbot span,
#assistant-chatbot div {
  color:#33415c!important;
}
#assistant-chatbot code {
  background:#f4f6fa!important;
  color:#33415c!important;
}
#assistant-chatbot pre {
  background:#f6f8fc!important;
  color:#33415c!important;
  border:1px solid #e5e9f1!important;
}

/* Composer */
#assistant-chatbot + * {}


#assistant-input textarea,
#assistant-input input {
  background:#fff!important;
  color:#26324b!important;
  border-color:#dfe5ef!important;
}
#assistant-input textarea::placeholder,
#assistant-input input::placeholder {
  color:#9aa4b7!important;
}
#assistant-send {
  min-height:48px!important;
}

@media(max-width:1100px){
  .stat-row,.course-grid,.folder-grid{grid-template-columns:1fr 1fr}
  .hero-orb{display:none}
}
@media(max-width:800px){
  .gradio-container{padding:10px!important}
  #sidebar{min-height:auto!important}
  .stat-row,.course-grid,.folder-grid{grid-template-columns:1fr}
  .hero h1{font-size:27px}
}

/* ===== Accueil : guide ===== */
.guide-box { background:#fff;border:1px solid #dfe5ef;border-radius:18px;padding:22px;margin:20px 0;box-shadow:0 4px 16px rgba(42,55,90,.03); }
.guide-box h3 { margin:0 0 12px;font-size:16px;font-weight:800; }
.guide-grid { display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;font-size:13px; }
.guide-item { background:#f8fafc;padding:14px;border-radius:12px;border:1px solid #e2e8f0; }
.guide-item b { color:#4f6df5!important;font-size:14px;display:block;margin-bottom:4px; }
.guide-item span { color:#64748b; }

/* ===== Quiz pratique ===== */
.quiz-topline { display:flex;justify-content:space-between;color:#758096;font-size:12px;font-weight:650;margin-bottom:8px; }
.quiz-bar { height:7px;background:#e9edf5;border-radius:999px;overflow:hidden;margin-bottom:14px; }
.quiz-bar div { height:100%;background:linear-gradient(90deg,#566ff0,#7b7df1);border-radius:999px;transition:width .3s ease; }
.quiz-card { min-height:0!important;padding:26px!important; }
.quiz-card h2 { margin:14px 0 18px!important;font-size:21px!important; }
.quiz-verdict { font-weight:800;margin-bottom:10px;font-size:15px; }
.quiz-verdict.ok { color:#168451; } .quiz-verdict.ko { color:#d95b67; }
.quiz-stats { display:flex;justify-content:center;gap:14px;flex-wrap:wrap;margin-top:14px; }
.quiz-stats>div { background:#eef2ff;border-radius:14px;padding:14px 22px;min-width:110px; }
.quiz-stats b { display:block;font-size:26px;color:#4f46e5; } .quiz-stats span { font-size:12px;color:#778298; }
.quiz-miss-list { background:#fff;border:1px solid var(--ak-line);border-radius:16px;padding:16px;margin-top:12px; }
.quiz-miss { padding:10px 0;border-bottom:1px solid #edf0f5; } .quiz-miss:last-child { border:0; }
.quiz-miss b { display:block;color:#26324b;font-size:13px; } .quiz-miss span { color:#168451;font-size:12px; }

/* Radio du quiz : grosses cartes cliquables */
.quiz-radio-group { background:transparent!important;border:0!important;padding:0!important; }
.quiz-radio-group .wrap { display:flex!important;flex-direction:column!important;gap:10px!important;background:transparent!important;border:0!important; }
.quiz-radio-group label.svelte-1mhtq7j, .quiz-radio-group .wrap > label {
  background:#fff!important;border:1.5px solid #e2e8f0!important;border-radius:14px!important;
  padding:14px 18px!important;cursor:pointer!important;transition:all .15s ease!important;
  font-size:14px!important;display:flex!important;align-items:center!important;
}
.quiz-radio-group .wrap > label:hover { border-color:#4f46e5!important;background:#eef2ff!important;transform:translateY(-1px); }
.quiz-radio-group .wrap > label.selected, .quiz-radio-group .wrap > label:has(input:checked) { border-color:#4f46e5!important;background:#eef2ff!important;box-shadow:0 0 0 2px #dfe5ff; }
.quiz-radio-group input[type="radio"] { accent-color:#4f46e5!important;margin-right:12px!important; }

/* ===== Bouton de thème ===== */
#theme-toggle { max-width:150px!important;min-width:130px!important; }

/* ===== THÈME SOMBRE ===== */
html:has(body.akori-dark), html:root:root:root:root body.akori-dark { background:#0e1422!important; }
html:root:root:root:root body.akori-dark {
  --ak-bg:#0e1422; --ak-card:#161e30; --ak-text:#e6ebf7; --ak-muted:#9aa6bf;
  --ak-soft:#1c2540; --ak-line:#27324a; --ak-blue-dark:#8ea2ff;
}
html:root:root:root:root body.akori-dark, html:root:root:root:root body.akori-dark .gradio-container { background:#0e1422!important;color:#e6ebf7!important; }
html:root:root:root:root body.akori-dark .gradio-container p, html:root:root:root:root body.akori-dark .gradio-container span:not(.katex *) { color:#a3aec6; }
html:root:root:root:root body.akori-dark .gradio-container h1, html:root:root:root:root body.akori-dark .gradio-container h2,
html:root:root:root:root body.akori-dark .gradio-container h3, html:root:root:root:root body.akori-dark .gradio-container h4,
html:root:root:root:root body.akori-dark .gradio-container strong, html:root:root:root:root body.akori-dark .gradio-container label,
html:root:root:root:root body.akori-dark .gr-markdown, html:root:root:root:root body.akori-dark .prose { color:#e6ebf7!important; }

/* surfaces */
html:root:root:root:root body.akori-dark #sidebar, html:root:root:root:root body.akori-dark #sidebar .block, html:root:root:root:root body.akori-dark #sidebar .form, html:root:root:root:root body.akori-dark #sidebar .gr-column,
html:root:root:root:root body.akori-dark #topbar, html:root:root:root:root body.akori-dark .mini-stat, html:root:root:root:root body.akori-dark .revision-banner, html:root:root:root:root body.akori-dark .course-card,
html:root:root:root:root body.akori-dark .folder-card, html:root:root:root:root body.akori-dark .flashcard, html:root:root:root:root body.akori-dark .quiz-card, html:root:root:root:root body.akori-dark .global-metrics>div,
html:root:root:root:root body.akori-dark .global-course-list, html:root:root:root:root body.akori-dark .progress-hero, html:root:root:root:root body.akori-dark .progress-metrics>div,
html:root:root:root:root body.akori-dark .progress-box, html:root:root:root:root body.akori-dark .course-overview, html:root:root:root:root body.akori-dark .quiz-result,
html:root:root:root:root body.akori-dark .quiz-miss-list, html:root:root:root:root body.akori-dark .history-ai, html:root:root:root:root body.akori-dark .guide-box,
html:root:root:root:root body.akori-dark .empty-state, html:root:root:root:root body.akori-dark .empty-study, html:root:root:root:root body.akori-dark .progress-list,
html:root:root:root:root body.akori-dark .progress-panel {
  background:#161e30!important;border-color:#27324a!important;
}
html:root:root:root:root body.akori-dark .hero { background:linear-gradient(120deg,#16203a 0%,#1b1838 100%)!important;border-color:#2a3558!important; }
html:root:root:root:root body.akori-dark .global-progress-hero { background:linear-gradient(135deg,#161e30,#1a2036)!important;border-color:#27324a!important; }
html:root:root:root:root body.akori-dark .guide-item, html:root:root:root:root body.akori-dark .global-detail-hint, html:root:root:root:root body.akori-dark .progress-note,
html:root:root:root:root body.akori-dark .progress-empty, html:root:root:root:root body.akori-dark .quiz-explanation, html:root:root:root:root body.akori-dark .answer-hidden,
html:root:root:root:root body.akori-dark .global-course-row.active-progress-course { background:#1b2438!important;border-color:#27324a!important; }
html:root:root:root:root body.akori-dark .history-user, html:root:root:root:root body.akori-dark .revision-chip, html:root:root:root:root body.akori-dark .quiz-stats>div { background:#1c2540!important; }
html:root:root:root:root body.akori-dark .global-progress-ring:before, html:root:root:root:root body.akori-dark .progress-ring:before { background:#161e30!important; }
html:root:root:root:root body.akori-dark .global-progress-ring, html:root:root:root:root body.akori-dark .progress-ring { background:conic-gradient(#7b8ff7 var(--progress,0%),#27324a 0)!important; }
html:root:root:root:root body.akori-dark .global-course-bar, html:root:root:root:root body.akori-dark .quiz-bar { background:#27324a!important; }
html:root:root:root:root body.akori-dark .global-course-row, html:root:root:root:root body.akori-dark .strength-row, html:root:root:root:root body.akori-dark .weakness-row,
html:root:root:root:root body.akori-dark .quiz-miss, html:root:root:root:root body.akori-dark .progress-list div { border-color:#27324a!important; }

/* textes principaux */
html:root:root:root:root body.akori-dark .mini-stat b, html:root:root:root:root body.akori-dark .global-metrics b, html:root:root:root:root body.akori-dark .progress-metrics b,
html:root:root:root:root body.akori-dark .section-head h2, html:root:root:root:root body.akori-dark .hero h1, html:root:root:root:root body.akori-dark .revision-banner strong,
html:root:root:root:root body.akori-dark .folder-card-name, html:root:root:root:root body.akori-dark .course-title, html:root:root:root:root body.akori-dark .course-overview-title,
html:root:root:root:root body.akori-dark .progress-title, html:root:root:root:root body.akori-dark .global-course-main b, html:root:root:root:root body.akori-dark .global-list-title,
html:root:root:root:root body.akori-dark .progress-box h3, html:root:root:root:root body.akori-dark .quiz-score, html:root:root:root:root body.akori-dark .global-progress-copy h2,
html:root:root:root:root body.akori-dark .global-progress-ring span, html:root:root:root:root body.akori-dark .progress-ring span, html:root:root:root:root body.akori-dark .strength-row,
html:root:root:root:root body.akori-dark .weakness-row, html:root:root:root:root body.akori-dark .history-user, html:root:root:root:root body.akori-dark .history-ai,
html:root:root:root:root body.akori-dark .option, html:root:root:root:root body.akori-dark .flashcard h2, html:root:root:root:root body.akori-dark .quiz-card h2,
html:root:root:root:root body.akori-dark .quiz-miss b, html:root:root:root:root body.akori-dark #sidebar .brand-name { color:#e6ebf7!important; }
html:root:root:root:root body.akori-dark .hero h1 span { color:#8ea2ff!important; }
html:root:root:root:root body.akori-dark .hero p, html:root:root:root:root body.akori-dark .home-slogan, html:root:root:root:root body.akori-dark .mini-stat span, html:root:root:root:root body.akori-dark .global-metrics span,
html:root:root:root:root body.akori-dark .progress-metrics span, html:root:root:root:root body.akori-dark .section-head p, html:root:root:root:root body.akori-dark .revision-banner span,
html:root:root:root:root body.akori-dark .folder-card-size, html:root:root:root:root body.akori-dark .folder-card-date, html:root:root:root:root body.akori-dark .course-overview-file,
html:root:root:root:root body.akori-dark .course-overview-meta, html:root:root:root:root body.akori-dark .global-course-main small, html:root:root:root:root body.akori-dark .progress-sub,
html:root:root:root:root body.akori-dark .quiz-topline, html:root:root:root:root body.akori-dark .quiz-result p, html:root:root:root:root body.akori-dark .guide-item span,
html:root:root:root:root body.akori-dark #sidebar .brand-sub, html:root:root:root:root body.akori-dark #sidebar .sidebar-note { color:#9aa6bf!important; }
html:root:root:root:root body.akori-dark .eyebrow, html:root:root:root:root body.akori-dark .card-label, html:root:root:root:root body.akori-dark .quiz-result-kicker { color:#8ea2ff!important; }

/* boutons de navigation */
html:root:root:root:root body.akori-dark #sidebar .navbtn, html:root:root:root:root body.akori-dark #sidebar .navbtn.gr-button, html:root:root:root:root body.akori-dark #sidebar button,
html:root:root:root:root body.akori-dark #sidebar .gr-button { background:#161e30!important;color:#c9d2e6!important;-webkit-text-fill-color:#c9d2e6!important; }
html:root:root:root:root body.akori-dark #sidebar .navbtn *, html:root:root:root:root body.akori-dark #sidebar button * { color:#c9d2e6!important;-webkit-text-fill-color:#c9d2e6!important; }
html:root:root:root:root body.akori-dark #sidebar .navbtn:hover, html:root:root:root:root body.akori-dark #sidebar button:hover,
html:root:root:root:root body.akori-dark #sidebar .navbtn:focus { background:#1f2a45!important;color:#a8b8ff!important;-webkit-text-fill-color:#a8b8ff!important;border-color:#2f3d66!important; }
html:root:root:root:root body.akori-dark .gradio-container .gr-button:not(.primary), html:root:root:root:root body.akori-dark .gradio-container button:not(.primary):not(.navbtn) {
  background:#1b2438!important;color:#d5dcf0!important;-webkit-text-fill-color:#d5dcf0!important;border-color:#2c3957!important; }
html:root:root:root:root body.akori-dark #sidebar .primary, html:root:root:root:root body.akori-dark #sidebar button.primary { background:linear-gradient(135deg,#4f6df5,#707cf0)!important;color:#fff!important;-webkit-text-fill-color:#fff!important; }

/* champs */
html:root:root:root:root body.akori-dark .gradio-container input, html:root:root:root:root body.akori-dark .gradio-container textarea, html:root:root:root:root body.akori-dark .gradio-container select,
html:root:root:root:root body.akori-dark #assistant-input textarea, html:root:root:root:root body.akori-dark #assistant-input input,
html:root:root:root:root body.akori-dark #topbar .gr-dropdown, html:root:root:root:root body.akori-dark #topbar .gr-dropdown .wrap, html:root:root:root:root body.akori-dark #topbar .gr-dropdown input {
  background:#1b2438!important;color:#e6ebf7!important;border-color:#2c3957!important; }
html:root:root:root:root body.akori-dark .gradio-container .wrap, html:root:root:root:root body.akori-dark .gradio-container .input-container { background:#1b2438!important;border-color:#2c3957!important; }
html:root:root:root:root body.akori-dark #topbar label, html:root:root:root:root body.akori-dark #topbar .prose, html:root:root:root:root body.akori-dark #topbar h3 { color:#c9d2e6!important; }

/* chat */
html:root:root:root:root body.akori-dark #assistant-chatbot, html:root:root:root:root body.akori-dark #assistant-chatbot > div, html:root:root:root:root body.akori-dark #assistant-chatbot .wrap,
html:root:root:root:root body.akori-dark #assistant-chatbot [data-testid="chatbot"] { background:#161e30!important;border-color:#27324a!important; }
html:root:root:root:root body.akori-dark #assistant-chatbot [data-testid="bot"], html:root:root:root:root body.akori-dark #assistant-chatbot .message.bot { background:#1b2438!important;border-color:#2c3957!important; }
html:root:root:root:root body.akori-dark #assistant-chatbot [data-testid="user"], html:root:root:root:root body.akori-dark #assistant-chatbot .message.user { background:#232f55!important;border-color:#33427a!important; }
html:root:root:root:root body.akori-dark #assistant-chatbot .message, html:root:root:root:root body.akori-dark #assistant-chatbot .prose, html:root:root:root:root body.akori-dark #assistant-chatbot .prose *,
html:root:root:root:root body.akori-dark #assistant-chatbot p, html:root:root:root:root body.akori-dark #assistant-chatbot li, html:root:root:root:root body.akori-dark #assistant-chatbot span,
html:root:root:root:root body.akori-dark #assistant-chatbot div { color:#dbe2f3!important;-webkit-text-fill-color:#dbe2f3!important; }
html:root:root:root:root body.akori-dark #assistant-chatbot code, html:root:root:root:root body.akori-dark #assistant-chatbot pre { background:#0f1626!important;color:#dbe2f3!important;border-color:#27324a!important; }
html:root:root:root:root body.akori-dark .katex, html:root:root:root:root body.akori-dark .katex * { color:#e6ebf7!important; }

/* quiz */
html:root:root:root:root body.akori-dark .option { border-color:#2c3957!important;background:#1b2438!important; }
html:root:root:root:root body.akori-dark .option span { background:#27324a!important;color:#c9d2e6!important; }
html:root:root:root:root body.akori-dark .option.correct { border-color:#2fa77a!important;background:#12352b!important; }
html:root:root:root:root body.akori-dark .option.wrong { border-color:#d9687a!important;background:#3a1b25!important; }
html:root:root:root:root body.akori-dark .quiz-radio-group .wrap > label { background:#1b2438!important;border-color:#2c3957!important;color:#e6ebf7!important; }
html:root:root:root:root body.akori-dark .quiz-radio-group .wrap > label span { color:#e6ebf7!important; }
html:root:root:root:root body.akori-dark .quiz-radio-group .wrap > label:hover,
html:root:root:root:root body.akori-dark .quiz-radio-group .wrap > label:has(input:checked) { background:#232f55!important;border-color:#7b8ff7!important;box-shadow:0 0 0 2px #2a3866; }
html:root:root:root:root body.akori-dark .quiz-stats b { color:#a8b8ff!important; }

/* dossiers / import */
html:root:root:root:root body.akori-dark .folder-card:hover { border-color:#3b4b7a!important; }
html:root:root:root:root body.akori-dark .folder-card.active-course { border-color:#7b8ff7!important;box-shadow:0 0 0 2px #1f2a4d; }
html:root:root:root:root body.akori-dark #add-document-tile .wrap { background:transparent!important; }

html:root:root:root:root body.akori-dark #sidebar:hover { box-shadow:0 18px 50px rgba(0,0,0,.55)!important; }
html:root:root:root:root body.akori-dark #sidebar .navbtn::before { color:inherit; }

/* cartes flashcards (iframe) : inversion douce */
html:root:root:root:root body.akori-dark iframe { filter:invert(.92) hue-rotate(180deg); }

/* ===== Tuile « Ajouter un document » ===== */
#add-document-tile, #add-document-tile.gr-button, button#add-document-tile {
  background:#111827!important;background-image:none!important;
  border:2px solid #1f2937!important;border-radius:18px!important;
  min-height:142px!important;height:100%!important;width:100%!important;
  display:flex!important;flex-direction:column!important;align-items:center!important;justify-content:center!important;
  white-space:pre-line!important;text-align:center!important;line-height:1.5!important;
  color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;
  font-size:15px!important;font-weight:600!important;
  box-shadow:0 10px 25px rgba(17,24,39,.2)!important;
  transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease!important;cursor:pointer;
}
#add-document-tile::first-line { font-size:34px;font-weight:300;line-height:1.3; }
#add-document-tile:hover { border-color:#4f6df5!important;transform:translateY(-2px);box-shadow:0 14px 30px rgba(79,109,245,.25)!important; }
#add-document-tile * { color:#fff!important;-webkit-text-fill-color:#fff!important; }


/* ===== Menu latéral : fixe, rail d'icônes, ouverture fluide au survol (sans superposition) ===== */
#nav-home::before { content:"⌂"; }
#nav-courses::before { content:"▣"; }
#nav-review::before { content:"◈"; }
#nav-flash::before { content:"▤"; }
#nav-quiz::before { content:"☷"; }
#nav-summary::before { content:"≡"; }
#nav-assistant::before { content:"✦"; }
#nav-progress::before { content:"↗"; }
#nav-reviewq::before { content:"!"; }
#nav-history::before { content:"◷"; }
#nav-settings::before { content:"⚙"; }
#nav-admin::before { content:"🛡"; }
#nav-logout::before { content:"⏻"; }

/* NB : Gradio recopie chaque règle avec un préfixe ".gradio-container … .contain" qui lui donne plus
   de poids. Les états (replié / survol) passent donc par des variables CSS posées sur <body>. */
:root { --sb-open:264px; --sb-rail:72px; --sb-left:max(16px, calc((100vw - 1280px)/2 + 32px)); --sb-ease:cubic-bezier(.4,0,.2,1); --sb-dur:.32s; }
body { --sb-w:var(--sb-open); --main-ml:calc(var(--sb-open) + 16px); }
body.sidebar-collapsed { --sb-w:var(--sb-rail); --main-ml:calc(var(--sb-rail) + 16px); }
/* survol du rail : le menu s'ouvre et la page glisse en même temps (même durée, même courbe) */
body.sidebar-collapsed:has(#sidebar:hover) { --sb-w:var(--sb-open); --main-ml:calc(var(--sb-open) + 16px); }

/* mise en page stable : toutes les pages ont la même largeur, barre de défilement toujours présente */
html { overflow-y:scroll!important; }
.gradio-container, .gradio-container .main, .gradio-container .main > .wrap, .gradio-container .contain,
.gradio-container .contain > .column, .column:has(> .row > #main-column), .row:has(> #main-column) { width:100%!important; }
.row:has(> #main-column) { flex-wrap:nowrap!important; }

#sidebar {
  position:fixed!important;top:14px!important;left:var(--sb-left)!important;
  width:var(--sb-w)!important;min-width:var(--sb-w)!important;max-width:var(--sb-w)!important;flex:none!important;
  height:calc(100vh - 28px)!important;max-height:calc(100vh - 28px)!important;min-height:0!important;
  display:flex!important;flex-direction:column!important;flex-wrap:nowrap!important;gap:2px!important;--layout-gap:2px;
  padding-left:12px!important;padding-right:12px!important;
  overflow-x:hidden!important;overflow-y:auto!important;scrollbar-width:none;
  box-shadow:0 8px 30px rgba(42,55,90,.06)!important;z-index:60;
  transition:width var(--sb-dur) var(--sb-ease), min-width var(--sb-dur) var(--sb-ease), max-width var(--sb-dur) var(--sb-ease);
  will-change:width;
}
#sidebar::-webkit-scrollbar { display:none; }
#main-column {
  flex:1 1 0!important;min-width:0!important;width:auto!important;
  margin-left:var(--main-ml)!important;
  transition:margin-left var(--sb-dur) var(--sb-ease);
  overflow-wrap:break-word;word-break:normal;
}
#main-tabs .tabitem { min-height:calc(100vh - 150px)!important;width:100%!important; }

/* barre du haut : le titre et le sélecteur de cours se replient proprement, sans casser les mots */
#topbar { flex-wrap:wrap!important;gap:8px 16px!important;align-items:center!important; }
#topbar > * { flex:1 1 240px!important;min-width:min(240px,100%)!important; }
#topbar h3 { white-space:normal!important;word-break:normal!important;overflow-wrap:normal!important;margin:0!important; }

/* Contenu du menu : positions constantes, le texte est simplement « découvert » par la largeur qui s'anime */
#sidebar .block { padding:0!important;margin:0!important;border:0!important;min-height:0!important;flex:none!important; }
#sidebar .navbtn {
  display:flex!important;align-items:center!important;justify-content:flex-start!important;gap:12px!important;
  width:100%!important;min-height:clamp(24px, calc((100vh - 345px) / 13), 40px)!important;margin:0!important;padding:0 12px!important;
  font-size:13px!important;white-space:nowrap!important;overflow:hidden!important;text-overflow:clip!important;flex:none!important;
  transition:background-color .18s ease, color .18s ease, border-color .18s ease, transform .18s ease!important;
}
#sidebar::-webkit-scrollbar { width:0; }
#sidebar .navbtn::before { display:inline-block;width:24px;flex:none;text-align:center;font-size:17px;line-height:1; }
#sidebar .navbtn:hover { transform:translateX(2px); }
#sidebar #nav-settings { margin-top:auto!important;border-top:1px solid var(--ak-line)!important; }

#sidebar #sidebar-toggle {
  width:48px!important;min-width:48px!important;max-width:48px!important;height:40px!important;flex:none!important;
  padding:0!important;font-size:20px!important;align-self:flex-start!important;margin:0 0 6px 0!important;
  display:flex!important;align-items:center!important;justify-content:center!important;
}
#sidebar .html-container, #sidebar .prose { padding:0!important;max-width:none!important; }
#sidebar .nav-title { transition:opacity .2s ease; }
body.sidebar-collapsed:not(:has(#sidebar:hover)) #sidebar .nav-title { opacity:0; }
#sidebar .brand { display:flex!important;align-items:center;gap:11px;width:100%;margin:2px 0 14px!important;padding-left:3px;overflow:hidden;white-space:nowrap; }
#sidebar .brand-mark { flex:none!important;width:42px!important;min-width:42px!important;height:42px!important; }
#sidebar .brand > div:last-child { flex:none;width:170px; }
#sidebar .brand-name { white-space:nowrap; }
#sidebar .brand-sub { white-space:normal!important;overflow-wrap:normal;word-break:normal;font-size:10px!important;line-height:1.35!important; }
#sidebar .nav-title { white-space:nowrap;overflow:hidden;padding-left:3px; }

/* écrans étroits : rail fixe, ouverture en surimpression (pas la place de pousser la page) */
@media (max-width:900px) {
  :root { --sb-left:10px; }
  body, body.sidebar-collapsed, body.sidebar-collapsed:has(#sidebar:hover) { --sb-w:var(--sb-rail); --main-ml:calc(var(--sb-rail) + 12px); }
  body:not(.sidebar-collapsed) { --sb-w:var(--sb-open); }
  body:not(.sidebar-collapsed) #sidebar, body.sidebar-collapsed:has(#sidebar:hover) #sidebar { box-shadow:0 14px 40px rgba(30,41,80,.2)!important; }
  body.sidebar-collapsed:has(#sidebar:hover) { --sb-w:var(--sb-open); }
  .gradio-container { padding:6px!important; }
  .gradio-container .main { padding-left:6px!important;padding-right:6px!important; }
}
@media (max-width:600px) {
  :root { --sb-rail:60px; --sb-open:240px; }
  .hero { padding:20px!important; }
}

/* ===== Choix du thème ===== */
#theme-choice .wrap { display:flex!important;gap:10px!important; }
#theme-choice label { flex:1;border:1.5px solid #dfe5ef!important;border-radius:14px!important;padding:14px 18px!important;cursor:pointer;font-weight:650; }
#theme-choice label:has(input:checked) { border-color:#4f46e5!important;background:#eef2ff!important; }
html:root:root:root:root body.akori-dark #theme-choice label { border-color:#2c3957!important;background:#1b2438!important;color:#e6ebf7!important; }
html:root:root:root:root body.akori-dark #theme-choice label:has(input:checked) { border-color:#7b8ff7!important;background:#232f55!important; }

/* ===== Accueil : mêmes cartes que « Mes dossiers » ===== */
.folder-grid { display:grid!important;grid-template-columns:repeat(auto-fill,minmax(200px,1fr))!important;gap:14px!important;align-items:stretch; }
.folder-grid > .folder-card { min-width:0;height:100%; }
.folder-grid > .empty-state { grid-column:1 / -1; }

/* ===== Paramètres : cartes + bouton de thème animé + sélecteur de langue ===== */
.set-card { align-items:center!important;gap:18px!important;background:#fff!important;border:1px solid var(--ak-line)!important;border-radius:18px!important;padding:18px 22px!important;margin:12px 0!important;box-shadow:0 4px 16px rgba(42,55,90,.04);flex-wrap:nowrap!important; }
.set-card > div:first-child { flex:1 1 auto!important;min-width:0!important; }
.set-title { font-weight:800;font-size:15px;color:#1f2a44; }
.set-desc { font-size:12.5px;color:#778298;margin-top:3px;line-height:1.45; }
html:root:root:root:root body.akori-dark .set-card { background:#161e30!important;border-color:#27324a!important; }
html:root:root:root:root body.akori-dark .set-title { color:#e6ebf7!important; }
html:root:root:root:root body.akori-dark .set-desc { color:#9aa6bf!important; }

#theme-toggle-btn {
  position:relative!important;width:84px!important;min-width:84px!important;max-width:84px!important;height:42px!important;flex:none!important;
  padding:0!important;border:0!important;border-radius:999px!important;overflow:hidden!important;cursor:pointer;
  font-size:0!important;color:transparent!important;-webkit-text-fill-color:transparent!important;
  background:linear-gradient(135deg,#7cc0ff 0%,#d4ecff 100%)!important;
  box-shadow:inset 0 2px 6px rgba(20,40,90,.18), 0 4px 14px rgba(79,109,245,.18)!important;
  transition:background .45s ease, box-shadow .45s ease, transform .15s ease!important;
}
#theme-toggle-btn:active { transform:scale(.96); }
#theme-toggle-btn::before {           /* soleil → lune */
  content:"";position:absolute;top:5px;left:5px;width:32px;height:32px;border-radius:50%;
  background:#ffc93c;box-shadow:0 0 0 5px rgba(255,201,60,.30), 0 3px 8px rgba(0,0,0,.22);
  transition:transform .5s cubic-bezier(.68,-.35,.27,1.35), background .4s ease, box-shadow .4s ease;
}
#theme-toggle-btn::after {            /* nuage → étoiles */
  content:"";position:absolute;right:12px;top:17px;width:20px;height:9px;border-radius:9px;background:#fff;opacity:.95;
  box-shadow:-7px 5px 0 -1px #fff, 5px -5px 0 -2px #fff;
  transition:opacity .4s ease, transform .5s ease;
}
html:root:root:root:root body.akori-dark #theme-toggle-btn {
  background:linear-gradient(135deg,#090d26 0%,#1c2766 100%)!important;
  box-shadow:inset 0 2px 8px rgba(0,0,0,.5), 0 4px 14px rgba(0,0,0,.35)!important;
}
html:root:root:root:root body.akori-dark #theme-toggle-btn::before {
  transform:translateX(42px);background:transparent;
  box-shadow:inset -10px -4px 0 0 #f2efe2, 0 0 14px rgba(242,239,226,.45);
}
html:root:root:root:root body.akori-dark #theme-toggle-btn::after {
  right:auto;left:0;top:0;width:100%;height:100%;border-radius:0;opacity:1;transform:none;
  background:
    radial-gradient(circle at 18% 30%,#fff 1.3px,transparent 2.2px),
    radial-gradient(circle at 34% 68%,#fff 1px,transparent 2px),
    radial-gradient(circle at 52% 24%,#fff 1.5px,transparent 2.3px),
    radial-gradient(circle at 26% 82%,#cfd6ff 0.9px,transparent 1.8px);
  box-shadow:none;
}

#lang-choice { flex:none!important;min-width:0!important;width:auto!important; }
#lang-choice .wrap { display:flex!important;gap:4px!important;background:#eef2ff!important;border:0!important;border-radius:14px!important;padding:4px!important; }
#lang-choice label { border:0!important;border-radius:11px!important;padding:8px 14px!important;cursor:pointer;font-weight:650;font-size:13px;background:transparent!important;color:#52607d!important;transition:background .2s ease,color .2s ease,box-shadow .2s ease; }
#lang-choice label:hover { background:rgba(255,255,255,.7)!important; }
#lang-choice label:has(input:checked) { background:#fff!important;color:#3f5ed8!important;box-shadow:0 2px 8px rgba(42,55,90,.14); }
#lang-choice input[type="radio"] { display:none!important; }
html:root:root:root:root body.akori-dark #lang-choice .wrap { background:#1b2438!important; }
html:root:root:root:root body.akori-dark #lang-choice label { color:#b4bfd8!important; }
html:root:root:root:root body.akori-dark #lang-choice label:has(input:checked) { background:#2a3866!important;color:#fff!important;box-shadow:none; }
@media (max-width:700px) { .set-card { flex-wrap:wrap!important; } }

/* ===== Mes dossiers : liste de PDF + tuile d'import (largeur fixe, jamais écrasée) ===== */
#documents-row { display:flex!important;flex-wrap:nowrap!important;gap:14px!important;align-items:flex-start!important; }
#documents-row > :first-child { flex:1 1 0!important;min-width:0!important; }
#documents-row #add-document-tile {
  flex:0 0 210px!important;width:210px!important;min-width:210px!important;max-width:210px!important;
  height:150px!important;min-height:150px!important;align-self:flex-start!important;
}
@media (max-width:760px) {
  #documents-row { flex-wrap:wrap!important; }
  #documents-row > :first-child { flex:1 1 100%!important; }
  #documents-row #add-document-tile { flex:1 1 100%!important;width:100%!important;max-width:none!important;height:110px!important;min-height:110px!important;order:-1; }
}

/* ===== Éléments techniques masqués ===== */
.akori-hidden { display:none!important; }
.brand-logo { width:42px;height:42px;border-radius:12px;object-fit:cover;flex:none;display:block; }
.auth-brand .brand-logo { width:52px;height:52px;border-radius:15px; }
#sidebar .brand-logo { width:42px!important;height:42px!important;min-width:42px!important; }
/* Mot de passe : icône œil dans le champ */
.pw-wrap { position:relative!important; }
.pw-wrap input { padding-right:54px!important; }
html:root:root:root:root body .pw-eye { position:absolute!important;right:26px!important;bottom:21px!important;top:auto!important;width:30px!important;height:30px!important;min-width:0!important;
  border:0!important;border-radius:9px!important;background:transparent!important;color:#8a8a97!important;box-shadow:none!important;
  cursor:pointer;display:flex!important;align-items:center;justify-content:center;z-index:5;padding:0!important;transition:color .15s,background .15s; }
html:root:root:root:root body .pw-eye:hover, html:root:root:root:root body .pw-eye.on { color:#fff!important;background:rgba(255,255,255,.1)!important; }
.pw-eye svg { width:19px;height:19px;pointer-events:none; }

/* ===== Actions sous les réponses, sources, chapitres épinglés ===== */
#nav-pins::before { content:"§"; }
.ak-msg-actions { display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:12px; }
.ak-act { display:inline-flex;align-items:center;gap:6px;padding:5px 11px;border-radius:999px;border:1px solid var(--ak-line)!important;
  background:transparent!important;color:var(--ak-muted)!important;-webkit-text-fill-color:var(--ak-muted)!important;font-size:12.5px;font-weight:500;
  cursor:pointer;line-height:1.3;transition:background .15s,border-color .15s,color .15s; }
.ak-act:hover { background:var(--ak-soft)!important;border-color:var(--ak-blue)!important;color:var(--ak-blue-dark)!important;-webkit-text-fill-color:var(--ak-blue-dark)!important; }
.ak-act.on { background:var(--ak-soft)!important;border-color:var(--ak-blue)!important;color:var(--ak-blue-dark)!important;-webkit-text-fill-color:var(--ak-blue-dark)!important; }
.ak-act span { color:inherit!important;-webkit-text-fill-color:inherit!important; }
.ak-timing { font-size:12px;color:var(--ak-muted)!important;margin-left:auto; }
.ak-src-list { margin:10px 0 0;padding:10px 12px 10px 30px;border:1px solid var(--ak-line);border-radius:12px;background:var(--ak-soft);font-size:12.5px;line-height:1.5; }
.ak-src-list[hidden] { display:none!important; }
.ak-src-list li { margin:6px 0; }
.ak-src-meta { color:var(--ak-muted)!important;font-weight:600; }
.ak-src-quote { color:var(--ak-text)!important;opacity:.85;font-style:italic;margin-top:2px; }
.pin-list { display:grid;gap:14px; }
.pin-card { background:var(--ak-card);border:1px solid var(--ak-line);border-radius:16px;padding:16px 18px; }
.pin-head { display:flex;flex-direction:column;gap:4px;margin-bottom:6px; }
.pin-head h3 { margin:0;font-size:16px;color:var(--ak-text)!important; }
.pin-num { font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--ak-blue)!important; }
.pin-body { font-size:14px;line-height:1.6;color:var(--ak-text)!important; }
.pin-body p,.pin-body li { color:var(--ak-text)!important;margin:6px 0; }
.pin-card .ak-msg-actions { margin-top:12px; }
#ak-toast { position:fixed;left:50%;bottom:28px;transform:translateX(-50%) translateY(20px);z-index:99999;background:#182033;color:#fff;
  padding:11px 18px;border-radius:12px;font-size:13.5px;box-shadow:0 10px 30px rgba(15,23,42,.35);opacity:0;pointer-events:none;transition:opacity .2s,transform .2s; }
#ak-toast.show { opacity:1;transform:translateX(-50%) translateY(0); }
body.akori-reconnecting #login-view { visibility:hidden; }

/* ===== Badge utilisateur (menu) ===== */
#user-badge { flex:none!important; }
.user-badge { display:flex;align-items:center;gap:10px;padding:10px 2px 4px;white-space:nowrap;overflow:hidden; }
.ub-avatar { width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#4f6df5,#8e6cf4);color:#fff!important;font-weight:800;font-size:14px;display:flex;align-items:center;justify-content:center;flex:none;margin-left:7px; }
.ub-text b { display:block;font-size:13px;color:var(--ak-text);line-height:1.2; }
.ub-text span { display:block;font-size:11px;color:#8993a7;line-height:1.3; }
#sidebar #nav-logout { margin-top:0!important; }

/* ===== Page de connexion ===== */
#login-view { min-height:calc(100vh - 40px);display:flex!important;flex-direction:column!important;align-items:center!important;justify-content:center!important;gap:18px!important;padding:28px 14px; }
.login-hero { text-align:center; }
.login-hero .brand-mark { width:68px;height:68px;border-radius:20px;background:linear-gradient(135deg,#4768f5,#8e6cf4);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:34px;margin:0 auto 12px;box-shadow:0 12px 28px rgba(79,109,245,.28); }
.login-hero h1 { margin:0;font-size:34px;letter-spacing:.04em;color:var(--ak-text)!important; }
.login-hero p { margin:6px 0 0;font-size:12px;letter-spacing:.06em;color:#7b8497!important; }
#login-card { width:min(440px,100%)!important;max-width:440px;flex:none!important;background:#fff!important;border:1px solid var(--ak-line)!important;border-radius:22px!important;padding:26px!important;box-shadow:0 18px 50px rgba(42,55,90,.10);gap:12px!important; }
.login-welcome h2 { margin:0 0 4px;font-size:21px;color:var(--ak-text)!important; }
.login-welcome p { margin:0 0 6px;font-size:13px;color:#778298!important;line-height:1.5; }
#li-btn, #su-btn { width:100%!important;min-height:46px!important;margin-top:4px; }
#login-status { min-height:22px;font-size:13px; }
.login-foot { text-align:center;font-size:11.5px;color:#8993a7;line-height:1.5; }
#auth-tabs .tab-nav, #auth-tabs [role="tablist"] { display:flex!important;visibility:visible!important;height:auto!important;margin:0 0 10px!important; }

/* ===== Console d'administration ===== */
#admin-view { width:min(1240px,100%)!important;margin:0 auto!important;padding:18px 6px 40px;gap:16px!important; }
#admin-top { align-items:center!important;gap:12px!important;flex-wrap:wrap!important; }
.admin-title { display:flex;align-items:center;gap:14px; }
.admin-title h1 { margin:0;font-size:24px;color:var(--ak-text)!important; }
.admin-title p { margin:3px 0 0;font-size:13px;color:#778298!important; }
.admin-badge { width:50px;height:50px;border-radius:16px;background:linear-gradient(135deg,#4768f5,#8e6cf4);display:flex;align-items:center;justify-content:center;font-size:24px;box-shadow:0 8px 20px rgba(79,109,245,.25); }
.admin-kpis { display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px; }
.admin-kpi { background:#fff;border:1px solid var(--ak-line);border-radius:16px;padding:16px;box-shadow:0 4px 14px rgba(42,55,90,.03); }
.admin-kpi b { display:block;font-size:24px;color:var(--ak-text); }
.admin-kpi span { font-size:12px;color:#778298; }
.admin-kpi.warn b { color:#d95b67; }
.admin-service { display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin-bottom:12px; }
.admin-service > div { background:#fff;border:1px solid var(--ak-line);border-radius:14px;padding:12px 14px; }
.admin-service span { display:block;font-size:11.5px;color:#778298;margin-bottom:3px; }
.admin-service b { font-size:13.5px;color:var(--ak-text);word-break:break-all; }
#admin-top .gr-button, #admin-top button { min-height:44px; }
@media (max-width:700px) { #login-card { padding:20px!important; } .login-hero h1 { font-size:28px; } }

/* thème sombre : connexion, admin, badge */
html:root:root:root:root body.akori-dark #login-card,
html:root:root:root:root body.akori-dark .admin-kpi,
html:root:root:root:root body.akori-dark .admin-service > div { background:#161e30!important;border-color:#27324a!important; }
html:root:root:root:root body.akori-dark .login-hero h1, html:root:root:root:root body.akori-dark .login-welcome h2,
html:root:root:root:root body.akori-dark .admin-title h1, html:root:root:root:root body.akori-dark .admin-kpi b,
html:root:root:root:root body.akori-dark .admin-service b, html:root:root:root:root body.akori-dark .ub-text b { color:#e6ebf7!important; }
html:root:root:root:root body.akori-dark .login-welcome p, html:root:root:root:root body.akori-dark .login-hero p,
html:root:root:root:root body.akori-dark .admin-title p, html:root:root:root:root body.akori-dark .admin-kpi span,
html:root:root:root:root body.akori-dark .admin-service span, html:root:root:root:root body.akori-dark .login-foot { color:#9aa6bf!important; }

/* le conteneur racine de Gradio ne doit pas former un panneau plein */
.gradio-container > .main > .wrap { background:transparent!important;border-color:transparent!important;box-shadow:none!important; }
html:root:root:root:root body.akori-dark .gradio-container > .main > .wrap { background:transparent!important;border-color:transparent!important; }

/* ===== Messages techniques de Gradio masqués : l'utilisateur voit un message calme à la place ===== */
.toast-wrap { display:none!important; }
#akori-banner { position:fixed;top:16px;left:50%;transform:translate(-50%,-140%);z-index:9999;max-width:min(560px,92vw);
  background:#eef2ff;color:#34415b;border:1px solid #cfd8ff;border-radius:14px;padding:12px 18px;font-size:13.5px;font-weight:600;
  box-shadow:0 12px 34px rgba(42,55,90,.18);transition:transform .35s cubic-bezier(.4,0,.2,1);text-align:center; }
#akori-banner.show { transform:translate(-50%,0); }
html:root:root:root:root body.akori-dark #akori-banner { background:#1c2540;color:#dbe2f3;border-color:#2c3957; }

/* ===== Messages d'erreur : rouge naturel ===== */
.ak-err { color:#dc2626!important;-webkit-text-fill-color:#dc2626!important; }
.ak-ok { color:#15803d!important;-webkit-text-fill-color:#15803d!important; }
#assistant-chatbot .ak-err, #assistant-chatbot p.ak-err, #assistant-chatbot span.ak-err { color:#dc2626!important;-webkit-text-fill-color:#dc2626!important; }
html:root:root:root:root body.akori-dark .ak-err, html:root:root:root:root body.akori-dark #assistant-chatbot .ak-err { color:#f87171!important;-webkit-text-fill-color:#f87171!important; }
html:root:root:root:root body.akori-dark .ak-ok { color:#4ade80!important;-webkit-text-fill-color:#4ade80!important; }
#akori-banner { background:#fef2f2!important;color:#b91c1c!important;border:1px solid #fecaca!important; }
html:root:root:root:root body.akori-dark #akori-banner { background:#2a1215!important;color:#fca5a5!important;border-color:#5b2128!important; }

/* ===== Pages d'accès (connexion, inscription, onboarding) : style sombre ===== */
html:has(.akori-dark-page), html:root:root:root:root body:has(.akori-dark-page),
html:root:root:root:root body:has(.akori-dark-page) .gradio-container { background:#0a0a0c!important; }
.akori-dark-page { width:min(560px,100%)!important;max-width:560px;margin:0 auto!important;padding:28px 16px 56px!important;gap:14px!important;
  min-height:auto!important;justify-content:flex-start!important;align-items:stretch!important;background:transparent!important;border:0!important; }
.auth-brand { display:flex;align-items:center;justify-content:center;gap:12px;margin:22px 0 34px; }
.auth-brand .brand-mark { width:46px;height:46px;border-radius:13px;background:#f5f5f4;color:#111;font-weight:900;font-size:22px;display:flex;align-items:center;justify-content:center; }
.auth-brand span { font-size:28px;font-weight:800;color:#fff!important;letter-spacing:.01em; }
.auth-title { text-align:center;margin:6px 0 20px; }
.auth-title h1 { font-size:30px;font-weight:800;color:#fff!important;margin:0 0 8px;line-height:1.2; }
.auth-title p { color:#9a9aa6!important;font-size:16px;margin:0; }
.auth-panel { background:transparent!important;border:0!important;box-shadow:none!important;padding:0!important;gap:14px!important; }
.akori-dark-page .block, .akori-dark-page .form, .akori-dark-page .gap, .akori-dark-page fieldset { background:transparent!important;border:0!important;box-shadow:none!important; }
.akori-dark-page label > span, .akori-dark-page legend, .akori-dark-page [data-testid="block-info"] {
  background:transparent!important;color:#f4f4f6!important;padding:0!important;font-size:15px!important;font-weight:700!important;border:0!important; }
.akori-dark-page input[type="text"], .akori-dark-page input[type="password"], .akori-dark-page textarea {
  background:#141418!important;border:1px solid #26262d!important;border-radius:16px!important;color:#f4f4f6!important;-webkit-text-fill-color:#f4f4f6!important;
  min-height:56px!important;padding:0 20px!important;font-size:16px!important;box-shadow:none!important; }
.akori-dark-page textarea { padding:16px 20px!important; }
/* Saisie automatique de Chrome : fond clair + texte foncé, identique en aperçu, sélection et champ rempli (toujours lisible) */
html:root:root:root:root body .akori-dark-page input:-webkit-autofill, html:root:root:root:root body .akori-dark-page input:-webkit-autofill:hover,
html:root:root:root:root body .akori-dark-page input:-webkit-autofill:focus, html:root:root:root:root body .akori-dark-page input:-webkit-autofill:active {
  -webkit-text-fill-color:#0b1220!important;color:#0b1220!important;caret-color:#0b1220;
  box-shadow:0 0 0 1000px #e8f0fe inset!important;-webkit-box-shadow:0 0 0 1000px #e8f0fe inset!important;
  background-color:#e8f0fe!important;transition:background-color 600000s ease-out 0s!important; }
html:root:root:root:root body .akori-dark-page input:autofill, html:root:root:root:root body .akori-dark-page input:-internal-autofill-previewed,
html:root:root:root:root body .akori-dark-page input:-internal-autofill-selected { color:#0b1220!important;-webkit-text-fill-color:#0b1220!important;
  box-shadow:0 0 0 1000px #e8f0fe inset!important;background-color:#e8f0fe!important; }
.akori-dark-page input::placeholder, .akori-dark-page textarea::placeholder { color:#6c6c78!important;-webkit-text-fill-color:#6c6c78!important; }
.akori-dark-page input:focus, .akori-dark-page textarea:focus { border-color:#5a5a68!important;outline:none!important; }
.akori-dark-page .wrap { background:transparent!important;border:0!important; }
.akori-dark-page input[type="checkbox"] + span, .akori-dark-page .checkbox-label span { color:#cfcfd6!important; }

#li-btn, #su-btn, #ob-next, #ob-enter, #ob-back { width:100%!important;max-width:none!important;min-height:56px!important;border-radius:16px!important;font-size:16px!important;font-weight:700!important;box-shadow:none!important; }
#li-btn, #su-btn, #ob-next { background:#f5f5f4!important;background-image:none!important;color:#0a0a0c!important;-webkit-text-fill-color:#0a0a0c!important;border:0!important; }
#ob-enter { background:linear-gradient(90deg,#16b8d8,#7c3aed 55%,#ec2f8f)!important;color:#fff!important;-webkit-text-fill-color:#fff!important;border:0!important; }
#ob-back { background:transparent!important;color:#fff!important;-webkit-text-fill-color:#fff!important;border:1px solid #26262d!important; }
#li-btn:hover, #su-btn:hover, #ob-next:hover { background:#ffffff!important; }
.link-btn, .link-btn.gr-button { background:transparent!important;border:0!important;box-shadow:none!important;color:#9a9aa6!important;-webkit-text-fill-color:#9a9aa6!important;
  font-weight:500!important;min-height:30px!important;padding:0 4px!important;width:auto!important;flex:none!important; }
.link-btn.accent { color:#8b6cff!important;-webkit-text-fill-color:#8b6cff!important;font-weight:700!important; }
.link-btn:hover { text-decoration:underline; }
.akori-dark-page .link-btn:only-child, .auth-panel > .link-btn { align-self:center; }
.auth-foot-row { justify-content:center!important;align-items:center!important;gap:6px!important;flex-wrap:nowrap!important;background:transparent!important;border:0!important; }
.auth-foot-row span { color:#9a9aa6!important;font-size:15px; }
.auth-foot-row > * { flex:none!important;min-width:0!important;width:auto!important; }
.auth-status, .auth-status p { text-align:center;margin:0!important;min-height:22px;font-size:14.5px; }
.auth-status p { color:#f87171!important; }
.auth-status p.ak-ok { color:#4ade80!important; }

.ob-progress { display:flex;gap:10px;margin:0 auto 34px;width:min(340px,92%); }
.ob-progress span { flex:1;height:4px;border-radius:4px;background:#2a2a30; }
.ob-progress span.on { background:#fff; }
.ob-cards .wrap { display:flex!important;flex-direction:column!important;gap:12px!important; }
.ob-cards .wrap > label { background:#141418!important;border:1px solid #26262d!important;border-radius:16px!important;padding:16px 20px!important;cursor:pointer;
  color:#f4f4f6!important;line-height:1.45;display:flex!important;align-items:center;transition:border-color .15s ease,background .15s ease; }
.ob-cards .wrap > label span { display:block!important;white-space:pre-line!important;color:#cfcfd6!important;font-size:14px!important;font-weight:400!important; }
.ob-cards .wrap > label span::first-line { color:#fff;font-weight:700;font-size:16px; }
.ob-cards .wrap > label:hover { border-color:#3a3a44!important; }
.ob-cards .wrap > label:has(input:checked) { border-color:#fff!important;background:#1b1b21!important; }
.ob-cards input[type="radio"] { display:none!important; }
.ob-done { text-align:center;padding:60px 0 6px; }
.ob-rocket { width:100px;height:100px;border-radius:50%;background:#222226;display:flex;align-items:center;justify-content:center;font-size:42px;margin:0 auto 28px; }
.ob-done h1 { font-size:30px;font-weight:800;color:#fff!important;margin:0 0 14px; }
.ob-done p { color:#9a9aa6!important;font-size:17px;line-height:1.6;margin:0 auto;max-width:460px; }
.ob-actions { gap:12px!important; }

/* finitions des pages d'accès */
.akori-dark-page .input-container, .akori-dark-page .container, .akori-dark-page label.block, .akori-dark-page .wrap-inner { background:transparent!important;border:0!important;box-shadow:none!important; }
.akori-dark-page input[type="checkbox"] { appearance:auto;width:18px!important;height:18px!important;accent-color:#8b6cff;background:#141418!important;border:1px solid #3a3a44!important;border-radius:5px; }
.akori-dark-page input[type="checkbox"]:checked { background-color:#8b6cff!important; }
footer, .gradio-container footer { display:none!important; }
"""


THEME_TOGGLE_JS = r"""
() => {
  const dark = !document.body.classList.contains('akori-dark');
  document.body.classList.toggle('akori-dark', dark);
  document.body.classList.toggle('dark', dark);
  try { localStorage.setItem('akori-theme', dark ? 'dark' : 'light'); } catch (e) {}
}
"""

LANG_APPLY_JS = r"""
(label) => { window.__akTouched = true; if (window.__akoriSetLang) window.__akoriSetLang(label); }
"""

I18N_SRC = r"""
  const I18N = [["NAVIGATION", "NAVIGATION"], ["Accueil", "Home"], ["Mes dossiers", "My folders"], ["Réviser", "Revise"], ["Flashcards", "Flashcards"], ["Quiz / QCM", "Quiz / MCQ"], ["Résumé", "Summary"], ["Assistant IA", "AI Assistant"], ["Progression", "Progress"], ["À revoir", "To review"], ["Historique", "History"], ["Paramètres", "Settings"], ["Administration", "Administration"], ["Déconnexion", "Log out"], ["Administrateur", "Administrator"], ["Utilisateur", "User"], ["AKORI · Espace de révision", "AKORI · Revision space"], ["Cours actif", "Active course"], ["Bonjour 👋", "Hello 👋"], ["Prêt à booster votre révision ?", "Ready to boost your revision?"], ["AKORI transforme vos cours PDF en un espace de révision intelligent : résumé, flashcards roulette, quiz et assistant RAG.", "AKORI turns your PDF courses into a smart revision space: summary, roulette flashcards, quiz and RAG assistant."], ["💡 Guide de révision AKORI & mode roulette flashcards", "💡 AKORI revision guide & flashcards roulette mode"], ["1. Chargez vos dossiers", "1. Load your folders"], ["Glissez vos PDF dans Mes dossiers pour activer l'indexation FAISS.", "Drop your PDFs in My folders to enable FAISS indexing."], ["2. Lancement roulette", "2. Start the roulette"], ["Générez au moins 7 flashcards. Les questions défilent automatiquement en boucle.", "Generate at least 7 flashcards. Questions scroll automatically in a loop."], ["3. Clic & réponse effacée", "3. Click & answer hidden"], ["Cliquez sur la carte pour stopper/relancer. La réponse s'efface à chaque relance.", "Click the card to stop/restart. The answer is hidden at each restart."], ["Cours importés", "Imported courses"], ["Fragments indexés", "Indexed fragments"], ["Échanges du cours", "Course exchanges"], ["Recherche active", "Search active"], ["Réviser un cours en un clic", "Revise a course in one click"], ["Générez les outils principaux à partir du cours actif.", "Generate the main tools from the active course."], ["Résumé · Flashcards · Quiz / QCM · Assistant IA", "Summary · Flashcards · Quiz / MCQ · AI Assistant"], ["Mes cours", "My courses"], ["Vos supports de révision indexés localement.", "Your revision materials indexed locally."], ["Aucun cours sélectionné", "No course selected"], ["Commencer la révision →", "Start revising →"], ["Tous vos supports PDF sont centralisés ici. Dès qu'un document est sélectionné, AKORI extrait son contenu et construit automatiquement son index.", "All your PDF materials are gathered here. As soon as a document is selected, AKORI extracts its content and builds its index automatically."], ["＋\nAjouter un document", "＋\nAdd a document"], ["Sélectionnez un PDF : extraction et indexation automatiques.", "Select a PDF: automatic extraction and indexing."], ["Indexé", "Indexed"], ["Sélectionnez un cours pour afficher son espace de révision.", "Select a course to display its revision space."], ["Date inconnue", "Unknown date"], ["Général", "General"], ["📁 Aucun document pour le moment.", "📁 No document yet."], ["Ajoutez votre premier PDF pour commencer.", "Add your first PDF to get started."], ["📚 Aucun cours pour le moment.", "📚 No course yet."], ["Importez votre premier PDF dans 'Mes dossiers' pour commencer.", "Import your first PDF in 'My folders' to get started."], ["Aucun document chargé.", "No document loaded."], ["Réviser ce cours", "Revise this course"], ["Une vue centrale pour accéder rapidement au résumé, aux flashcards, au quiz et à l'assistant.", "A central view to quickly reach the summary, flashcards, quiz and assistant."], ["Flashcards — Mode Roulette", "Flashcards — Roulette mode"], ["✦ Générer les flashcards", "✦ Generate flashcards"], ["🔄 Régénérer (7 minimum)", "🔄 Regenerate (7 minimum)"], ["Afficher la réponse", "Show the answer"], ["✓ Je savais", "✓ I knew it"], ["↻ À revoir", "↻ To review"], ["Aucune flashcard générée.", "No flashcards generated."], ["Choisissez un cours puis cliquez sur « Générer les flashcards ».", "Choose a course then click “Generate flashcards”."], ["SESSION TERMINÉE", "SESSION COMPLETED"], ["Maîtrise", "Mastery"], ["Maîtrisées", "Mastered"], ["Cliquez sur « 🔄 Régénérer (7 minimum) » pour obtenir une nouvelle série.", "Click “🔄 Regenerate (7 minimum)” to get a new set."], ["Quiz d'évaluation", "Evaluation quiz"], ["Cliquez sur une réponse, validez, puis passez à la question suivante.", "Click an answer, validate, then go to the next question."], ["✦ Générer le quiz", "✦ Generate the quiz"], ["🔁 Rejouer", "🔁 Replay"], ["Valider la réponse", "Validate the answer"], ["Question suivante →", "Next question →"], ["Voir le résultat 🎯", "See the result 🎯"], ["Choisissez votre réponse", "Choose your answer"], ["Aucun quiz généré.", "No quiz generated."], ["Choisissez un cours puis cliquez sur « Générer le quiz ».", "Choose a course then click “Generate the quiz”."], ["QUIZ TERMINÉ", "QUIZ COMPLETED"], ["Votre résultat a été enregistré dans la progression de ce cours.", "Your result has been saved in this course's progress."], ["🏆 Sans faute, bravo !", "🏆 Perfect score, well done!"], ["✅ Bonne réponse !", "✅ Correct answer!"], ["❌ Réponse incorrecte", "❌ Incorrect answer"], ["Explication :", "Explanation:"], ["Résumé du cours", "Course summary"], ["Générer le résumé", "Generate the summary"], ["Sélectionnez un cours puis lancez la génération.", "Select a course then start the generation."], ["Assistant AKORI", "AKORI Assistant"], ["Posez une question sur le cours actif. Le moteur récupère d'abord les passages pertinents avec FAISS, puis Gemini génère la réponse à partir du contexte récupéré.", "Ask a question about the active course. The engine first retrieves the relevant passages with FAISS, then Gemini generates the answer from the retrieved context."], ["Envoyer", "Send"], ["Posez une question sur le cours…", "Ask a question about the course…"], ["Ma progression", "My progress"], ["Commencez par la vue globale, puis consultez le détail du cours sélectionné.", "Start with the global view, then check the details of the selected course."], ["Détail du cours sélectionné", "Selected course details"], ["Votre progression commencera ici.", "Your progress will start here."], ["Importez un cours pour créer votre premier suivi. Tant qu'aucun cours n'est chargé, la progression reste à 0 %.", "Import a course to create your first tracking. As long as no course is loaded, progress stays at 0%."], ["Aucune progression à afficher.", "No progress to display."], ["Importez un cours puis utilisez les flashcards ou terminez un quiz pour commencer à construire votre progression.", "Import a course then use the flashcards or finish a quiz to start building your progress."], ["VUE D'ENSEMBLE", "OVERVIEW"], ["Progression globale", "Global progress"], ["Synthèse de tous vos cours réellement importés et de vos interactions de révision.", "Summary of all your imported courses and your revision interactions."], ["Cours suivis", "Tracked courses"], ["Cours commencés", "Started courses"], ["Flashcards maîtrisées", "Mastered flashcards"], ["Moyenne des quiz", "Quiz average"], ["Progression par cours", "Progress by course"], ["Le détail pédagogique reste lié au cours sélectionné : points forts, points faibles, flashcards et résultats des quiz.", "The detailed breakdown stays linked to the selected course: strengths, weaknesses, flashcards and quiz results."], ["Calculée uniquement à partir des interactions enregistrées sur ce cours.", "Computed only from the interactions recorded on this course."], ["Quiz terminés", "Completed quizzes"], ["✦ Points forts", "✦ Strengths"], ["↗ Points faibles", "↗ Weaknesses"], ["Pas encore assez de réponses de quiz pour identifier un point fort.", "Not enough quiz answers yet to identify a strength."], ["Pas encore assez de réponses de quiz pour identifier un point faible.", "Not enough quiz answers yet to identify a weakness."], ["Aucun point fort clairement établi pour le moment.", "No clear strength established yet."], ["Aucun point faible clairement établi pour le moment.", "No clear weakness established yet."], ["Cette section regroupera les flashcards marquées « À revoir » et les erreurs de quiz.", "This section will gather the flashcards marked “To review” and the quiz mistakes."], ["📌 Votre file « À revoir » apparaîtra ici après les interactions.", "📌 Your “To review” queue will appear here after some interactions."], ["Historique du cours actif", "Active course history"], ["Aucun historique pour le moment.", "No history yet."], ["Aucune conversation enregistrée.", "No conversation recorded."], ["Vous", "You"], ["Personnalisez AKORI. Vos choix sont mémorisés dans ce navigateur.", "Customize AKORI. Your choices are saved in this browser."], ["Mode sombre", "Dark mode"], ["Basculez entre le thème clair et le thème sombre d'un seul clic.", "Switch between the light and dark theme with one click."], ["Langue", "Language"], ["Langue de l'interface et des réponses générées par l'IA.", "Language of the interface and of the generated summaries, flashcards and quizzes. In the chat, AKORI answers in the language you write in."], ["Connexion", "Sign in"], ["Créer un compte", "Create an account"], ["Nom d'utilisateur", "Username"], ["Mot de passe", "Password"], ["Confirmer le mot de passe", "Confirm the password"], ["Rester connecté", "Stay signed in"], ["Se connecter", "Sign in"], ["Créer mon compte", "Create my account"], ["Bienvenue sur AKORI", "Welcome to AKORI"], ["Connectez-vous pour retrouver vos cours, flashcards, quiz et historiques archivés.", "Sign in to find your archived courses, flashcards, quizzes and history."], ["Vos données sont enregistrées dans votre compte et restent disponibles à chaque connexion.", "Your data is saved in your account and available every time you sign in."], ["⚠️ Nom d'utilisateur invalide (3 à 32 caractères : lettres, chiffres, . _ -).", "⚠️ Invalid username (3 to 32 characters: letters, digits, . _ -)."], ["⚠️ Mot de passe trop court (6 caractères minimum).", "⚠️ Password too short (6 characters minimum)."], ["⚠️ Ce nom d'utilisateur existe déjà.", "⚠️ This username already exists."], ["✅ Compte créé.", "✅ Account created."], ["⚠️ Identifiants incorrects.", "⚠️ Incorrect username or password."], ["⛔ Ce compte est désactivé. Contactez l'administrateur.", "⛔ This account is disabled. Contact the administrator."], ["⏳ Trop de tentatives. Réessayez dans une minute.", "⏳ Too many attempts. Try again in a minute."], ["⚠️ Les mots de passe ne correspondent pas.", "⚠️ The passwords do not match."], ["✅ Vous êtes déconnecté.", "✅ You are signed out."], ["Connexion requise.", "Sign-in required."], ["⏳ Analyse du PDF… extraction, découpage et indexation en cours.", "⏳ Analyzing the PDF… extraction, splitting and indexing in progress."], ["⏳ Extraction du texte…", "⏳ Extracting the text…"], ["⏳ Enregistrement…", "⏳ Saving…"], ["⚠️ Aucun fichier sélectionné.", "⚠️ No file selected."], ["⚠️ Aucun texte exploitable dans ce PDF (document scanné ?).", "⚠️ No usable text in this PDF (scanned document?)."], ["⚠️ Veuillez d'abord sélectionner un cours.", "⚠️ Please select a course first."], ["⏳ Génération…", "⏳ Generating…"], ["⚠️ Le titre du document est indisponible.", "⚠️ The document title is unavailable."], ["⚠️ Aucune information pertinente n'a pu être extraite du document.", "⚠️ No relevant information could be extracted from the document."], ["Gemini est très sollicité en ce moment. Réessayez dans une dizaine de secondes.", "Gemini is very busy right now. Try again in about ten seconds."], ["Quota ou limite temporaire Gemini atteinte. Attendez un peu avant de relancer la génération.", "Gemini quota or temporary limit reached. Wait a little before generating again."], ["⚠️ Gemini est temporairement indisponible. Réessayez dans quelques secondes.", "⚠️ Gemini is temporarily unavailable. Try again in a few seconds."], ["⚠️ La limite temporaire de Gemini a été atteinte. Attendez un peu puis réessayez.", "⚠️ Gemini's temporary limit was reached. Wait a little then try again."], ["⚠️ Impossible de générer les flashcards à partir du contexte.", "⚠️ Unable to generate flashcards from the context."], ["⚠️ Impossible de générer le quiz.", "⚠️ Unable to generate the quiz."], ["⚠️ Sélectionnez d'abord un cours.", "⚠️ Select a course first."], ["⚠️ Aucun document sélectionné.", "⚠️ No document selected."], ["❌ Document vide.", "❌ Empty document."], ["❌ Le modèle n'a renvoyé aucune flashcard valide.", "❌ The model returned no valid flashcard."], ["⏳ Génération des flashcards…", "⏳ Generating flashcards…"], ["⏳ Génération du quiz…", "⏳ Generating the quiz…"], ["⏳ Génération du résumé…", "⏳ Generating the summary…"], ["🔁 Quiz relancé.", "🔁 Quiz restarted."], ["⚠️ Générez d'abord un quiz.", "⚠️ Generate a quiz first."], ["ℹ️ Réponse déjà validée : passez à la suite.", "ℹ️ Answer already validated: move on."], ["⚠️ Sélectionnez une réponse avant de valider.", "⚠️ Select an answer before validating."], ["✅ Bonne réponse.", "✅ Correct answer."], ["❌ Réponse incorrecte. Consultez l'explication.", "❌ Incorrect answer. Check the explanation."], ["⚠️ Validez d'abord cette réponse.", "⚠️ Validate this answer first."], ["Console d'administration", "Administration console"], ["Suivi, maintenance et gestion des comptes AKORI", "Monitoring, maintenance and account management for AKORI"], ["← Retour à l'application", "← Back to the app"], ["↻ Actualiser", "↻ Refresh"], ["Utilisateurs", "Users"], ["Activité", "Activity"], ["Maintenance", "Maintenance"], ["Nouveau mot de passe", "New password"], ["Activer / Désactiver", "Enable / Disable"], ["Réinitialiser le mot de passe", "Reset the password"], ["Promouvoir / Rétrograder", "Promote / Demote"], ["Supprimer le compte", "Delete the account"], ["Je confirme la suppression définitive du compte et de ses données", "I confirm the permanent deletion of the account and its data"], ["Sauvegarde complète (ZIP)", "Full backup (ZIP)"], ["Recharger les index", "Reload the indexes"], ["Purger les connexions expirées", "Purge expired sign-ins"], ["Alléger le journal", "Trim the log"], ["Sauvegarde", "Backup"], ["Comptes", "Accounts"], ["Actifs (24 h)", "Active (24 h)"], ["Stockage utilisé", "Storage used"], ["Requêtes IA", "AI requests"], ["Jetons estimés", "Estimated tokens"], ["Erreurs (24 h)", "Errors (24 h)"], ["Sessions en ligne", "Online sessions"], ["Modèle principal", "Main model"], ["Modèles de secours", "Fallback models"], ["Clé API Gemini", "Gemini API key"], ["Disponibilité", "Uptime"], ["Dossier de données", "Data folder"], ["Taille PDF maximale", "Maximum PDF size"], ["✅ configurée", "✅ configured"], ["⛔ Accès réservé à l'administrateur.", "⛔ Reserved for the administrator."], ["⚠️ Sélectionnez un utilisateur.", "⚠️ Select a user."], ["⚠️ Vous ne pouvez pas désactiver votre propre compte.", "⚠️ You cannot disable your own account."], ["⚠️ Vous ne pouvez pas supprimer votre propre compte.", "⚠️ You cannot delete your own account."], ["⚠️ Il doit rester au moins un administrateur.", "⚠️ At least one administrator must remain."], ["⚠️ Cochez la confirmation pour supprimer définitivement ce compte et ses données.", "⚠️ Tick the confirmation to permanently delete this account and its data."], ["Rôle", "Role"], ["Statut", "Status"], ["Créé le", "Created on"], ["Dernière connexion", "Last sign-in"], ["Connexions", "Sign-ins"], ["Cours", "Courses"], ["Échanges", "Exchanges"], ["Quiz", "Quizzes"], ["Stockage", "Storage"], ["Actif", "Active"], ["Désactivé", "Disabled"], ["Date", "Date"], ["Événement", "Event"], ["Détail", "Detail"], ["⚠️ Session expirée : rechargez la page pour vous reconnecter.", "⚠️ Session expired: reload the page to sign in again."], ["Gemini met trop de temps à répondre. Réessayez dans un instant.", "Gemini is taking too long to answer. Try again in a moment."], ["Content de te revoir", "Good to see you again"], ["Connecte-toi pour accéder à tes cours", "Sign in to access your courses"], ["Email", "Email"], ["Mot de passe oublié ?", "Forgot your password?"], ["Pas encore de compte ?", "No account yet?"], ["Crée ton compte", "Create your account"], ["Gratuit, sans carte bancaire", "Free, no credit card"], ["Déjà un compte ?", "Already have an account?"], ["ton@email.com", "you@email.com"], ["8 caractères minimum", "8 characters minimum"], ["Comment tu t'appelles ?", "What's your name?"], ["Et quel est ton niveau scolaire ?", "And what's your school level?"], ["Prénom", "First name"], ["Ton prénom", "Your first name"], ["Niveau", "Level"], ["Objectif", "Goal"], ["🏫 Collège\nDe la 6ème à la 3ème", "🏫 Middle school\nGrades 6 to 9"], ["🎓 Lycée\nSeconde, Première, Terminale", "🎓 High school\nYears 10 to 12"], ["🎯 Études supérieures\nUniversité, Prépa, BTS...", "🎯 Higher education\nUniversity, prep school, vocational..."], ["📚 Réviser mes cours\nRelire, comprendre et mémoriser régulièrement", "📚 Revise my courses\nReread, understand and memorize regularly"], ["📝 Préparer un examen\nFlashcards et quiz avant le jour J", "📝 Prepare for an exam\nFlashcards and quizzes before the big day"], ["🧠 Mieux comprendre un chapitre\nPoser mes questions directement au cours", "🧠 Understand a chapter better\nAsk my questions directly to the course"], ["Continuer", "Continue"], ["Retour", "Back"], ["Accéder à AKORI", "Open AKORI"], ["Que veux-tu faire avec AKORI ?", "What do you want to do with AKORI?"], ["Choisis ton objectif principal", "Choose your main goal"], ["Dans quelle langue veux-tu travailler ?", "Which language do you want to work in?"], ["Tu pourras la changer à tout moment dans les paramètres", "You can change it at any time in the settings"], ["Ton espace est prêt. Importe ton premier cours en PDF et laisse AKORI générer tes flashcards, quiz et résumés.", "Your space is ready. Import your first course as a PDF and let AKORI generate your flashcards, quizzes and summaries."], ["⚠️ Adresse e-mail invalide.", "⚠️ Invalid email address."], ["⚠️ Un compte existe déjà avec cet e-mail.", "⚠️ An account already exists with this email."], ["⚠️ Entre ton prénom et choisis ton niveau.", "⚠️ Enter your first name and choose your level."], ["⚠️ Choisis ton objectif principal.", "⚠️ Choose your main goal."], ["⚠️ Ton prénom est trop long (40 caractères maximum).", "⚠️ Your first name is too long (40 characters maximum)."], ["ℹ️ Pour réinitialiser ton mot de passe, contacte l'administrateur d'AKORI.", "ℹ️ To reset your password, contact the AKORI administrator."], ["Stockage", "Storage"], ["✅ Google Drive", "✅ Google Drive"], ["⚠️ Temporaire (non sauvegardé)", "⚠️ Temporary (not backed up)"], ["⏳ Mise à jour de l'index du cours (une seule fois)…", "⏳ Updating the course index (one time only)…"], ["OCR (PDF scannés)", "OCR (scanned PDFs)"], ["désactivé", "disabled"], ["⚠️ Aucun texte n'a pu être lu dans ce PDF scanné (OCR sans résultat).", "⚠️ No text could be read in this scanned PDF (OCR returned nothing)."], ["Tester Gemini", "Test Gemini"], ["Afficher le mot de passe", "Show password"], ["Épingler comme chapitre", "Pin as chapter"], ["Lire à voix haute", "Read aloud"], ["Arrêter la lecture", "Stop reading"], ["Épinglé ✓", "Pinned ✓"], ["Lecture vocale indisponible", "Voice reading unavailable"], ["Extrait", "Excerpt"], ["Chapitres", "Chapters"], ["Chapitres épinglés", "Pinned chapters"], ["Les réponses de l'assistant que vous épinglez depuis le chat sont rangées ici, cours par cours. Relisez-les, ou écoutez-les, avant l'examen.", "The assistant answers you pin from the chat are kept here, course by course. Reread them, or listen to them, before the exam."], ["Aucun chapitre épinglé pour ce cours.", "No pinned chapter for this course."], ["Dans l'assistant, cliquez sur « Épingler comme chapitre » sous une réponse utile.", "In the assistant, click “Pin as chapter” under a useful answer."], ["Retirer", "Remove"], ["📌 Chapitre épinglé", "📌 Chapter pinned"], ["Retrouvez-le dans À revoir.", "Find it in To review."], ["Recherche (médiane)", "Search (median)"], ["1er mot (médiane)", "First word (median)"], ["Réponse complète (médiane)", "Full answer (median)"], ["Réponse complète (95e centile)", "Full answer (95th percentile)"], ["Questions hors cours refusées", "Off-topic questions declined"], ["Modèle d'embeddings", "Embedding model"], ["Seuil de pertinence", "Relevance threshold"], ["Mesures sur", "Measured over"]];
  const PATTERNS = [["^Clé API Gemini invalide : elle contient des caractères non autorisés \\(accents, symboles\\)\\. Collez la clé exacte, qui commence par AIza, dans le secret GEMINI_API_KEY\\.$", "Invalid Gemini API key: it contains forbidden characters (accents, symbols). Paste the exact key, starting with AIza, into the GEMINI_API_KEY secret."], ["^Gemini est très sollicité en ce moment\\. Réessayez dans une dizaine de secondes\\.(.*)$", "Gemini is very busy right now. Try again in about ten seconds.$1"], ["^Quota Gemini atteint : la limite de requêtes de votre clé est épuisée \\(par minute ou par jour\\)\\. Attendez une minute ou utilisez une autre clé\\.(.*)$", "Gemini quota reached: your key's request limit is used up (per minute or per day). Wait a minute or use another key.$1"], ["^Aucun modèle Gemini n'est accessible avec cette clé \\(404\\)\\. Vérifiez GEMINI_API_KEY et les noms de modèles\\.(.*)$", "No Gemini model is available with this key (404). Check GEMINI_API_KEY and the model names.$1"], ["^Connexion à Gemini impossible \\(réseau ou délai dépassé\\)\\. Réessayez dans un instant\\.(.*)$", "Cannot reach Gemini (network or timeout). Try again in a moment.$1"], ["^⏳ Lecture OCR des pages scannées (\\d+)\\/(\\d+)…$", "⏳ Reading scanned pages with OCR $1/$2…"], ["^✅ '(.+)' indexé avec succès \\((\\d+) fragments, OCR sur (\\d+) pages\\)\\.$", "✅ '$1' indexed successfully ($2 fragments, OCR on $3 pages)."], ["^Chapitre (\\d+)$", "Chapter $1"], ["^(\\d+) réponse\\(s\\)$", "$1 answer(s)"], ["^✅ '(.+)' indexé avec succès \\((\\d+) fragments\\)\\.$", "✅ '$1' indexed successfully ($2 fragments)."], ["^✅ '(.+)' est déjà indexé : réutilisé instantanément\\.$", "✅ '$1' is already indexed: reused instantly."], ["^❌ Erreur lors de l'indexation : (.*)$", "❌ Indexing error: $1"], ["^⏳ Indexation (\\d+)\\/(\\d+)…$", "⏳ Indexing $1/$2…"], ["^⚠️ Fichier trop volumineux \\(maximum (\\d+) Mo\\)\\.$", "⚠️ File too large (maximum $1 MB)."], ["^✅ (\\d+) flashcards prêtes !$", "✅ $1 flashcards ready!"], ["^⚠️ Seulement (\\d+) flashcard\\(s\\) obtenue\\(s\\) : il en faut au moins (\\d+)\\. Relancez la génération\\.$", "⚠️ Only $1 flashcard(s) obtained: at least $2 are needed. Generate again."], ["^❌ Erreur : (.*)$", "❌ Error: $1"], ["^⚠️ Génération impossible : (.*)$", "⚠️ Generation failed: $1"], ["^✅ Quiz de (\\d+) questions généré\\.$", "✅ Quiz of $1 questions generated."], ["^🎯 Quiz terminé : (\\d+)\\/(\\d+)\\.$", "🎯 Quiz completed: $1/$2."], ["^Score : (\\d+)$", "Score: $1"], ["^(\\d+)% de bonnes réponses$", "$1% correct answers"], ["^Bonne réponse : (.*)$", "Correct answer: $1"], ["^Le titre du document est : (.*)$", "The title of the document is: $1"], ["^QUIZ \\/ QCM · (.*)$", "QUIZ / MCQ · $1"], ["^(\\d+) fragments indexés · Progression (\\d+)% · RAG local$", "$1 indexed fragments · Progress $2% · Local RAG"], ["^Progression réelle · (.*)$", "Real progress · $1"], ["^Ajouté le (.*)$", "Added on $1"], ["^Indexé · (\\d+) fragments$", "Indexed · $1 fragments"], ["^✅ Compte « (.+) » (activé|désactivé)\\.$", "✅ Account “$1” updated."], ["^✅ Mot de passe de « (.+) » réinitialisé.*$", "✅ Password of “$1” reset (their remembered sign-ins are revoked)."], ["^✅ « (.+) » est maintenant (administrateur|utilisateur)\\.$", "✅ “$1” role updated."], ["^✅ Compte « (.+) » et toutes ses données supprimés\\.$", "✅ Account “$1” and all its data deleted."], ["^✅ Sauvegarde créée \\((.+)\\)\\.$", "✅ Backup created ($1)."], ["^✅ (\\d+) connexion\\(s\\) mémorisée\\(s\\) expirée\\(s\\) purgée\\(s\\)\\.$", "✅ $1 expired remembered sign-in(s) purged."], ["^✅ Journal allégé : (\\d+) ligne\\(s\\) ancienne\\(s\\) supprimée\\(s\\)\\.$", "✅ Log trimmed: $1 old line(s) removed."], ["^✅ Index et cours rechargés.*$", "✅ Indexes and courses will be reloaded from disk on each user's next action."], ["^⏳ Génération des flashcards… \\((\\d+) s\\)$", "⏳ Generating flashcards… ($1 s)"], ["^⏳ Génération du quiz… \\((\\d+) s\\)$", "⏳ Generating the quiz… ($1 s)"], ["^⏳ Génération du résumé… \\((\\d+) s\\)$", "⏳ Generating the summary… ($1 s)"], ["^C'est parti (.+) !$", "Let's go $1!"], ["^Bonjour (.+) 👋$", "Hello $1 👋"], ["^⚠️ Mot de passe trop court \\((\\d+) caractères minimum\\)\\.$", "⚠️ Password too short ($1 characters minimum)."]].map(([re, to]) => [new RegExp(re), to]);
  const EXACT = new Map(I18N.map(([fr, en]) => [fr.trim(), en]));
  const codeOf = (label) => /English/.test(String(label)) ? 'en' : 'fr';
  const labelOf = (code) => code === 'en' ? 'English' : 'Français';
  window.__akoriLang = window.__akoriLang || 'fr';
  const trCore = (t) => {
    if (EXACT.has(t)) return EXACT.get(t);
    for (const [re, to] of PATTERNS) if (re.test(t)) return t.replace(re, to);
    return null;
  };
  const tr = (frText, lang) => {
    if (lang === 'fr') return frText;
    const key = frText.trim();
    if (!key) return frText;
    let out = trCore(key);
    if (out === null) {   // préfixe de symboles (⚠️, ✅, ⏳…) devant un message connu
      const m = key.match(/^([^\p{L}\p{N}'"«]*\s*)([\s\S]+)$/u);
      if (m && m[1]) { const inner = trCore(m[2]); if (inner !== null) out = m[1] + inner; }
    }
    if (out === null) return frText;
    return frText.match(/^\s*/)[0] + out + frText.match(/\s*$/)[0];
  };
  const SKIP = new Set(['SCRIPT', 'STYLE', 'TEXTAREA', 'IFRAME']);
  const translateTextNode = (node) => {
    // Un texte réécrit par l'application (donc de nouveau en français) est détecté et retraduit.
    if (node.__akLast === undefined || node.nodeValue !== node.__akLast) node.__akFr = node.nodeValue;
    const out = tr(node.__akFr, window.__akoriLang);
    if (node.nodeValue !== out) node.nodeValue = out;
    node.__akLast = node.nodeValue;
  };
  const translateAttrs = (el) => {
    if (el.tagName && /^(INPUT|TEXTAREA)$/.test(el.tagName) && el.placeholder) {
      if (el.__akPh === undefined || el.placeholder !== el.__akPhLast) el.__akPh = el.placeholder;
      const out = tr(el.__akPh, window.__akoriLang);
      if (el.placeholder !== out) el.placeholder = out;
      el.__akPhLast = el.placeholder;
    }
  };
  const walk = (root) => {
    if (!root) return;
    if (root.nodeType === 3) { if (root.parentNode && !SKIP.has(root.parentNode.tagName)) translateTextNode(root); return; }
    if (root.nodeType !== 1 || SKIP.has(root.tagName)) return;
    translateAttrs(root);
    const tw = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    let n;
    while ((n = tw.nextNode())) {
      if (n.nodeType === 3) { if (n.parentNode && !SKIP.has(n.parentNode.tagName)) translateTextNode(n); }
      else translateAttrs(n);
    }
  };
  let pending = false;
  const schedule = (nodes) => {
    window.__akPendingNodes = window.__akPendingNodes || new Set();
    nodes.forEach((n) => window.__akPendingNodes.add(n));
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      const set = window.__akPendingNodes; window.__akPendingNodes = new Set();
      set.forEach((n) => { if (n.isConnected) walk(n); });
    });
  };
  window.__akoriSetLang = (label) => {
    const code = codeOf(label);
    window.__akoriLang = code;
    try { localStorage.setItem('akori-lang', code); } catch (e) {}
    document.documentElement.lang = code;
    walk(document.body);
  };
  if (!window.__akoriI18nObserver) {
    window.__akoriI18nObserver = new MutationObserver((muts) => {
      if (window.__akoriLang === 'fr' && !window.__akTouched) return;
      const nodes = [];
      muts.forEach((m) => {
        if (m.type === 'characterData') nodes.push(m.target);
        else m.addedNodes.forEach((a) => nodes.push(a));
      });
      if (nodes.length) schedule(nodes);
    });
    window.__akoriI18nObserver.observe(document.body, { childList: true, subtree: true, characterData: true });
  }
"""

THEME_INIT_JS = r"""
() => {
  let dark = false, pref = null;
  try {
    dark = localStorage.getItem('akori-theme') === 'dark';
    pref = localStorage.getItem('akori-sidebar');
  } catch (e) {}
  const small = window.innerWidth <= 900;
  const collapsed = pref ? pref === 'closed' : small;
  document.body.classList.toggle('akori-dark', dark);
  document.body.classList.toggle('dark', dark);
  document.body.classList.toggle('sidebar-collapsed', collapsed);
  /*I18N*/
  let lang = 'fr';
  try { lang = localStorage.getItem('akori-lang') || 'fr'; } catch (e) {}
  if (lang !== 'fr') window.__akTouched = true;
  window.__akoriSetLang(labelOf(lang));
  if (!window.__akoriStatusColors) {
    window.__akoriStatusColors = true;
    // Messages d'erreur en rouge, naturels : tout texte qui commence par ⚠️ ❌ ⛔ (✅ en vert).
    const colorize = (root) => {
      if (!root || root.nodeType !== 1) return;
      const els = [root];
      if (root.querySelectorAll) els.push(...root.querySelectorAll('p,span,div,li'));
      for (const el of els) {
        if (el.childElementCount) continue;
        const t = (el.textContent || '').trim();
        const want = !t || t.length > 400 ? null : (/^[⚠❌⛔]/.test(t) ? 'ak-err' : (/^✅/.test(t) ? 'ak-ok' : null));
        if (want) { if (!el.classList.contains(want)) { el.classList.remove('ak-err', 'ak-ok'); el.classList.add(want); } }
        else if (el.classList.contains('ak-err') || el.classList.contains('ak-ok')) el.classList.remove('ak-err', 'ak-ok');
      }
    };
    let queued = new Set(), scheduled = false;
    new MutationObserver((muts) => {
      for (const m of muts) {
        if (m.type === 'characterData') { if (m.target.parentElement) queued.add(m.target.parentElement); }
        else m.addedNodes.forEach((n) => { if (n.nodeType === 1) queued.add(n); else if (n.parentElement) queued.add(n.parentElement); });
      }
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(() => { scheduled = false; const todo = queued; queued = new Set(); todo.forEach((n) => n.isConnected && colorize(n)); });
    }).observe(document.body, { childList: true, subtree: true, characterData: true });
    colorize(document.body);
  }
  if (!window.__akoriRecoveryBound) {
    window.__akoriRecoveryBound = true;
    // Les erreurs réseau (tunnel, coupure…) ne doivent jamais inquiéter l'utilisateur : message calme + reprise automatique.
    const TRANSPORT = /Connection to the server was lost|Attempting reconnection|Could not parse server response|Connection errored out|Failed to fetch|NetworkError|Load failed|Unexpected token|Unexpected end of JSON|Bad Gateway|Gateway Time/i;
    const showBanner = (msg) => {
      let b = document.getElementById('akori-banner');
      if (!b) { b = document.createElement('div'); b.id = 'akori-banner'; document.body.appendChild(b); }
      b.textContent = msg; b.classList.add('show');
      clearTimeout(window.__akBannerTimer);
      window.__akBannerTimer = setTimeout(() => b.classList.remove('show'), 9000);
    };
    window.__akBanner = showBanner;
    document.addEventListener('click', (ev) => {
      const b = ev.target.closest && ev.target.closest('#sidebar .navbtn');
      if (b && b.id) { try { sessionStorage.setItem('akori-tab', b.id); } catch (e) {} }
    });
    let recovering = false;
    const recover = () => {
      if (recovering) return;
      recovering = true;
      try { sessionStorage.setItem('akori-recovering', '1'); } catch (e) {}
      const en = window.__akoriLang === 'en';
      showBanner(en ? 'Connection interrupted. Retrying…' : 'Connexion interrompue. Nouvelle tentative…');
      [2500, 8000, 18000, 35000].forEach((ms) => setTimeout(() => { const b = document.getElementById('resync-btn'); if (b) b.click(); }, ms));
      setTimeout(() => { recovering = false; }, 36000);
    };
    new MutationObserver((muts) => {
      for (const m of muts) for (const n of m.addedNodes) {
        if (n.nodeType !== 1) continue;
        const toast = (n.closest && n.closest('.toast-wrap')) || (n.querySelector ? n.querySelector('.toast-wrap') : null);
        if (!toast) continue;
        const text = toast.innerText || '';
        console.warn('[AKORI] message technique masqué :', text.replace(/\s+/g, ' ').trim());
        if (TRANSPORT.test(text)) recover();
        else showBanner(window.__akoriLang === 'en' ? 'That action did not complete. Please try again.'
                                                   : "L'action n'a pas abouti. Réessaie dans un instant.");
        toast.style.display = 'none';
      }
    }).observe(document.body, { childList: true, subtree: true });
    window.addEventListener('unhandledrejection', (ev) => {
      if (TRANSPORT.test(String(ev.reason && (ev.reason.message || ev.reason)))) { ev.preventDefault(); recover(); }
    });
  }
  if (!window.__akoriSidebarBound) {
    window.__akoriSidebarBound = true;
    // Le menu est fixe : on l'aligne sur le bord gauche réel de la grille Gradio.
    const alignSidebar = () => {
      const main = document.querySelector('#main-column');
      if (!main || !main.parentElement) return;
      const left = main.parentElement.getBoundingClientRect().left;
      document.documentElement.style.setProperty('--sb-left', Math.max(8, Math.round(left)) + 'px');
    };
    window.__akoriAlign = alignSidebar;
    alignSidebar();
    window.addEventListener('resize', alignSidebar);
    setTimeout(alignSidebar, 300);
    setTimeout(alignSidebar, 1200);
    document.addEventListener('click', (ev) => {
      // Écran étroit : après le choix d'une page, le menu se referme pour libérer l'affichage.
      if (window.innerWidth <= 900 && ev.target.closest('#sidebar .navbtn')) {
        document.body.classList.add('sidebar-collapsed');
        try { localStorage.setItem('akori-sidebar', 'closed'); } catch (e) {}
      }
    });
  }
  return labelOf(lang);
}
"""

THEME_INIT_JS = THEME_INIT_JS.replace('/*I18N*/', I18N_SRC)

ALIGN_JS = r"""
() => {
  document.body.classList.remove('akori-reconnecting');
  const run = () => window.__akoriAlign && window.__akoriAlign();
  setTimeout(run, 120); setTimeout(run, 500); setTimeout(run, 1200);
  try {
    // Après une reconnexion automatique : on rouvre l'onglet où l'utilisateur se trouvait.
    const recovering = sessionStorage.getItem('akori-recovering') === '1';
    const tab = sessionStorage.getItem('akori-tab');
    if (recovering) {
      sessionStorage.removeItem('akori-recovering');
      if (tab && tab !== 'nav-home' && tab !== 'nav-logout') setTimeout(() => { const b = document.getElementById(tab); if (b) b.click(); }, 700);
      const bn = document.getElementById('akori-banner'); if (bn) bn.classList.remove('show');
    }
  } catch (e) {}
}
"""

GET_TOKEN_JS = r"""
() => {
  try {
    let t = localStorage.getItem('akori-token') || '';
    if (!t) { const m = document.cookie.match(/(?:^|; )akori_token=([^;]+)/); t = m ? m[1] : ''; }
    if (t) document.body.classList.add('akori-reconnecting');
    return t;
  } catch (e) { return ''; }
}
"""

LOGOUT_JS = r"""
(t) => { try { sessionStorage.removeItem('akori-tab'); sessionStorage.removeItem('akori-recovering'); return [localStorage.getItem('akori-token') || '']; } catch (e) { return ['']; } }
"""

TOKEN_STORE_JS = r"""
(v) => {
  const setCookie = (t, persistent) => {
    document.cookie = 'akori_token=' + t + '; path=/; SameSite=Lax' + (persistent ? '; max-age=2592000' : '') +
      (location.protocol === 'https:' ? '; Secure' : '');
  };
  try {
    if (v && v.startsWith('set:')) { localStorage.setItem('akori-token', v.slice(4)); setCookie(v.slice(4), true); }
    else if (v && v.startsWith('sess:')) { localStorage.removeItem('akori-token'); setCookie(v.slice(5), false); }
    else if (v && v.startsWith('clear:')) { localStorage.removeItem('akori-token'); document.cookie = 'akori_token=; path=/; max-age=0'; }
  } catch (e) {}
}"""

SIDEBAR_TOGGLE_JS = r"""
() => {
  const closed = document.body.classList.toggle('sidebar-collapsed');
  try { localStorage.setItem('akori-sidebar', closed ? 'closed' : 'open'); } catch (e) {}
}
"""

# Petit script client : la navigation reste dans la même page Gradio et
# ajoute seulement une transition visuelle au changement de section.
AKORI_NAV_JS = r"""
() => {
  const animatePage = () => {
    const tabs = document.querySelector('#main-tabs');
    if (!tabs) return;
    tabs.classList.remove('akori-switching');
    void tabs.offsetWidth;
    tabs.classList.add('akori-switching');
    window.setTimeout(() => tabs.classList.remove('akori-switching'), 260);
  };

  document.addEventListener('click', (event) => {
    const button = event.target.closest('#sidebar .navbtn button, #sidebar .navbtn');
    if (button) window.setTimeout(animatePage, 40);
  });

  // --- Mot de passe : icône œil dans le champ
  const EYE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>';
  const EYE_OFF = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M17.9 17.9A10.8 10.8 0 0 1 12 19C5.6 19 2 12 2 12a18 18 0 0 1 4.1-5.1M9.9 5.2A9.7 9.7 0 0 1 12 5c6.4 0 10 7 10 7a18 18 0 0 1-2.2 3.2"/><path d="M1 1l22 22"/><path d="M14.1 14.1a3 3 0 1 1-4.2-4.2"/></svg>';
  const addEyes = () => {
    ['li-pass', 'su-pass', 'su-pass2'].forEach((id) => {
      const box = document.getElementById(id);
      const input = box && box.querySelector('input');
      if (!input || box.querySelector('.pw-eye')) return;
      box.classList.add('pw-wrap');
      const btn = document.createElement('button');
      btn.type = 'button'; btn.className = 'pw-eye'; btn.innerHTML = EYE;
      btn.title = 'Afficher le mot de passe'; btn.setAttribute('aria-label', 'Afficher le mot de passe');
      btn.addEventListener('mousedown', (e) => e.preventDefault());
      btn.addEventListener('click', (e) => {
        e.preventDefault(); e.stopPropagation();
        const show = input.type === 'password';
        input.type = show ? 'text' : 'password';
        btn.innerHTML = show ? EYE_OFF : EYE; btn.classList.toggle('on', show);
      });
      box.appendChild(btn);
    });
  };
  new MutationObserver(addEyes).observe(document.body, { childList: true, subtree: true });
  addEyes();

  // --- Actions sous les réponses : épingler comme chapitre, lire à voix haute, sources.
  const FR_W = /\b(le|la|les|des|est|une|dans|pour|que|qui|et|du|sur)\b/gi;
  const EN_W = /\b(the|is|are|of|and|to|in|that|with|for|this|what)\b/gi;
  const guessLang = (t) => ((t.match(EN_W) || []).length > (t.match(FR_W) || []).length ? 'en-US' : 'fr-FR');
  let speakingBtn = null;
  const setLabel = (btn, text) => { const sp = btn.querySelector('span'); if (sp) sp.textContent = text; };
  const stopSpeech = () => {
    try { window.speechSynthesis.cancel(); } catch (e) {}
    if (speakingBtn) { speakingBtn.classList.remove('on'); setLabel(speakingBtn, 'Lire à voix haute'); speakingBtn = null; }
  };
  const textOf = (btn) => {
    const box = btn.closest('.pin-card') || btn.closest('[data-testid="bot"], .message.bot, .bot, .prose');
    if (!box) return '';
    const clone = box.cloneNode(true);
    clone.querySelectorAll('.ak-msg-actions, .ak-src-list, h3.pin-title, .pin-num, .pin-head').forEach((n) => n.remove());
    return (clone.textContent || '').replace(/\s+/g, ' ').trim();
  };
  const speak = (btn) => {
    if (!('speechSynthesis' in window)) { setLabel(btn, 'Lecture vocale indisponible'); return; }
    const same = speakingBtn === btn;
    stopSpeech();
    if (same) return;
    const clean = textOf(btn).replace(/\[\d+\]/g, '').replace(/[*_`#$]/g, '').trim();
    if (!clean) return;
    const lang = guessLang(clean);
    const parts = clean.match(/[^.!?;]+[.!?;]?/g) || [clean];
    speakingBtn = btn; btn.classList.add('on'); setLabel(btn, 'Arrêter la lecture');
    parts.forEach((p, i) => {
      const u = new SpeechSynthesisUtterance(p.trim());
      u.lang = lang;
      if (i === parts.length - 1) { u.onend = () => { if (speakingBtn === btn) stopSpeech(); }; }
      u.onerror = () => { if (speakingBtn === btn) stopSpeech(); };
      window.speechSynthesis.speak(u);
    });
  };
  const sendHidden = (boxId, btnId, value) => {
    const box = document.getElementById(boxId);
    const field = box && box.querySelector('textarea, input');
    const trigger = document.getElementById(btnId);
    if (!field || !trigger) return false;
    const proto = field.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(field, value);
    field.dispatchEvent(new Event('input', { bubbles: true }));
    window.setTimeout(() => trigger.click(), 90);
    return true;
  };
  const toast = (title, sub) => {
    let el = document.getElementById('ak-toast');
    if (!el) { el = document.createElement('div'); el.id = 'ak-toast'; document.body.appendChild(el); }
    el.innerHTML = '<b></b><div style="opacity:.8;font-size:12.5px;margin-top:2px"></div>';
    el.firstChild.textContent = title; el.lastChild.textContent = sub || '';
    el.classList.add('show');
    window.clearTimeout(window.__akToastT);
    window.__akToastT = window.setTimeout(() => el.classList.remove('show'), 2800);
  };
  document.addEventListener('click', (event) => {
    const btn = event.target.closest('.ak-act');
    if (!btn) return;
    event.preventDefault();
    const cls = btn.className;
    const act = (cls.match(/ak-a-(\w+)/) || [])[1];
    if (act === 'speak') speak(btn);
    else if (act === 'sources') {
      const list = btn.closest('.ak-msg-actions').nextElementSibling;
      if (list && list.classList.contains('ak-src-list')) { list.hidden = !list.hidden; btn.classList.toggle('on', !list.hidden); }
    } else if (act === 'pin') {
      if (btn.classList.contains('on')) return;
      if (sendHidden('pin-payload', 'pin-btn', ((cls.match(/ak-i-(\d+)/) || [])[1] || '') + ':' + Date.now())) {
        btn.classList.add('on'); setLabel(btn, 'Épinglé ✓');
        toast('📌 Chapitre épinglé', "Retrouvez-le dans À revoir.");
      }
    } else if (act === 'unpin') {
      sendHidden('unpin-payload', 'unpin-btn', (cls.match(/ak-id-(\w+)/) || [])[1] || '');
    }
  });
};
"""

# Gradio UI


# =============================================================================
# CONNEXION, SESSIONS ET CONSOLE D'ADMINISTRATION
# =============================================================================
def _user_badge_html(key):
    if not key:
        return ""
    u = _users().get(key, {})
    name = _display_name(key)
    role = "Administrateur" if u.get("role") == "admin" else "Utilisateur"
    return (f"<div class='user-badge'><div class='ub-avatar'>{_escape_html(name[:1].upper() or '?')}</div>"
            f"<div class='ub-text'><b>{_escape_html(name)}</b><span>{role}</span></div></div>")


def _views_for(user, status="", token=None):
    """Tout ce que l'interface doit afficher pour `user` (None = déconnecté)."""
    tok = _CUR_USER.set(user)
    try:
        names = list(documents_db.keys())
        sel = names[-1] if names else None
        data = (
            dashboard_html(sel), documents_html_v16(sel), course_overview_html(sel), course_overview_html(sel),
            global_progress_html(sel), progress_detail_html(sel), _chat_history_for_ui(sel), history_view(sel),
        )
    finally:
        _CUR_USER.reset(tok)
    logged = bool(user)
    admin = logged and _user_role(user) == "admin"
    profile = _profile(user) if logged else {}
    needs_onboarding = logged and not profile.get("onboarded", True)
    lang_update = gr.update(value="English" if profile.get("lang") == "en" else "Français") if profile.get("lang") else gr.update()
    return (
        gr.update(visible=not logged), gr.update(visible=needs_onboarding),
        gr.update(visible=logged and not needs_onboarding), gr.update(visible=False),
        _user_badge_html(user), gr.update(visible=admin),
        gr.update(choices=names, value=sel), *data,
        status, status, (gr.update() if token is None else token), gr.Tabs(selected="home"), lang_update,
    )


def do_login(username, password, remember, request: gr.Request):
    key, msg = _authenticate(username, password)
    if not key:
        return _views_for(None, msg)
    SESSIONS[request.session_hash] = key
    _SESSION_SEEN[request.session_hash] = time.time()
    _purge_sessions()
    if remember:
        return _views_for(key, "", f"set:{_issue_token(key)}")
    return _views_for(key, "", f"sess:{_issue_token(key, days=1)}")


def do_register(email, password, password2, remember, request: gr.Request):
    if str(password or "") != str(password2 or ""):
        return _views_for(None, "⚠️ Les mots de passe ne correspondent pas.")
    ok, msg = _create_user(email, password)
    if not ok:
        return _views_for(None, msg)
    return do_login(email, password, remember, request)


def do_logout(stored_token, request: gr.Request):
    key = SESSIONS.pop(request.session_hash, None)
    _revoke_token(stored_token)
    _revoke_token(_cookie_token(request))
    _log("logout", key or "")
    return _views_for(None, "", f"clear:{secrets.token_hex(3)}")


def do_auto_login(token, request: gr.Request):
    key = _user_from_token(token)
    if not key:
        return _views_for(None, "", f"clear:{secrets.token_hex(3)}" if token else None)
    SESSIONS[request.session_hash] = key
    _SESSION_SEEN[request.session_hash] = time.time()
    _log("auto_login", key)
    return _views_for(key)


def do_enter_app(request: gr.Request):
    """« Accéder à AKORI » à la fin de l'onboarding."""
    key = _session_user(request)
    return _views_for(key) if key else _views_for(None)


# ---- onboarding : 4 étapes (prénom + niveau, objectif, langue, prêt)
OB_LEVELS = [("🏫 Collège\nDe la 6ème à la 3ème", "college"), ("🎓 Lycée\nSeconde, Première, Terminale", "lycee"),
             ("🎯 Études supérieures\nUniversité, Prépa, BTS...", "superieur")]
OB_GOALS = [("📚 Réviser mes cours\nRelire, comprendre et mémoriser régulièrement", "revise"),
            ("📝 Préparer un examen\nFlashcards et quiz avant le jour J", "exam"),
            ("🧠 Mieux comprendre un chapitre\nPoser mes questions directement au cours", "understand")]


def _ob_progress_html(step):
    return "<div class='ob-progress'>" + "".join(f"<span class='{'on' if i <= step else ''}'></span>" for i in range(1, 5)) + "</div>"


def _ob_state(step, error="", name=""):
    done = (f"<div class='ob-done'><div class='ob-rocket'>🚀</div><h1>C'est parti {_escape_html(name)} !</h1>"
            "<p>Ton espace est prêt. Importe ton premier cours en PDF et laisse AKORI générer tes flashcards, quiz et résumés.</p></div>")
    return (step, gr.update(visible=step == 1), gr.update(visible=step == 2), gr.update(visible=step == 3),
            gr.update(visible=step == 4), _ob_progress_html(step), gr.update(visible=step < 4),
            gr.update(visible=step == 4), gr.update(visible=step > 1), error, done if step == 4 else "")


def ob_next_handler(step, name, level, goal, lang):
    step = int(step or 1)
    name = str(name or "").strip()
    if step == 1:
        if not name or not level:
            return _ob_state(1, "⚠️ Entre ton prénom et choisis ton niveau.")
        if len(name) > 40:
            return _ob_state(1, "⚠️ Ton prénom est trop long (40 caractères maximum).")
        return _ob_state(2)
    if step == 2:
        if not goal:
            return _ob_state(2, "⚠️ Choisis ton objectif principal.")
        return _ob_state(3)
    if step == 3:
        user = _CUR_USER.get()
        if user:
            _save_profile(user, name=name, level=level, goal=goal, lang="en" if _lang_code(lang) == "en" else "fr", onboarded=True)
            _log("onboarded", user, f"{level} · {goal}")
        return _ob_state(4, name=name)
    return _ob_state(4, name=name)


def ob_back_handler(step, name):
    return _ob_state(max(1, int(step or 1) - 1), name=str(name or "").strip())


def ob_reset():
    return (*_ob_state(1), "", None, None, "Français")


_SESSION_SEEN = {}   # session_hash -> dernier passage (purge des sessions inactives)


def _purge_sessions(max_idle=86400):
    now = time.time()
    for sid in [sid for sid, t in list(_SESSION_SEEN.items()) if now - t > max_idle]:
        _SESSION_SEEN.pop(sid, None)
        SESSIONS.pop(sid, None)


def _cookie_token(request):
    try:
        return (request.cookies or {}).get("akori_token", "")
    except Exception:
        return ""


def _session_user(request):
    """Utilisateur de la session ; si le serveur ne la connaît plus, on la retrouve grâce au cookie de connexion."""
    sid = getattr(request, "session_hash", None)
    user = SESSIONS.get(sid)
    if not user and request is not None:
        user = _user_from_token(_cookie_token(request))
        if user and sid:
            SESSIONS[sid] = user
    if user and sid:
        _SESSION_SEEN[sid] = time.time()
    return user


def _scoped(fn):
    """Exécute `fn` pour l'utilisateur connecté de la session (isolation des données entre comptes)."""
    import functools
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    req = inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None, annotation=gr.Request)
    new_sig = sig.replace(parameters=[*params, req])

    def _user(args):
        request = args[-1] if args else None
        return args[:-1], _session_user(request)

    def _report(exc):
        print(f"⚠️ Erreur dans « {getattr(fn, '__name__', 'action')} » : {exc}")
        traceback.print_exc()
        _log("error", detail=f"{getattr(fn, '__name__', 'action')} · {exc}")

    if inspect.isgeneratorfunction(fn):
        @functools.wraps(fn)
        def wrapper(*args):
            real, user = _user(args)
            gen = fn(*real)
            while True:
                tok = _CUR_USER.set(user)
                try:
                    item = next(gen)
                except StopIteration:
                    return
                except Exception as exc:
                    _report(exc)
                    raise
                finally:
                    _CUR_USER.reset(tok)
                yield item
    else:
        @functools.wraps(fn)
        def wrapper(*args):
            real, user = _user(args)
            tok = _CUR_USER.set(user)
            try:
                return fn(*real)
            except Exception as exc:
                _report(exc)
                raise
            finally:
                _CUR_USER.reset(tok)
    wrapper.__signature__ = new_sig
    wrapper.__annotations__ = {**getattr(fn, "__annotations__", {}), "request": gr.Request}
    return wrapper


# ---------------------------- console d'administration ----------------------------
ADMIN_USER_HEADERS = ["Utilisateur", "Rôle", "Statut", "Créé le", "Dernière connexion", "Connexions", "Cours", "Échanges", "Quiz", "Stockage"]
ADMIN_LOG_HEADERS = ["Date", "Événement", "Utilisateur", "Détail"]


def _admin_denied():
    return None if _is_admin() else "⛔ Accès réservé à l'administrateur."


def _fmt_duration(seconds):
    seconds = int(seconds)
    d, r = divmod(seconds, 86400)
    h, r = divmod(r, 3600)
    m = r // 60
    return (f"{d} j " if d else "") + f"{h} h {m:02d} min"


def _percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    low, high = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _perf_stats(log):
    """Médianes des temps de réponse mesurés (événements « chat » du journal)."""
    chats = [r.get("m") or {} for r in log if r.get("event") == "chat" and r.get("m")]
    refused = sum(1 for r in log if r.get("event") == "chat_refused")

    def col(name):
        return [float(m[name]) for m in chats if m.get(name) is not None]

    return {
        "n": len(chats), "refused": refused,
        "search": _percentile(col("search_ms"), 50), "first": _percentile(col("ttft_ms"), 50),
        "total": _percentile(col("total_ms"), 50), "total95": _percentile(col("total_ms"), 95),
    }


def _admin_overview():
    users = _users()
    now = datetime.now()
    day_ago = (now.timestamp() - 86400)

    def recent(iso):
        try:
            return datetime.fromisoformat(iso).timestamp() >= day_ago
        except Exception:
            return False

    rows, total_docs, total_size = [], 0, 0
    for key, u in sorted(users.items()):
        st = _user_stats(key)
        total_docs += st["docs"]
        total_size += st["size"]
        rows.append([
            u.get("username", key), "Administrateur" if u.get("role") == "admin" else "Utilisateur",
            "Actif" if u.get("active", True) else "Désactivé", (u.get("created_at") or "")[:16].replace("T", " "),
            (u.get("last_login") or "—")[:16].replace("T", " "), int(u.get("logins", 0)),
            st["docs"], st["chats"], st["quizzes"], _format_file_size(st["size"]),
        ])
    active24 = sum(1 for u in users.values() if recent(u.get("last_login", "")))
    log = _read_log(500)
    errors24 = sum(1 for r in log if r["event"] in ("error", "gemini_overloaded", "upload_error") and recent(r["t"]))
    kpis = [
        (len(users), "Comptes"), (active24, "Actifs (24 h)"), (total_docs, "Cours importés"),
        (_format_file_size(total_size), "Stockage utilisé"), (session_usage["requests"], "Requêtes IA"),
        (f"{session_usage['input_tokens'] + session_usage['output_tokens']:,}".replace(",", " "), "Jetons estimés"),
        (errors24, "Erreurs (24 h)"), (len(SESSIONS), "Sessions en ligne"),
    ]
    kpi_html = "<div class='admin-kpis'>" + "".join(
        f"<div class='admin-kpi{' warn' if label == 'Erreurs (24 h)' and value else ''}'><b>{_escape_html(str(value))}</b><span>{label}</span></div>"
        for value, label in kpis) + "</div>"
    perf = _perf_stats(log)
    perf_html = ""
    if perf["n"]:
        perf_html = (
            f"<div><span>Recherche (médiane)</span><b>{_fmt_ms(perf['search'])}</b></div>"
            f"<div><span>1er mot (médiane)</span><b>{_fmt_ms(perf['first'])}</b></div>"
            f"<div><span>Réponse complète (médiane)</span><b>{_fmt_ms(perf['total'])}</b></div>"
            f"<div><span>Réponse complète (95e centile)</span><b>{_fmt_ms(perf['total95'])}</b></div>"
            f"<div><span>Mesures sur</span><b>{perf['n']} réponse(s)</b></div>"
        )
    if perf["refused"]:
        perf_html += f"<div><span>Questions hors cours refusées</span><b>{perf['refused']}</b></div>"
    service = (
        "<div class='admin-service'>"
        f"<div><span>Modèle principal</span><b>{_escape_html(MODEL_NAME)}</b></div>"
        f"<div><span>Modèles de secours</span><b>{_escape_html(', '.join(_models_chain()[1:]) or '—')}</b></div>"
        f"<div><span>Clé API Gemini</span><b>{'✅ configurée' if GEMINI_API_KEY else ('❌ invalide (caractères non autorisés)' if not GEMINI_KEY_OK else '❌ manquante (GEMINI_API_KEY)')}</b></div>"
        f"<div><span>Disponibilité</span><b>{_fmt_duration(time.time() - APP_STARTED_AT)}</b></div>"
        f"<div><span>Stockage</span><b>{'✅ Google Drive' if STORAGE_ON_DRIVE else '⚠️ Temporaire (non sauvegardé)'}</b></div>"
        f"<div><span>Dossier de données</span><b>{_escape_html(DATA_DIR)}</b></div>"
        f"<div><span>Taille PDF maximale</span><b>{int(MAX_PDF_MB)} Mo</b></div>"
        f"<div><span>Modèle d'embeddings</span><b>{_escape_html(EMBED_MODEL_NAME)} ({EMBED_DEVICE})</b></div>"
        f"<div><span>Seuil de pertinence</span><b>{MIN_SIMILARITY:.2f}</b></div>"
        f"<div><span>OCR (PDF scannés)</span><b>{_escape_html(_ocr_label())}</b></div>"
        f"{perf_html}"
        "</div>"
    )
    logs = [[r["t"].replace("T", " "), r["event"], r["user"], r["detail"]] for r in log[:200]]
    return kpi_html, rows, gr.update(choices=[u.get("username", k) for k, u in sorted(users.items())]), logs, service


def _user_stats(key):
    docs = chats = quizzes = cards = 0
    root = _user_docs_dir(key)
    if os.path.isdir(root):
        for entry in os.listdir(root):
            meta = _read_json(os.path.join(root, entry, "metadata.json"), None)
            if not meta:
                continue
            docs += 1
            chats += len([h for h in meta.get("history", []) if h.get("role") == "user"])
            prog = meta.get("progress", {}) or {}
            quizzes += len(prog.get("quiz_attempts", []))
            cards += len(prog.get("flashcards", {}))
    return {"docs": docs, "chats": chats, "quizzes": quizzes, "cards": cards, "size": _dir_size(os.path.join(USERS_DIR, key))}


def _admin_pack(msg=""):
    return (*_admin_overview(), msg)


def admin_open():
    denied = _admin_denied()
    if denied:
        return (gr.update(), gr.update(), *_admin_pack(denied))
    return (gr.update(visible=False), gr.update(visible=True), *_admin_pack(""))


def admin_refresh():
    denied = _admin_denied()
    return _admin_pack(denied or "")


def _target_key(target):
    key = str(target or "").strip().lower()
    return key if key in _users() else None


def admin_toggle_active(target):
    denied = _admin_denied()
    if denied:
        return _admin_pack(denied)
    key = _target_key(target)
    if not key:
        return _admin_pack("⚠️ Sélectionnez un utilisateur.")
    if key == _CUR_USER.get():
        return _admin_pack("⚠️ Vous ne pouvez pas désactiver votre propre compte.")
    with _STORE_LOCK:
        users = _users()
        users[key]["active"] = not users[key].get("active", True)
        _write_json(USERS_FILE, users)
        state = users[key]["active"]
    if not state:
        _revoke_user_tokens(key)
        for sid in [sid for sid, k in SESSIONS.items() if k == key]:
            SESSIONS.pop(sid, None)
    _log("admin_toggle", _CUR_USER.get(), f"{key} → {'actif' if state else 'désactivé'}")
    return _admin_pack(f"✅ Compte « {key} » {'activé' if state else 'désactivé'}.")


def admin_reset_password(target, new_password):
    denied = _admin_denied()
    if denied:
        return _admin_pack(denied)
    key = _target_key(target)
    if not key:
        return _admin_pack("⚠️ Sélectionnez un utilisateur.")
    if len(str(new_password or "")) < 6:
        return _admin_pack("⚠️ Mot de passe trop court (6 caractères minimum).")
    with _STORE_LOCK:
        users = _users()
        users[key]["salt"], users[key]["hash"] = _hash_password(new_password)
        _write_json(USERS_FILE, users)
    _revoke_user_tokens(key)
    _log("admin_reset_password", _CUR_USER.get(), key)
    return _admin_pack(f"✅ Mot de passe de « {key} » réinitialisé (ses connexions mémorisées sont révoquées).")


def admin_toggle_role(target):
    denied = _admin_denied()
    if denied:
        return _admin_pack(denied)
    key = _target_key(target)
    if not key:
        return _admin_pack("⚠️ Sélectionnez un utilisateur.")
    users = _users()
    admins = [k for k, u in users.items() if u.get("role") == "admin"]
    if users[key].get("role") == "admin":
        if len(admins) <= 1:
            return _admin_pack("⚠️ Il doit rester au moins un administrateur.")
        new_role = "user"
    else:
        new_role = "admin"
    with _STORE_LOCK:
        users = _users()
        users[key]["role"] = new_role
        _write_json(USERS_FILE, users)
    _log("admin_role", _CUR_USER.get(), f"{key} → {new_role}")
    return _admin_pack(f"✅ « {key} » est maintenant {'administrateur' if new_role == 'admin' else 'utilisateur'}.")


def admin_delete_user(target, confirmed):
    denied = _admin_denied()
    if denied:
        return _admin_pack(denied)
    key = _target_key(target)
    if not key:
        return _admin_pack("⚠️ Sélectionnez un utilisateur.")
    if key == _CUR_USER.get():
        return _admin_pack("⚠️ Vous ne pouvez pas supprimer votre propre compte.")
    if not confirmed:
        return _admin_pack("⚠️ Cochez la confirmation pour supprimer définitivement ce compte et ses données.")
    with _STORE_LOCK:
        users = _users()
        users.pop(key, None)
        _write_json(USERS_FILE, users)
    _revoke_user_tokens(key)
    for sid in [sid for sid, k in SESSIONS.items() if k == key]:
        SESSIONS.pop(sid, None)
    _delete_user_data(key)
    _log("admin_delete", _CUR_USER.get(), key)
    return _admin_pack(f"✅ Compte « {key} » et toutes ses données supprimés.")


def admin_test_gemini():
    """Teste chaque modèle de la chaîne avec une toute petite requête et affiche le résultat réel."""
    denied = _admin_denied()
    if denied:
        return _admin_pack(denied)
    available = _discover_models()
    lines = []
    if available:
        lines.append("Modèles « flash » disponibles avec cette clé : " + ", ".join(available[:12]) + "\n")
    lines += ["| Modèle | Résultat | Détail |", "|---|---|---|"]
    if not GEMINI_KEY_OK:
        lines.append("| — | ❌ clé invalide | La clé contient des caractères non autorisés (accents, symboles) : collez la clé exacte AIza… |")
    elif not GEMINI_API_KEY:
        lines.append("| — | ❌ | Aucune clé Gemini configurée (GEMINI_API_KEY) |")
    else:
        for model in _models_chain():
            started = time.monotonic()
            try:
                client.models.generate_content(model=model, contents="Réponds uniquement : OK",
                                               config=types.GenerateContentConfig(max_output_tokens=8))
                lines.append(f"| {model} | ✅ répond | {(time.monotonic() - started) * 1000:.0f} ms |")
                _mark_model(model)
            except APIError as e:
                hint = {429: "quota épuisé (par minute ou par jour)", 404: "modèle inconnu ou retiré pour cette clé",
                        503: "serveur surchargé", 504: "délai dépassé (serveur lent)", 500: "erreur serveur", 400: "requête ou clé invalide",
                        401: "clé refusée", 403: "accès refusé à ce modèle"}.get(e.code, "erreur")
                lines.append(f"| {model} | ❌ {e.code} | {hint} |")
            except Exception as e:
                lines.append(f"| {model} | ❌ réseau | {type(e).__name__} |")
    _log("gemini_test")
    return _admin_pack("\n".join(lines))


def admin_backup():
    denied = _admin_denied()
    if denied:
        return None, denied
    base = os.path.join(tempfile.gettempdir(), f"akori_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    path = shutil.make_archive(base, "zip", DATA_DIR)
    _log("admin_backup", _CUR_USER.get(), os.path.basename(path))
    return path, f"✅ Sauvegarde créée ({_format_file_size(os.path.getsize(path))})."


def admin_reload_indexes():
    denied = _admin_denied()
    if denied:
        return _admin_pack(denied)
    _USER_DBS.clear()
    _log("admin_reload", _CUR_USER.get())
    return _admin_pack("✅ Index et cours rechargés depuis le disque à la prochaine action de chaque utilisateur.")


def admin_purge():
    denied = _admin_denied()
    if denied:
        return _admin_pack(denied)
    with _STORE_LOCK:
        tokens = _read_json(TOKENS_FILE, {})
        now = time.time()
        kept = {h: t for h, t in tokens.items() if t.get("exp", 0) > now}
        _write_json(TOKENS_FILE, kept)
    _LOGIN_FAILS.clear()
    _log("admin_purge", _CUR_USER.get(), f"{len(tokens) - len(kept)} jeton(s) expiré(s)")
    return _admin_pack(f"✅ {len(tokens) - len(kept)} connexion(s) mémorisée(s) expirée(s) purgée(s).")


def admin_trim_log():
    denied = _admin_denied()
    if denied:
        return _admin_pack(denied)
    try:
        with _STORE_LOCK:
            with open(ACTIVITY_LOG, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(ACTIVITY_LOG, "w", encoding="utf-8") as f:
                f.writelines(lines[-500:])
        removed = max(0, len(lines) - 500)
    except OSError:
        removed = 0
    return _admin_pack(f"✅ Journal allégé : {removed} ligne(s) ancienne(s) supprimée(s).")


theme_akori = gr.themes.Soft(primary_hue="indigo", secondary_hue="purple", neutral_hue="slate")

with gr.Blocks(title="AKORI — AI Study Assistant") as demo:
    flashcards_state = gr.State([])
    flash_index = gr.State(0)
    flash_revealed = gr.State(False)
    quiz_state = gr.State([])
    quiz_index = gr.State(0)
    quiz_validated = gr.State(False)
    quiz_answers = gr.State([])

    # ------------------------------------------------------------ vue : connexion
    token_in = gr.Textbox(elem_classes="akori-hidden", show_label=False, container=False)
    token_out = gr.Textbox(elem_classes="akori-hidden", show_label=False, container=False)

    with gr.Column(visible=True, elem_id="login-view", elem_classes="akori-dark-page") as login_view:
        gr.HTML(f"<div class='auth-brand'><img class='brand-logo' src='{LOGO_DATA_URI}' alt='AKORI'><span>AKORI</span></div>")
        with gr.Column(visible=True, elem_id="login-panel", elem_classes="auth-panel") as login_panel:
            gr.HTML("<div class='auth-title'><h1>Content de te revoir</h1><p>Connecte-toi pour accéder à tes cours</p></div>")
            li_user = gr.Textbox(label="Email", placeholder="ton@email.com", elem_id="li-user")
            li_pass = gr.Textbox(label="Mot de passe", type="password", placeholder="••••••••", elem_id="li-pass")
            li_remember = gr.State(True)
            li_btn = gr.Button("Se connecter", elem_id="li-btn")
            login_status = gr.Markdown("", elem_id="login-status", elem_classes="auth-status")
            forgot_btn = gr.Button("Mot de passe oublié ?", elem_classes="link-btn")
            with gr.Row(elem_classes="auth-foot-row"):
                gr.HTML("<span>Pas encore de compte ?</span>")
                go_register = gr.Button("Créer un compte", elem_classes="link-btn accent")
        with gr.Column(visible=False, elem_id="register-panel", elem_classes="auth-panel") as register_panel:
            gr.HTML("<div class='auth-title'><h1>Crée ton compte</h1><p>Gratuit, sans carte bancaire</p></div>")
            su_user = gr.Textbox(label="Email", placeholder="ton@email.com", elem_id="su-user")
            su_pass = gr.Textbox(label="Mot de passe", type="password", placeholder="8 caractères minimum", elem_id="su-pass")
            su_pass2 = gr.Textbox(label="Confirmer le mot de passe", type="password", placeholder="••••••••", elem_id="su-pass2")
            su_remember = gr.State(True)
            su_btn = gr.Button("Créer mon compte", elem_id="su-btn")
            register_status = gr.Markdown("", elem_id="register-status", elem_classes="auth-status")
            with gr.Row(elem_classes="auth-foot-row"):
                gr.HTML("<span>Déjà un compte ?</span>")
                go_login = gr.Button("Se connecter", elem_classes="link-btn accent")

    # ------------------------------------------------------------ vue : onboarding (4 étapes)
    ob_step = gr.State(1)
    with gr.Column(visible=False, elem_id="onboarding-view", elem_classes="akori-dark-page") as onboarding_view:
        gr.HTML(f"<div class='auth-brand'><img class='brand-logo' src='{LOGO_DATA_URI}' alt='AKORI'><span>AKORI</span></div>")
        ob_progress = gr.HTML(_ob_progress_html(1))
        with gr.Column(visible=True, elem_classes="auth-panel ob-panel") as ob1:
            gr.HTML("<div class='auth-title'><h1>Comment tu t'appelles ?</h1><p>Et quel est ton niveau scolaire ?</p></div>")
            ob_name = gr.Textbox(label="Prénom", placeholder="Ton prénom", elem_id="ob-name")
            ob_level = gr.Radio(choices=OB_LEVELS, label="Niveau", elem_id="ob-level", elem_classes="ob-cards")
        with gr.Column(visible=False, elem_classes="auth-panel ob-panel") as ob2:
            gr.HTML("<div class='auth-title'><h1>Que veux-tu faire avec AKORI ?</h1><p>Choisis ton objectif principal</p></div>")
            ob_goal = gr.Radio(choices=OB_GOALS, label="Objectif", elem_id="ob-goal", elem_classes="ob-cards")
        with gr.Column(visible=False, elem_classes="auth-panel ob-panel") as ob3:
            gr.HTML("<div class='auth-title'><h1>Dans quelle langue veux-tu travailler ?</h1><p>Tu pourras la changer à tout moment dans les paramètres</p></div>")
            ob_lang = gr.Radio(choices=LANG_CHOICES, value="Français", label="Langue", elem_id="ob-lang", elem_classes="ob-cards")
        with gr.Column(visible=False, elem_classes="auth-panel ob-panel") as ob4:
            ob_done = gr.HTML("")
        with gr.Column(elem_classes="auth-panel ob-actions"):
            ob_error = gr.Markdown("", elem_id="ob-error", elem_classes="auth-status")
            ob_next = gr.Button("Continuer", elem_id="ob-next")
            ob_enter = gr.Button("Accéder à AKORI", elem_id="ob-enter", visible=False)
            ob_back = gr.Button("Retour", elem_id="ob-back", visible=False)

    # ------------------------------------------------------------ vue : application
    with gr.Column(visible=False, elem_id="app-view") as app_view:
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=60, elem_id="sidebar"):
                menu_btn = gr.Button("☰", elem_id="sidebar-toggle")
                gr.HTML(f"<div class='brand'><img class='brand-logo' src='{LOGO_DATA_URI}' alt='AKORI'><div><div class='brand-name'>AKORI</div><div class='brand-sub'>Assistant Knowledge Organized to Revise Intelligently</div></div></div>")
                gr.Markdown("**NAVIGATION**", elem_classes="nav-title")
                nav_home = gr.Button("Accueil", elem_id="nav-home", elem_classes="navbtn")
                nav_courses = gr.Button("Mes dossiers", elem_id="nav-courses", elem_classes="navbtn")
                nav_review = gr.Button("Réviser", elem_id="nav-review", elem_classes="navbtn")
                nav_flash = gr.Button("Flashcards", elem_id="nav-flash", elem_classes="navbtn")
                nav_quiz = gr.Button("Quiz / QCM", elem_id="nav-quiz", elem_classes="navbtn")
                nav_summary = gr.Button("Résumé", elem_id="nav-summary", elem_classes="navbtn")
                nav_assistant = gr.Button("Assistant IA", elem_id="nav-assistant", elem_classes="navbtn")
                nav_progress = gr.Button("Progression", elem_id="nav-progress", elem_classes="navbtn")
                nav_reviewq = gr.Button("À revoir", elem_id="nav-reviewq", elem_classes="navbtn")
                nav_history = gr.Button("Historique", elem_id="nav-history", elem_classes="navbtn")
                nav_admin = gr.Button("Administration", elem_id="nav-admin", elem_classes="navbtn", visible=False)
                nav_settings = gr.Button("Paramètres", elem_id="nav-settings", elem_classes="navbtn")
                nav_logout = gr.Button("Déconnexion", elem_id="nav-logout", elem_classes="navbtn")
                user_badge = gr.HTML("", elem_id="user-badge")

            with gr.Column(scale=4, min_width=280, elem_id="main-column"):
                with gr.Row(elem_id="topbar"):
                    gr.Markdown("### AKORI · Espace de révision")
                    doc_selector = gr.Dropdown(label="Cours actif", choices=list(documents_db.keys()), value=(next(iter(documents_db), None)), scale=2)

                with gr.Tabs(elem_id="main-tabs") as tabs:
                    with gr.Tab("Accueil", id="home") as tab_home:
                        home_html = gr.HTML(dashboard_html(doc_selector.value))
                        start_review = gr.Button("Commencer la révision →", variant="primary")

                    with gr.Tab("Mes dossiers", id="courses") as tab_courses:
                        gr.Markdown("## Mes dossiers")
                        gr.Markdown("Tous vos supports PDF sont centralisés ici. Dès qu'un document est sélectionné, AKORI extrait son contenu et construit automatiquement son index.")
                        with gr.Row(elem_id="documents-row", equal_height=True):
                            docs_html = gr.HTML(documents_html_v16(doc_selector.value), scale=3)
                            pdf_input = gr.UploadButton(
                                "＋\nAjouter un document",
                                file_types=[".pdf"],
                                file_count="single",
                                type="filepath",
                                elem_id="add-document-tile",
                                scale=1,
                            )
                        upload_status = gr.Markdown("Sélectionnez un PDF : extraction et indexation automatiques.", elem_id="upload-status")
                        gr.Markdown("### Cours actif")
                        course_info = gr.HTML(course_overview_html(doc_selector.value))

                    with gr.Tab("Réviser", id="review") as tab_review:
                        gr.Markdown("## Réviser ce cours")
                        gr.Markdown("Une vue centrale pour accéder rapidement au résumé, aux flashcards, au quiz et à l'assistant.")
                        review_cards = gr.HTML(course_overview_html(doc_selector.value))
                        with gr.Row():
                            review_summary = gr.Button("Résumé", variant="secondary")
                            review_flash = gr.Button("Flashcards", variant="primary")
                            review_quiz = gr.Button("Quiz / QCM", variant="primary")
                            review_chat = gr.Button("Assistant IA", variant="secondary")

                    with gr.Tab("Flashcards", id="flashcards"):
                        gr.Markdown("## Flashcards — Mode Roulette")
                        with gr.Row():
                            generate_flash = gr.Button("✦ Générer les flashcards", variant="primary")
                            flash_status = gr.Markdown("", elem_id="flash-status")
                        flash_view = gr.HTML(flashcard_view([], 0, False))
                        with gr.Column(visible=False) as flash_actions:
                            flash_reveal = gr.Button("Afficher la réponse", variant="primary")
                            with gr.Row():
                                flash_known = gr.Button("✓ Je savais", variant="primary")
                                flash_review = gr.Button("↻ À revoir")

                    with gr.Tab("Quiz / QCM", id="quiz"):
                        gr.Markdown("## Quiz d'évaluation")
                        gr.Markdown("Cliquez sur une réponse, validez, puis passez à la question suivante.")
                        with gr.Row():
                            generate_quiz_btn = gr.Button("✦ Générer le quiz", variant="primary")
                            quiz_restart = gr.Button("🔁 Rejouer")
                            quiz_status = gr.Markdown("")
                        quiz_view_box = gr.HTML(quiz_view([], 0))
                        quiz_choice = gr.Radio(
                            choices=[], label="Choisissez votre réponse", container=False,
                            interactive=True, visible=False, elem_classes="quiz-radio-group",
                        )
                        with gr.Row():
                            quiz_validate = gr.Button("Valider la réponse", variant="primary", elem_id="quiz_validate")
                            quiz_next = gr.Button(NEXT_LABEL, elem_id="quiz_next")

                    with gr.Tab("Résumé", id="summary"):
                        gr.Markdown("## Résumé du cours")
                        summary_btn = gr.Button("Générer le résumé", variant="primary")
                        summary_output = gr.Markdown("Sélectionnez un cours puis lancez la génération.")

                    with gr.Tab("Assistant IA", id="assistant"):
                        gr.Markdown("## Assistant AKORI")
                        gr.Markdown("Posez une question sur le cours actif. Le moteur récupère d'abord les passages pertinents avec FAISS, puis Gemini génère la réponse à partir du contexte récupéré.")
                        chatbot = gr.Chatbot(
                            value=_chat_history_for_ui(doc_selector.value) if doc_selector.value else [],
                            height=470,
                            elem_id="assistant-chatbot",
                            render_markdown=True,
                            line_breaks=True,
                            latex_delimiters=[
                                {"left": "$$", "right": "$$", "display": True},
                                {"left": "$", "right": "$", "display": False},
                                {"left": "\\[", "right": "\\]", "display": True},
                                {"left": "\\(", "right": "\\)", "display": False},
                            ],
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(placeholder="Posez une question sur le cours…", show_label=False, scale=5, elem_id="assistant-input")
                            send_btn = gr.Button("Envoyer", variant="primary", scale=1, elem_id="assistant-send")

                    with gr.Tab("Progression", id="progress"):
                        gr.Markdown("## Ma progression")
                        gr.Markdown("Commencez par la vue globale, puis consultez le détail du cours sélectionné.")
                        global_progress_output = gr.HTML(global_progress_html(doc_selector.value))
                        gr.Markdown("### Détail du cours sélectionné")
                        progress_output = gr.HTML(progress_detail_html(doc_selector.value))

                    with gr.Tab("À revoir", id="reviewq") as tab_reviewq:
                        gr.Markdown("## À revoir")
                        gr.Markdown("Les réponses de l'assistant que vous épinglez depuis le chat sont rangées ici, cours par cours. Relisez-les, ou écoutez-les, avant l'examen.")
                        pins_box = gr.HTML(pins_view(doc_selector.value))

                    with gr.Tab("Paramètres", id="settings"):
                        gr.Markdown("## Paramètres")
                        gr.Markdown("Personnalisez AKORI. Vos choix sont mémorisés dans ce navigateur.")
                        with gr.Row(elem_classes="set-card"):
                            gr.HTML("<div class='set-text'><div class='set-title'>Mode sombre</div><div class='set-desc'>Basculez entre le thème clair et le thème sombre d'un seul clic.</div></div>")
                            theme_toggle = gr.Button(" ", elem_id="theme-toggle-btn", scale=0, min_width=84)
                        with gr.Row(elem_classes="set-card"):
                            gr.HTML("<div class='set-text'><div class='set-title'>Langue</div><div class='set-desc'>Langue de l'interface et des réponses générées par l'IA.</div></div>")
                            lang_choice = gr.Radio(choices=LANG_CHOICES, value="Français", show_label=False, container=False, elem_id="lang-choice", scale=0)

                    with gr.Tab("Historique", id="history"):
                        gr.Markdown("## Historique du cours actif")
                        history_box = gr.HTML(history_view(doc_selector.value))


    # ------------------------------------------------------------ vue : administration (une seule page)
    with gr.Column(visible=False, elem_id="admin-view") as admin_view:
        with gr.Row(elem_id="admin-top"):
            gr.HTML("<div class='admin-title'><div class='admin-badge'>🛡</div><div><h1>Console d'administration</h1><p>Suivi, maintenance et gestion des comptes AKORI</p></div></div>")
            admin_refresh_btn = gr.Button("↻ Actualiser", scale=0, min_width=130)
            admin_back = gr.Button("← Retour à l'application", variant="primary", scale=0, min_width=220)
        admin_kpis = gr.HTML()
        with gr.Tabs(elem_id="admin-tabs"):
            with gr.Tab("Utilisateurs"):
                admin_table = gr.Dataframe(headers=ADMIN_USER_HEADERS, value=[], interactive=False, wrap=True, elem_id="admin-users")
                with gr.Row():
                    admin_user_sel = gr.Dropdown(label="Utilisateur", choices=[], scale=2, elem_id="admin-user-sel")
                    admin_newpwd = gr.Textbox(label="Nouveau mot de passe", type="password", scale=2, elem_id="admin-newpwd")
                with gr.Row():
                    admin_btn_toggle = gr.Button("Activer / Désactiver")
                    admin_btn_reset = gr.Button("Réinitialiser le mot de passe")
                    admin_btn_role = gr.Button("Promouvoir / Rétrograder")
                with gr.Row():
                    admin_confirm = gr.Checkbox(label="Je confirme la suppression définitive du compte et de ses données", value=False)
                    admin_btn_delete = gr.Button("Supprimer le compte", variant="stop")
            with gr.Tab("Activité"):
                admin_log = gr.Dataframe(headers=ADMIN_LOG_HEADERS, value=[], interactive=False, wrap=True, elem_id="admin-log")
            with gr.Tab("Maintenance"):
                admin_service = gr.HTML()
                with gr.Row():
                    admin_btn_backup = gr.Button("Sauvegarde complète (ZIP)", variant="primary")
                    admin_btn_reload = gr.Button("Recharger les index")
                    admin_btn_purge = gr.Button("Purger les connexions expirées")
                    admin_btn_trim = gr.Button("Alléger le journal")
                    admin_btn_gemini = gr.Button("Tester Gemini")
                admin_file = gr.File(label="Sauvegarde", interactive=False)
        admin_msg = gr.Markdown("", elem_id="admin-msg")

    # Bouton invisible : le navigateur le déclenche seul après une coupure réseau pour réafficher l'état enregistré.
    resync_btn = gr.Button("resync", elem_id="resync-btn", elem_classes="akori-hidden")
    # Relais cachés des boutons « Épingler » / « Retirer » (affichés dans le HTML du chat, donc hors de Gradio).
    pin_payload = gr.Textbox(elem_id="pin-payload", elem_classes="akori-hidden", show_label=False, container=False)
    pin_btn = gr.Button("pin", elem_id="pin-btn", elem_classes="akori-hidden")
    unpin_payload = gr.Textbox(elem_id="unpin-payload", elem_classes="akori-hidden", show_label=False, container=False)
    unpin_btn = gr.Button("unpin", elem_id="unpin-btn", elem_classes="akori-hidden")

    # Upload : la sélection du PDF déclenche directement extraction + chunking + indexation.
    REVISION_N = 16

    def upload_and_refresh(file_path, lang=None):
        """Importe le PDF en affichant l'avancement, puis met à jour toutes les vues d'un coup."""
        file_path = getattr(file_path, "name", file_path)
        selected, message = None, ""
        for event in indexer_pdf_stream(file_path):
            if event[0] == "status":
                yield (event[1], *([gr.update()] * (9 + REVISION_N)))
            else:
                _, selected, message = event
        names = list(documents_db.keys())
        if not selected:
            yield (message, gr.update(choices=names), *([gr.update()] * (8 + REVISION_N)))
            return
        yield (
            message, gr.update(choices=names, value=selected), documents_html_v16(selected),
            dashboard_html(selected), course_overview_html(selected),
            course_overview_html(selected), global_progress_html(selected),
            progress_detail_html(selected), _chat_history_for_ui(selected),
            history_view(selected), *_revision_views(selected, lang),
        )

    # Sections de révision : leurs composants sont recalculés à chaque changement de PDF.
    REVISION_OUT = [flashcards_state, flash_index, flash_revealed, flash_view, flash_status, flash_actions, generate_flash,
                    quiz_state, quiz_index, quiz_validated, quiz_view_box, quiz_status, quiz_choice, quiz_answers, quiz_next,
                    summary_output]

    # Retour immédiat dès la sélection du PDF, puis import par étapes (extraction, indexation par lots).
    upload_event = pdf_input.upload(
        lambda: "⏳ Analyse du PDF… extraction, découpage et indexation en cours.",
        inputs=None, outputs=[upload_status], queue=False
    )
    upload_event.then(
        upload_and_refresh, inputs=[pdf_input, lang_choice],
        outputs=[upload_status, doc_selector, docs_html, home_html, course_info,
                 review_cards, global_progress_output, progress_output, chatbot, history_box] + REVISION_OUT,
        queue=True, show_progress="hidden", concurrency_limit=3
    )

    def refresh_all(d, lang=None):
        return (dashboard_html(d), documents_html_v16(d), course_overview_html(d), course_overview_html(d),
                global_progress_html(d), progress_detail_html(d), _chat_history_for_ui(d), history_view(d),
                *_revision_views(d, lang))

    doc_selector.change(refresh_all, inputs=[doc_selector, lang_choice],
                        outputs=[home_html, docs_html, course_info, review_cards, global_progress_output, progress_output,
                                 chatbot, history_box] + REVISION_OUT)

    resync_btn.click(refresh_all, inputs=[doc_selector, lang_choice],
                     outputs=[home_html, docs_html, course_info, review_cards, global_progress_output, progress_output,
                              chatbot, history_box] + REVISION_OUT, queue=False, show_progress="hidden")

    # Navigation vers les onglets
    nav_home.click(lambda: gr.Tabs(selected="home"), outputs=tabs, queue=False)
    nav_courses.click(lambda: gr.Tabs(selected="courses"), outputs=tabs, queue=False)
    nav_review.click(lambda: gr.Tabs(selected="review"), outputs=tabs, queue=False)
    nav_flash.click(lambda: gr.Tabs(selected="flashcards"), outputs=tabs, queue=False)
    nav_quiz.click(lambda: gr.Tabs(selected="quiz"), outputs=tabs, queue=False)
    nav_summary.click(lambda: gr.Tabs(selected="summary"), outputs=tabs, queue=False)
    nav_assistant.click(lambda: gr.Tabs(selected="assistant"), outputs=tabs, queue=False)
    nav_progress.click(lambda: gr.Tabs(selected="progress"), outputs=tabs, queue=False)
    nav_reviewq.click(lambda: gr.Tabs(selected="reviewq"), outputs=tabs, queue=False)
    nav_history.click(lambda: gr.Tabs(selected="history"), outputs=tabs, queue=False)
    pin_btn.click(pin_message, inputs=[pin_payload, doc_selector], outputs=[pins_box], queue=False, show_progress="hidden")
    unpin_btn.click(unpin_chapter, inputs=[unpin_payload, doc_selector], outputs=[pins_box], queue=False, show_progress="hidden")
    tab_reviewq.select(pins_view, inputs=[doc_selector], outputs=[pins_box], queue=False, show_progress="hidden")
    doc_selector.change(pins_view, inputs=[doc_selector], outputs=[pins_box], queue=False, show_progress="hidden")
    nav_settings.click(lambda: gr.Tabs(selected="settings"), outputs=tabs, queue=False)
    start_review.click(lambda: gr.Tabs(selected="review"), outputs=tabs, queue=False)
    review_flash.click(lambda: gr.Tabs(selected="flashcards"), outputs=tabs, queue=False)
    review_quiz.click(lambda: gr.Tabs(selected="quiz"), outputs=tabs, queue=False)
    review_summary.click(lambda: gr.Tabs(selected="summary"), outputs=tabs, queue=False)
    review_chat.click(lambda: gr.Tabs(selected="assistant"), outputs=tabs, queue=False)

    # Assistant conversation : étape 1 instantanée, puis étape 2 streaming.
    pending_message = gr.State("")

    send_event = send_btn.click(
        _ui_chat_user_message,
        inputs=[msg_input, chatbot, doc_selector],
        outputs=[msg_input, chatbot, pending_message],
        queue=False,
    )
    send_event.then(
        repondre_akori_chat_stream,
        inputs=[pending_message, doc_selector],
        outputs=[chatbot, pending_message],
        show_progress="hidden",
    ).then(history_view, inputs=[doc_selector], outputs=[history_box], queue=False)

    submit_event = msg_input.submit(
        _ui_chat_user_message,
        inputs=[msg_input, chatbot, doc_selector],
        outputs=[msg_input, chatbot, pending_message],
        queue=False,
    )
    submit_event.then(
        repondre_akori_chat_stream,
        inputs=[pending_message, doc_selector],
        outputs=[chatbot, pending_message],
        show_progress="hidden",
    ).then(history_view, inputs=[doc_selector], outputs=[history_box], queue=False)

    # Flashcards
    generate_flash.click(
        lambda: "⏳ Génération des flashcards…", outputs=[flash_status], queue=False,
    ).then(
        flashcard_generate_handler,
        inputs=[doc_selector, lang_choice],
        outputs=[flashcards_state, flash_index, flash_revealed, flash_view, flash_status, flash_actions, generate_flash],
        show_progress="minimal",
        concurrency_limit=1,
    )
    flash_reveal.click(
        flashcard_reveal_handler,
        inputs=[flashcards_state, flash_index, doc_selector, lang_choice],
        outputs=[flash_revealed, flash_view],
        queue=False,
    )
    flash_known.click(
        lambda c, i, d, lg: flashcard_mark_handler(c, i, "known", d, lg),
        inputs=[flashcards_state, flash_index, doc_selector, lang_choice],
        outputs=[flash_index, flash_revealed, flash_view, flash_actions, generate_flash],
        queue=False,
    ).then(lambda d: (global_progress_html(d), progress_detail_html(d)), inputs=[doc_selector], outputs=[global_progress_output, progress_output], queue=False)
    flash_review.click(
        lambda c, i, d, lg: flashcard_mark_handler(c, i, "review", d, lg),
        inputs=[flashcards_state, flash_index, doc_selector, lang_choice],
        outputs=[flash_index, flash_revealed, flash_view, flash_actions, generate_flash],
        queue=False,
    ).then(lambda d: (global_progress_html(d), progress_detail_html(d)), inputs=[doc_selector], outputs=[global_progress_output, progress_output], queue=False)

    # Quiz
    quiz_outputs = [quiz_state, quiz_index, quiz_validated, quiz_view_box, quiz_status, quiz_choice, quiz_answers, quiz_next]
    generate_quiz_btn.click(
        lambda: "⏳ Génération du quiz…", outputs=[quiz_status], queue=False
    ).then(quiz_generate_handler, inputs=[doc_selector, lang_choice], outputs=quiz_outputs, show_progress="minimal")
    quiz_restart.click(quiz_restart_handler, inputs=[quiz_state, doc_selector], outputs=quiz_outputs, queue=False)
    quiz_validate.click(
        quiz_validate_handler,
        inputs=[quiz_state, quiz_index, quiz_choice, quiz_validated, quiz_answers, doc_selector],
        outputs=[quiz_validated, quiz_view_box, quiz_status, quiz_answers, quiz_choice, quiz_next],
        queue=False,
    )
    quiz_next.click(
        quiz_next_handler,
        inputs=[quiz_state, quiz_index, quiz_validated, quiz_answers, doc_selector],
        outputs=[quiz_index, quiz_validated, quiz_view_box, quiz_status, quiz_choice, quiz_next],
        queue=False,
    ).then(lambda d: (global_progress_html(d), progress_detail_html(d)), inputs=[doc_selector], outputs=[global_progress_output, progress_output], queue=False)

    # Thème clair / sombre (mémorisé dans le navigateur)
    theme_toggle.click(None, None, None, js=THEME_TOGGLE_JS)
    lang_choice.input(None, lang_choice, None, js=LANG_APPLY_JS)
    menu_btn.click(None, None, None, js=SIDEBAR_TOGGLE_JS)
    demo.load(None, None, lang_choice, js=THEME_INIT_JS)

    # Résumé — conserve le comportement existant, mais l'affiche dans son propre espace.
    def summary_handler(d, lang=None):
        if not d or d not in documents_db:
            yield "⚠️ Sélectionnez d'abord un cours."
            return
        cache_key = _lang_code(lang)
        cached = documents_db[d].setdefault("summary_cache", {}).get(cache_key)
        if cached:
            yield cached
            return
        context = "\n\n".join(preparer_contexte_global(documents_db[d]))
        try:
            result = yield from _with_heartbeat(lambda: _summary_text(context, lang), lambda sec: f"⏳ Génération du résumé… ({sec} s)")
            documents_db[d]["summary_cache"][cache_key] = result
            _save_history(d)
            _log("summary", detail=d)
            yield result
        except Exception as e:
            _log("error", detail=f"summary · {e}")
            yield f"⚠️ {e}"
    summary_btn.click(
        lambda: "⏳ Génération du résumé…", outputs=summary_output, queue=False
    ).then(summary_handler, inputs=[doc_selector, lang_choice], outputs=summary_output)

    # ------------------------------------------------------------ connexion, déconnexion, reconnexion
    ENTER_OUT = [login_view, onboarding_view, app_view, admin_view, user_badge, nav_admin, doc_selector, home_html, docs_html,
                 course_info, review_cards, global_progress_output, progress_output, chatbot, history_box,
                 login_status, register_status, token_out, tabs, lang_choice]
    OB_OUT = [ob_step, ob1, ob2, ob3, ob4, ob_progress, ob_next, ob_enter, ob_back, ob_error, ob_done]
    OB_RESET_OUT = OB_OUT + [ob_name, ob_level, ob_goal, ob_lang]

    def _after_enter(event):
        """Après toute connexion : alignement du menu, langue du profil, onboarding remis à la première étape."""
        return (event.then(None, None, None, js=ALIGN_JS)
                     .then(None, lang_choice, None, js=LANG_APPLY_JS)
                     .then(ob_reset, None, OB_RESET_OUT, queue=False))

    _after_enter(li_btn.click(do_login, [li_user, li_pass, li_remember], ENTER_OUT))
    _after_enter(li_pass.submit(do_login, [li_user, li_pass, li_remember], ENTER_OUT))
    _after_enter(su_btn.click(do_register, [su_user, su_pass, su_pass2, su_remember], ENTER_OUT))
    _after_enter(su_pass2.submit(do_register, [su_user, su_pass, su_pass2, su_remember], ENTER_OUT))
    _after_enter(nav_logout.click(do_logout, [token_in], ENTER_OUT, js=LOGOUT_JS)).then(
        lambda: (gr.update(visible=True), gr.update(visible=False)), None, [login_panel, register_panel], queue=False)
    _after_enter(token_in.change(do_auto_login, [token_in], ENTER_OUT))
    token_out.change(None, token_out, None, js=TOKEN_STORE_JS)
    demo.load(None, None, token_in, js=GET_TOKEN_JS)

    # connexion <-> création de compte (sans aller-retour serveur lourd)
    go_register.click(lambda: (gr.update(visible=False), gr.update(visible=True), "", ""), None,
                      [login_panel, register_panel, login_status, register_status], queue=False)
    go_login.click(lambda: (gr.update(visible=True), gr.update(visible=False), "", ""), None,
                   [login_panel, register_panel, login_status, register_status], queue=False)
    forgot_btn.click(lambda: "ℹ️ Pour réinitialiser ton mot de passe, contacte l'administrateur d'AKORI.", None, login_status, queue=False)

    # onboarding
    ob_next.click(ob_next_handler, [ob_step, ob_name, ob_level, ob_goal, ob_lang], OB_OUT, queue=False)
    ob_back.click(ob_back_handler, [ob_step, ob_name], OB_OUT, queue=False)
    ob_lang.input(lambda v: v, ob_lang, lang_choice, queue=False).then(None, ob_lang, None, js=LANG_APPLY_JS)
    _after_enter(ob_enter.click(do_enter_app, None, ENTER_OUT))

    # ------------------------------------------------------------ console d'administration
    ADMIN_OUT = [admin_kpis, admin_table, admin_user_sel, admin_log, admin_service, admin_msg]
    nav_admin.click(admin_open, None, [app_view, admin_view] + ADMIN_OUT)
    admin_back.click(lambda: (gr.update(visible=True), gr.update(visible=False)), None, [app_view, admin_view]).then(None, None, None, js=ALIGN_JS)
    admin_refresh_btn.click(admin_refresh, None, ADMIN_OUT)
    admin_btn_toggle.click(admin_toggle_active, [admin_user_sel], ADMIN_OUT)
    admin_btn_reset.click(admin_reset_password, [admin_user_sel, admin_newpwd], ADMIN_OUT)
    admin_btn_role.click(admin_toggle_role, [admin_user_sel], ADMIN_OUT)
    admin_btn_delete.click(admin_delete_user, [admin_user_sel, admin_confirm], ADMIN_OUT)
    admin_btn_backup.click(admin_backup, None, [admin_file, admin_msg])
    admin_btn_reload.click(admin_reload_indexes, None, ADMIN_OUT)
    admin_btn_purge.click(admin_purge, None, ADMIN_OUT)
    admin_btn_trim.click(admin_trim_log, None, ADMIN_OUT)
    admin_btn_gemini.click(admin_test_gemini, None, ADMIN_OUT)

# Chaque action s'exécute pour l'utilisateur connecté de la session (données isolées par compte).
for _bf in list(demo.fns.values()):
    _fn = getattr(_bf, "fn", None)
    if _fn is None:
        continue
    try:
        if "request" in inspect.signature(_fn).parameters:
            continue
        _bf.fn = _scoped(_fn)
    except (TypeError, ValueError):
        continue

# =============================================================================
# PHASE « BUILD » : tout ce qui n'a pas besoin d'être refait pendant la démonstration
# =============================================================================
def build_library(pdf_dir=None, user=None, password=None, pregenerate=True, langs=("fr",)):
    """Prépare un compte pour la démonstration (à lancer une seule fois, notebook Build).

    1. indexe chaque PDF du dossier (extraction, embeddings, index FAISS) et copie le PDF sur Drive ;
    2. recalcule les anciens index (changement de modèle d'embeddings) ;
    3. pré-génère résumé, flashcards et quiz (mémorisés avec le cours) ;
    4. charge les index en mémoire pour vérifier qu'ils sont lisibles.
    Pendant la démonstration, il ne reste que la recherche et la génération des réponses du chat."""
    started = time.time()
    key = (user or os.getenv("AKORI_BUILD_USER") or "demo@akori.app").strip().lower()
    if key not in _users():
        pwd = password or os.getenv("AKORI_BUILD_PASSWORD") or secrets.token_urlsafe(9)
        ok, msg = _create_user(key, pwd, onboarded=True, allow_username=True)
        if not ok:
            raise RuntimeError(f"Compte de démonstration impossible à créer : {msg}")
        print(f"👤 Compte créé : {key} / {pwd}   (notez-le : il sert à se connecter pendant la démonstration)")
    pdf_dir = pdf_dir or os.path.join(DATA_DIR, "a_indexer")
    os.makedirs(pdf_dir, exist_ok=True)
    pdfs = sorted(os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))
    print(f"📚 Build pour {key} : {len(pdfs)} PDF dans {pdf_dir}")
    token = _CUR_USER.set(key)
    try:
        for path in pdfs:
            for event in indexer_pdf_stream(path):
                if event[0] == "done":
                    print("  ", event[2])
        for name in list(documents_db.keys()):
            entry = documents_db[name]
            getattr(entry["vector_db"], "_load", lambda: None)()   # réindexe si besoin, vérifie que l'index se charge
            if pregenerate:
                context = "\n\n".join(preparer_contexte_global(entry))
                if not entry.get("flashcards"):
                    cards, err = _flash_cards_from_context(context, langs[0])
                    if len(cards) >= MIN_FLASHCARDS:
                        entry["flashcards"], entry["flash_index"] = cards, 0
                    print(f"   🧠 {name} : {len(cards)} flashcards" + (f" ⚠️ {err}" if err else ""))
                if not (entry.get("quiz") or {}).get("questions"):
                    qs, status = _generate_quiz_from_context(context, 5, langs[0])
                    if qs:
                        entry["quiz"] = {"questions": qs, "index": 0, "answers": [None] * len(qs), "validated": False}
                    print(f"   📝 {name} : {status}")
                for lang in langs:
                    cache = entry.setdefault("summary_cache", {})
                    if not cache.get(lang):
                        try:
                            cache[lang] = _summary_text(context, lang)
                            print(f"   ≡ {name} : résumé ({lang})")
                        except Exception as e:
                            print(f"   ⚠️ {name} : résumé ({lang}) impossible : {e}")
            _save_history(name)
    finally:
        _CUR_USER.reset(token)
    _log("build", user=key, detail=f"{len(pdfs)} PDF · {time.time() - started:.0f} s")
    print(f"✅ Build terminé en {time.time() - started:.0f} s. Lancez maintenant AKORI normalement.")
    return key


def calibrate_similarity(doc_name, relevant, offtopic, user=None):
    """Aide au réglage du seuil de pertinence : similarité du meilleur passage pour des questions du cours
    et des questions hors cours. Le seuil conseillé est le milieu entre les deux groupes."""
    key = (user or os.getenv("AKORI_BUILD_USER") or "demo@akori.app").strip().lower()
    token = _CUR_USER.set(key)
    try:
        store = documents_db[doc_name]["vector_db"]

        def best(question):
            return max(1.0 - float(d) / 2.0 for _, d in store.similarity_search_with_score(question, k=RETRIEVAL_K))

        rel = [(q, best(q)) for q in relevant]
        off = [(q, best(q)) for q in offtopic]
    finally:
        _CUR_USER.reset(token)
    print(f"Modèle : {EMBED_MODEL_NAME} · seuil actuel : {MIN_SIMILARITY:.2f}\n")
    print("Questions DU cours (similarité du meilleur passage) :")
    for q, v in rel:
        print(f"  {v:.3f}  {q}")
    print("Questions HORS cours :")
    for q, v in off:
        print(f"  {v:.3f}  {q}")
    if rel and off:
        low, high = min(v for _, v in rel), max(v for _, v in off)
        if low > high:
            print(f"\n→ Seuil conseillé : {(low + high) / 2:.2f}   (AKORI_MIN_SIMILARITY={(low + high) / 2:.2f})")
        else:
            print("\n→ Les deux groupes se chevauchent : choisissez un seuil prudent (sous la plus basse question du cours).")
    return rel, off


def _prewarm_demo():
    """Démonstration : charge en mémoire les index déjà calculés, pour que la 1re question soit instantanée."""
    try:
        for user_key in list(_users().keys()):
            for entry in _db_for(user_key).values():
                if entry.get("embed_model", EMBED_MODEL_NAME) == EMBED_MODEL_NAME:
                    getattr(entry["vector_db"], "_load", lambda: None)()
        print("⚡ Index chargés en mémoire : la démonstration est prête.")
    except Exception as e:
        print(f"⚠️ Préchargement des index impossible : {e}")


def _cli_arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


_BUILD_MODE = __name__ == "__main__" and ("--build" in sys.argv or os.getenv("AKORI_MODE") == "build")

if _BUILD_MODE:
    build_library(
        pdf_dir=_cli_arg("--pdfs") or os.getenv("AKORI_PDF_DIR"),
        user=_cli_arg("--user"), password=_cli_arg("--password"),
        pregenerate="--no-pregenerate" not in sys.argv,
        langs=tuple(x.strip() for x in os.getenv("AKORI_BUILD_LANGS", "fr,en").split(",") if x.strip()),
    )
    if "--build" in sys.argv:
        sys.exit(0)
elif __name__ == "__main__":
    threading.Thread(target=_prewarm_demo, daemon=True).start()
    demo.queue(default_concurrency_limit=4, max_size=64).launch(
        share=True,
        debug=False,
        show_error=True,
        head='<meta name="google" content="notranslate"><meta http-equiv="Content-Language" content="fr"><script>document.documentElement.setAttribute("translate","no");document.documentElement.lang="fr";</script>',
        theme=theme_akori,
        css=custom_css,
        js=AKORI_NAV_JS,
        favicon_path=_logo_favicon_path(),
    )
