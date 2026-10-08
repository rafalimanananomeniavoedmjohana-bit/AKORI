import contextvars
import traceback
import secrets
import hashlib
import hmac
import inspect
import tempfile
import html
import shutil
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GEMINI_HTTP_TIMEOUT_MS = int(os.getenv("GEMINI_HTTP_TIMEOUT_MS", "25000"))   # durée max d'un appel
GEMINI_BUDGET_S = float(os.getenv("GEMINI_BUDGET_S", "50"))                 # durée max d'une génération (tous essais)
try:
    client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=GEMINI_HTTP_TIMEOUT_MS))
except Exception:
    client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3.5-flash-lite"
FALLBACK_MODEL_NAME = "gemini-3.1-flash-lite"  # Secours gratuit, si accessible au projet.
EXTRA_FALLBACK_MODELS = [
    m.strip() for m in os.getenv("GEMINI_EXTRA_FALLBACKS", "gemini-2.5-flash-lite").split(",") if m.strip()
]


def _models_chain():
  """Modèle principal puis modèles de secours, sans doublon."""
  chain = []
  for name in [MODEL_NAME, FALLBACK_MODEL_NAME, *EXTRA_FALLBACK_MODELS]:
    if name and name not in chain:
      chain.append(name)
  return chain

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


embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 64},
)


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


def _is_global_query(query):
    return (
        _is_title_query(query)
        or _is_summary_query(query)
        or _is_technology_query(query)
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
DATA_DIR = os.getenv(
    "AKORI_DATA_DIR", os.path.join(os.getcwd(), "akori_data_store")
)
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


def _log(event, user=None, detail=""):
  """Journal d'activité (une ligne JSON par événement) pour le suivi et la maintenance."""
  rec = {
      "t": datetime.now().isoformat(timespec="seconds"),
      "event": event,
      "user": user if user is not None else (_CUR_USER.get() or ""),
      "detail": str(detail)[:300],
  }
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


def _hash_password(password, salt=None):
  salt = salt or secrets.token_hex(16)
  digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 200_000).hex()
  return salt, digest


def _users():
  return _read_json(USERS_FILE, {})


def _create_user(username, password, role="user"):
  """Retourne (ok, message). Les messages sont en français : l'interface les traduit."""
  username = str(username or "").strip()
  password = str(password or "")
  if not USERNAME_RE.match(username):
    return False, "⚠️ Nom d'utilisateur invalide (3 à 32 caractères : lettres, chiffres, . _ -)."
  if len(password) < 6:
    return False, "⚠️ Mot de passe trop court (6 caractères minimum)."
  key = username.lower()
  with _STORE_LOCK:
    users = _users()
    if key in users:
      return False, "⚠️ Ce nom d'utilisateur existe déjà."
    salt, digest = _hash_password(password)
    users[key] = {
        "username": username, "salt": salt, "hash": digest, "role": role, "active": True,
        "created_at": datetime.now().isoformat(timespec="seconds"), "last_login": "", "logins": 0,
    }
    _write_json(USERS_FILE, users)
  os.makedirs(_user_docs_dir(key), exist_ok=True)
  _log("register", key, f"rôle={role}")
  return True, "✅ Compte créé."


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


def _bootstrap_admin():
  """Crée le compte administrateur au premier lancement et migre l'ancien stockage."""
  users = _users()
  if not any(u.get("role") == "admin" for u in users.values()):
    name = os.getenv("AKORI_ADMIN_USER", "admin")
    password = os.getenv("AKORI_ADMIN_PASSWORD")
    generated = not password
    if generated:
      password = secrets.token_urlsafe(9)
    ok, msg = _create_user(name, password, role="admin")
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
  # Anciens cours (stockage unique) -> rattachés à l'administrateur, rien n'est perdu.
  if admin_key and os.path.isdir(LEGACY_DOCUMENTS_DIR) and os.listdir(LEGACY_DOCUMENTS_DIR):
    target = _user_docs_dir(admin_key)
    os.makedirs(target, exist_ok=True)
    for entry in os.listdir(LEGACY_DOCUMENTS_DIR):
      dst = os.path.join(target, entry)
      if not os.path.exists(dst):
        shutil.move(os.path.join(LEGACY_DOCUMENTS_DIR, entry), dst)
    print(f"📦 Anciens documents migrés vers le compte « {admin_key} ».")


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
  """Index FAISS d'un cours chargé seulement à la première utilisation (recherche, résumé, quiz…)."""

  def __init__(self, path):
    self._path = path
    self._vs = None
    self._lock = threading.Lock()

  def _load(self):
    with self._lock:
      if self._vs is None:
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
      vector_db = _LazyVectorStore(index_path)
      docs[filename] = {
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
      }
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
    if previous and previous.get("content_hash") == content_hash:
      # Même fichier déjà indexé : on réutilise tout, c'est instantané.
      yield ("done", filename, f"✅ '{filename}' est déjà indexé : réutilisé instantanément.")
      return

    yield ("status", "⏳ Extraction du texte…")
    with fitz.open(file_path) as doc:
      texte_complet = "".join(page.get_text() for page in doc)
    if not texte_complet.strip():
      yield ("done", None, "⚠️ Aucun texte exploitable dans ce PDF (document scanné ?).")
      return

    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80).split_text(texte_complet)
    total = len(chunks)
    yield ("status", f"⏳ Indexation 0/{total}…")

    vectors = []
    batch = 64
    for i in range(0, total, batch):
      vectors.extend(embed_model.embed_documents(chunks[i:i + batch]))
      yield ("status", f"⏳ Indexation {min(i + batch, total)}/{total}…")
    vector_db = FAISS.from_embeddings(list(zip(chunks, vectors)), embed_model)

    same_name = previous is not None
    documents_db[filename] = {
        "vector_db": vector_db,
        "history": [],
        "progress": _default_progress(),
        "chunks_count": total,
        "full_text": "",
        "title": _extract_document_title(texte_complet),
        "file_size": size,
        "content_hash": content_hash,
        "added_at": (previous or {}).get("added_at") if same_name and (previous or {}).get("added_at") else datetime.now().isoformat(timespec="seconds"),
    }
    yield ("status", "⏳ Enregistrement…")
    _save_document(filename)
    _log("upload", detail=f"{filename} · {total} fragments · {size // 1024} Ko")
    yield ("done", filename, f"✅ '{filename}' indexé avec succès ({total} fragments).")
  except PermissionError:
    yield ("done", None, "⚠️ Session expirée : rechargez la page pour vous reconnecter.")
  except Exception as e:
    _log("upload_error", detail=f"{filename} · {e}")
    yield ("done", None, f"❌ Erreur lors de l'indexation : {str(e)}")


UI_CHAT_LIMIT = 30      # messages envoyés au navigateur (le reste reste archivé)
UI_HISTORY_LIMIT = 80   # messages affichés dans l'onglet Historique
HISTORY_KEEP = 300      # messages conservés par cours


def _chat_history_for_ui(doc_name):
  data = documents_db.get(doc_name, {})
  raw = data.get("history", [])
  result = []
  for item in raw[-UI_CHAT_LIMIT:]:
    if isinstance(item, dict) and item.get("role") and item.get("content"):
      result.append({"role": item["role"], "content": str(item["content"])})
  return result


def preparer_contexte_global(doc_data):
  vector_db = doc_data["vector_db"]
  all_docs = list(vector_db.docstore._dict.values())
  all_chunks = [
      doc.page_content.strip()
      for doc in all_docs
      if getattr(doc, "page_content", "").strip()
  ]

  if not all_chunks:
    return []

  MAX_GLOBAL_CHUNKS = 12
  if len(all_chunks) <= MAX_GLOBAL_CHUNKS:
    return all_chunks

  step = (len(all_chunks) - 1) / float(MAX_GLOBAL_CHUNKS - 1)
  indices = [round(i * step) for i in range(MAX_GLOBAL_CHUNKS)]
  return [all_chunks[i] for i in indices if i < len(all_chunks)]


# --- 4. PIPELINE CHAT & GEMINI ---
TOKEN_SESSION_LIMIT = 50000
session_usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}


def _estimate_tokens(text_value):
  return max(1, len(str(text_value or "")) // 4)


def _stream_gemini(user_prompt, system_instruction, throttle=False):
  """Flux de texte Gemini. Réessaie / bascule de modèle tant qu'aucun mot n'a été reçu."""
  last_error = None
  deadline = time.monotonic() + GEMINI_BUDGET_S
  for model_target in _models_chain():
    for attempt in range(2):
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
            started = True
            yield text
        return
      except APIError as e:
        last_error = e
        if started:
          raise
        if e.code in (429, 500, 503, 504) and attempt == 0:
          time.sleep(0.8 + random.uniform(0.0, 0.4))
          continue
        if e.code in (404, 429, 500, 503, 504):
          break
        raise
      except (PermissionError, GeneratorExit, KeyboardInterrupt):
        raise
      except Exception as e:  # délai dépassé, coupure réseau…
        last_error = e
        if started:
          raise
        print(f"⚠️ {model_target} : {type(e).__name__} ({e}). Nouvel essai.")
        time.sleep(0.6)
  if time.monotonic() > deadline:
    _log("gemini_timeout")
    raise RuntimeError("Gemini met trop de temps à répondre. Réessayez dans un instant.")
  _log("gemini_overloaded", detail=getattr(last_error, "code", ""))
  raise RuntimeError("Gemini est très sollicité en ce moment. Réessayez dans une dizaine de secondes.")


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
    # Requête locale : pas d'appel Gemini pour une question de titre simple.
    if _is_title_query(message_str):
      title = doc_data.get("title", "").strip()
      english = _normalize_query(message_str) in {"title", "the title", "what is the title", "what s the title"}
      if title:
        historique[-1]["content"] = (
            f"The title of the document is: **{title}**" if english
            else f"Le titre du document est : **{title}**"
        )
      else:
        historique[-1]["content"] = (
            "⚠️ The document title is unavailable." if english
            else "⚠️ Le titre du document est indisponible."
        )
      _save_history(doc_selectionne)
      yield _chat_history_for_ui(doc_selectionne), ""
      return

    if _is_global_query(message_str):
      docs_pertinents = preparer_contexte_global(doc_data)
    else:
      docs_and_scores = vector_db.similarity_search_with_score(
          message_str, k=4
      )
      docs_pertinents = [doc.page_content for doc, score in docs_and_scores]

    if not docs_pertinents:
      historique[-1]["content"] = (
          "⚠️ Aucune information pertinente n'a pu être extraite du document."
      )
      _save_history(doc_selectionne)
      yield _chat_history_for_ui(doc_selectionne), ""
      return

    contexte_brut = "\n\n".join(docs_pertinents)
    system_instruction = (
        "Tu es l'assistant académique AKORI. Réponds de façon claire et précise "
        "en t'appuyant STRICTEMENT sur le contexte fourni. "
        "Si une information n'est pas présente dans le contexte, indique-le "
        "plutôt que de l'inventer. "
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
    last_ui_update = time.monotonic()
    MIN_STREAM_CHARS = 10
    MAX_STREAM_DELAY = 0.05

    for texte_chunk in _stream_gemini(user_prompt, system_instruction):
      reponse += texte_chunk
      buffer += texte_chunk
      now = time.monotonic()

      if len(buffer) >= MIN_STREAM_CHARS or (now - last_ui_update) >= MAX_STREAM_DELAY:
        historique[-1]["content"] = reponse
        buffer = ""
        last_ui_update = now
        # L'historique précédent reste affiché ; seul le dernier message évolue.
        yield _chat_history_for_ui(doc_selectionne), ""

    # Affiche immédiatement le dernier fragment restant.
    historique[-1]["content"] = reponse or "…"
    session_usage["output_tokens"] += _estimate_tokens(reponse)
    session_usage["requests"] += 1
    _save_history(doc_selectionne)
    _log("chat", detail=f"{doc_selectionne} · {len(reponse)} car.")

    # Dernier état garanti après la fin du flux.
    yield _chat_history_for_ui(doc_selectionne), ""

  except Exception as e:
    _log("error", detail=f"chat · {e}")
    historique[-1]["content"] = f"⚠️ {str(e)}"
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
    cards = "".join(_course_card_html(n, d, n == active_doc) for n, d in documents_db.items())
    if not cards:
        cards = "<div class='empty-state'>📚 Aucun cours pour le moment.<br><span>Importez votre premier PDF dans 'Mes dossiers' pour commencer.</span></div>"

    return f"""
    <div class='hero'>
      <div>
        <div class='eyebrow'>ASSISTANT KNOWLEDGE ORGANIZED TO REVISE INTELLIGENTLY</div>
        <h1>Bonjour 👋<br><span>Prêt à booster votre révision ?</span></h1>
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
    """Génération JSON résiliente.

    Chaque modèle est réessayé avec une courte attente progressive en cas de
    surcharge (503/500) ou de limite (429), puis on bascule sur le modèle de
    secours suivant. L'erreur n'est montrée qu'une fois tous les essais épuisés.
    """
    attempts_per_model = 3
    last_error = None
    deadline = time.monotonic() + GEMINI_BUDGET_S
    for model_target in _models_chain():
        for attempt in range(attempts_per_model):
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
                return (response.text or "").strip()
            except APIError as e:
                last_error = e
                if e.code not in (429, 500, 503, 504):
                    # Modèle inconnu (404) : on tente directement le suivant ; autre erreur : on remonte.
                    if e.code == 404:
                        break
                    raise
                if attempt < attempts_per_model - 1:
                    time.sleep(1.2 * (2 ** attempt) + random.uniform(0.1, 0.5))
                else:
                    print(f"⚠️ {model_target} indisponible ({e.code}). Passage au modèle suivant.")
            except (PermissionError, KeyboardInterrupt):
                raise
            except Exception as e:  # délai dépassé, coupure réseau… : on réessaie / on change de modèle
                last_error = e
                print(f"⚠️ {model_target} : {type(e).__name__} ({e}). Nouvel essai.")
                time.sleep(0.8)
    if time.monotonic() > deadline:
        _log("gemini_timeout")
        raise RuntimeError("Gemini met trop de temps à répondre. Réessayez dans un instant.")

    if last_error is not None and getattr(last_error, "code", None) == 429:
        raise RuntimeError(
            "Quota ou limite temporaire Gemini atteinte. Attendez un peu avant de relancer la génération."
        ) from last_error
    _log("gemini_overloaded", detail=getattr(last_error, "code", ""))
    raise RuntimeError(
        "Gemini est très sollicité en ce moment. Réessayez dans une dizaine de secondes."
    ) from last_error


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

    def work():
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

    cards, last_error = yield from _with_heartbeat(
        work, lambda sec: (gr.update(),) * 4 + (f"⏳ Génération des flashcards… ({sec} s)", gr.update(), gr.update()))

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
#sidebar .nav-title { color:#9aa4b7!important;font-size:10px;font-weight:800;letter-spacing:.1em;margin:10px 5px 8px; }

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
  display:flex!important;flex-direction:column!important;gap:2px!important;--layout-gap:2px;
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
  width:100%!important;min-height:40px!important;margin:1px 0!important;padding:0 12px!important;
  font-size:13px!important;white-space:nowrap!important;overflow:hidden!important;text-overflow:clip!important;flex:none!important;
  transition:background-color .18s ease, color .18s ease, border-color .18s ease, transform .18s ease!important;
}
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
  const I18N = [["NAVIGATION", "NAVIGATION"], ["Accueil", "Home"], ["Mes dossiers", "My folders"], ["Réviser", "Revise"], ["Flashcards", "Flashcards"], ["Quiz / QCM", "Quiz / MCQ"], ["Résumé", "Summary"], ["Assistant IA", "AI Assistant"], ["Progression", "Progress"], ["À revoir", "To review"], ["Historique", "History"], ["Paramètres", "Settings"], ["Administration", "Administration"], ["Déconnexion", "Log out"], ["Administrateur", "Administrator"], ["Utilisateur", "User"], ["AKORI · Espace de révision", "AKORI · Revision space"], ["Cours actif", "Active course"], ["Bonjour 👋", "Hello 👋"], ["Prêt à booster votre révision ?", "Ready to boost your revision?"], ["AKORI transforme vos cours PDF en un espace de révision intelligent : résumé, flashcards roulette, quiz et assistant RAG.", "AKORI turns your PDF courses into a smart revision space: summary, roulette flashcards, quiz and RAG assistant."], ["💡 Guide de révision AKORI & mode roulette flashcards", "💡 AKORI revision guide & flashcards roulette mode"], ["1. Chargez vos dossiers", "1. Load your folders"], ["Glissez vos PDF dans Mes dossiers pour activer l'indexation FAISS.", "Drop your PDFs in My folders to enable FAISS indexing."], ["2. Lancement roulette", "2. Start the roulette"], ["Générez au moins 7 flashcards. Les questions défilent automatiquement en boucle.", "Generate at least 7 flashcards. Questions scroll automatically in a loop."], ["3. Clic & réponse effacée", "3. Click & answer hidden"], ["Cliquez sur la carte pour stopper/relancer. La réponse s'efface à chaque relance.", "Click the card to stop/restart. The answer is hidden at each restart."], ["Cours importés", "Imported courses"], ["Fragments indexés", "Indexed fragments"], ["Échanges du cours", "Course exchanges"], ["Recherche active", "Search active"], ["Réviser un cours en un clic", "Revise a course in one click"], ["Générez les outils principaux à partir du cours actif.", "Generate the main tools from the active course."], ["Résumé · Flashcards · Quiz / QCM · Assistant IA", "Summary · Flashcards · Quiz / MCQ · AI Assistant"], ["Mes cours", "My courses"], ["Vos supports de révision indexés localement.", "Your revision materials indexed locally."], ["Aucun cours sélectionné", "No course selected"], ["Commencer la révision →", "Start revising →"], ["Tous vos supports PDF sont centralisés ici. Dès qu'un document est sélectionné, AKORI extrait son contenu et construit automatiquement son index.", "All your PDF materials are gathered here. As soon as a document is selected, AKORI extracts its content and builds its index automatically."], ["＋\nAjouter un document", "＋\nAdd a document"], ["Sélectionnez un PDF : extraction et indexation automatiques.", "Select a PDF: automatic extraction and indexing."], ["Indexé", "Indexed"], ["Sélectionnez un cours pour afficher son espace de révision.", "Select a course to display its revision space."], ["Date inconnue", "Unknown date"], ["Général", "General"], ["📁 Aucun document pour le moment.", "📁 No document yet."], ["Ajoutez votre premier PDF pour commencer.", "Add your first PDF to get started."], ["📚 Aucun cours pour le moment.", "📚 No course yet."], ["Importez votre premier PDF dans 'Mes dossiers' pour commencer.", "Import your first PDF in 'My folders' to get started."], ["Aucun document chargé.", "No document loaded."], ["Réviser ce cours", "Revise this course"], ["Une vue centrale pour accéder rapidement au résumé, aux flashcards, au quiz et à l'assistant.", "A central view to quickly reach the summary, flashcards, quiz and assistant."], ["Flashcards — Mode Roulette", "Flashcards — Roulette mode"], ["✦ Générer les flashcards", "✦ Generate flashcards"], ["🔄 Régénérer (7 minimum)", "🔄 Regenerate (7 minimum)"], ["Afficher la réponse", "Show the answer"], ["✓ Je savais", "✓ I knew it"], ["↻ À revoir", "↻ To review"], ["Aucune flashcard générée.", "No flashcards generated."], ["Choisissez un cours puis cliquez sur « Générer les flashcards ».", "Choose a course then click “Generate flashcards”."], ["SESSION TERMINÉE", "SESSION COMPLETED"], ["Maîtrise", "Mastery"], ["Maîtrisées", "Mastered"], ["Cliquez sur « 🔄 Régénérer (7 minimum) » pour obtenir une nouvelle série.", "Click “🔄 Regenerate (7 minimum)” to get a new set."], ["Quiz d'évaluation", "Evaluation quiz"], ["Cliquez sur une réponse, validez, puis passez à la question suivante.", "Click an answer, validate, then go to the next question."], ["✦ Générer le quiz", "✦ Generate the quiz"], ["🔁 Rejouer", "🔁 Replay"], ["Valider la réponse", "Validate the answer"], ["Question suivante →", "Next question →"], ["Voir le résultat 🎯", "See the result 🎯"], ["Choisissez votre réponse", "Choose your answer"], ["Aucun quiz généré.", "No quiz generated."], ["Choisissez un cours puis cliquez sur « Générer le quiz ».", "Choose a course then click “Generate the quiz”."], ["QUIZ TERMINÉ", "QUIZ COMPLETED"], ["Votre résultat a été enregistré dans la progression de ce cours.", "Your result has been saved in this course's progress."], ["🏆 Sans faute, bravo !", "🏆 Perfect score, well done!"], ["✅ Bonne réponse !", "✅ Correct answer!"], ["❌ Réponse incorrecte", "❌ Incorrect answer"], ["Explication :", "Explanation:"], ["Résumé du cours", "Course summary"], ["Générer le résumé", "Generate the summary"], ["Sélectionnez un cours puis lancez la génération.", "Select a course then start the generation."], ["Assistant AKORI", "AKORI Assistant"], ["Posez une question sur le cours actif. Le moteur récupère d'abord les passages pertinents avec FAISS, puis Gemini génère la réponse à partir du contexte récupéré.", "Ask a question about the active course. The engine first retrieves the relevant passages with FAISS, then Gemini generates the answer from the retrieved context."], ["Envoyer", "Send"], ["Posez une question sur le cours…", "Ask a question about the course…"], ["Ma progression", "My progress"], ["Commencez par la vue globale, puis consultez le détail du cours sélectionné.", "Start with the global view, then check the details of the selected course."], ["Détail du cours sélectionné", "Selected course details"], ["Votre progression commencera ici.", "Your progress will start here."], ["Importez un cours pour créer votre premier suivi. Tant qu'aucun cours n'est chargé, la progression reste à 0 %.", "Import a course to create your first tracking. As long as no course is loaded, progress stays at 0%."], ["Aucune progression à afficher.", "No progress to display."], ["Importez un cours puis utilisez les flashcards ou terminez un quiz pour commencer à construire votre progression.", "Import a course then use the flashcards or finish a quiz to start building your progress."], ["VUE D'ENSEMBLE", "OVERVIEW"], ["Progression globale", "Global progress"], ["Synthèse de tous vos cours réellement importés et de vos interactions de révision.", "Summary of all your imported courses and your revision interactions."], ["Cours suivis", "Tracked courses"], ["Cours commencés", "Started courses"], ["Flashcards maîtrisées", "Mastered flashcards"], ["Moyenne des quiz", "Quiz average"], ["Progression par cours", "Progress by course"], ["Le détail pédagogique reste lié au cours sélectionné : points forts, points faibles, flashcards et résultats des quiz.", "The detailed breakdown stays linked to the selected course: strengths, weaknesses, flashcards and quiz results."], ["Calculée uniquement à partir des interactions enregistrées sur ce cours.", "Computed only from the interactions recorded on this course."], ["Quiz terminés", "Completed quizzes"], ["✦ Points forts", "✦ Strengths"], ["↗ Points faibles", "↗ Weaknesses"], ["Pas encore assez de réponses de quiz pour identifier un point fort.", "Not enough quiz answers yet to identify a strength."], ["Pas encore assez de réponses de quiz pour identifier un point faible.", "Not enough quiz answers yet to identify a weakness."], ["Aucun point fort clairement établi pour le moment.", "No clear strength established yet."], ["Aucun point faible clairement établi pour le moment.", "No clear weakness established yet."], ["Cette section regroupera les flashcards marquées « À revoir » et les erreurs de quiz.", "This section will gather the flashcards marked “To review” and the quiz mistakes."], ["📌 Votre file « À revoir » apparaîtra ici après les interactions.", "📌 Your “To review” queue will appear here after some interactions."], ["Historique du cours actif", "Active course history"], ["Aucun historique pour le moment.", "No history yet."], ["Aucune conversation enregistrée.", "No conversation recorded."], ["Vous", "You"], ["Personnalisez AKORI. Vos choix sont mémorisés dans ce navigateur.", "Customize AKORI. Your choices are saved in this browser."], ["Mode sombre", "Dark mode"], ["Basculez entre le thème clair et le thème sombre d'un seul clic.", "Switch between the light and dark theme with one click."], ["Langue", "Language"], ["Langue de l'interface et des réponses générées par l'IA.", "Language of the interface and of the generated summaries, flashcards and quizzes. In the chat, AKORI answers in the language you write in."], ["Connexion", "Sign in"], ["Créer un compte", "Create an account"], ["Nom d'utilisateur", "Username"], ["Mot de passe", "Password"], ["Confirmer le mot de passe", "Confirm the password"], ["Rester connecté", "Stay signed in"], ["Se connecter", "Sign in"], ["Créer mon compte", "Create my account"], ["Bienvenue sur AKORI", "Welcome to AKORI"], ["Connectez-vous pour retrouver vos cours, flashcards, quiz et historiques archivés.", "Sign in to find your archived courses, flashcards, quizzes and history."], ["Vos données sont enregistrées dans votre compte et restent disponibles à chaque connexion.", "Your data is saved in your account and available every time you sign in."], ["⚠️ Nom d'utilisateur invalide (3 à 32 caractères : lettres, chiffres, . _ -).", "⚠️ Invalid username (3 to 32 characters: letters, digits, . _ -)."], ["⚠️ Mot de passe trop court (6 caractères minimum).", "⚠️ Password too short (6 characters minimum)."], ["⚠️ Ce nom d'utilisateur existe déjà.", "⚠️ This username already exists."], ["✅ Compte créé.", "✅ Account created."], ["⚠️ Identifiants incorrects.", "⚠️ Incorrect username or password."], ["⛔ Ce compte est désactivé. Contactez l'administrateur.", "⛔ This account is disabled. Contact the administrator."], ["⏳ Trop de tentatives. Réessayez dans une minute.", "⏳ Too many attempts. Try again in a minute."], ["⚠️ Les mots de passe ne correspondent pas.", "⚠️ The passwords do not match."], ["✅ Vous êtes déconnecté.", "✅ You are signed out."], ["Connexion requise.", "Sign-in required."], ["⏳ Analyse du PDF… extraction, découpage et indexation en cours.", "⏳ Analyzing the PDF… extraction, splitting and indexing in progress."], ["⏳ Extraction du texte…", "⏳ Extracting the text…"], ["⏳ Enregistrement…", "⏳ Saving…"], ["⚠️ Aucun fichier sélectionné.", "⚠️ No file selected."], ["⚠️ Aucun texte exploitable dans ce PDF (document scanné ?).", "⚠️ No usable text in this PDF (scanned document?)."], ["⚠️ Veuillez d'abord sélectionner un cours.", "⚠️ Please select a course first."], ["⏳ Génération…", "⏳ Generating…"], ["⚠️ Le titre du document est indisponible.", "⚠️ The document title is unavailable."], ["⚠️ Aucune information pertinente n'a pu être extraite du document.", "⚠️ No relevant information could be extracted from the document."], ["Gemini est très sollicité en ce moment. Réessayez dans une dizaine de secondes.", "Gemini is very busy right now. Try again in about ten seconds."], ["Quota ou limite temporaire Gemini atteinte. Attendez un peu avant de relancer la génération.", "Gemini quota or temporary limit reached. Wait a little before generating again."], ["⚠️ Gemini est temporairement indisponible. Réessayez dans quelques secondes.", "⚠️ Gemini is temporarily unavailable. Try again in a few seconds."], ["⚠️ La limite temporaire de Gemini a été atteinte. Attendez un peu puis réessayez.", "⚠️ Gemini's temporary limit was reached. Wait a little then try again."], ["⚠️ Impossible de générer les flashcards à partir du contexte.", "⚠️ Unable to generate flashcards from the context."], ["⚠️ Impossible de générer le quiz.", "⚠️ Unable to generate the quiz."], ["⚠️ Sélectionnez d'abord un cours.", "⚠️ Select a course first."], ["⚠️ Aucun document sélectionné.", "⚠️ No document selected."], ["❌ Document vide.", "❌ Empty document."], ["❌ Le modèle n'a renvoyé aucune flashcard valide.", "❌ The model returned no valid flashcard."], ["⏳ Génération des flashcards…", "⏳ Generating flashcards…"], ["⏳ Génération du quiz…", "⏳ Generating the quiz…"], ["⏳ Génération du résumé…", "⏳ Generating the summary…"], ["🔁 Quiz relancé.", "🔁 Quiz restarted."], ["⚠️ Générez d'abord un quiz.", "⚠️ Generate a quiz first."], ["ℹ️ Réponse déjà validée : passez à la suite.", "ℹ️ Answer already validated: move on."], ["⚠️ Sélectionnez une réponse avant de valider.", "⚠️ Select an answer before validating."], ["✅ Bonne réponse.", "✅ Correct answer."], ["❌ Réponse incorrecte. Consultez l'explication.", "❌ Incorrect answer. Check the explanation."], ["⚠️ Validez d'abord cette réponse.", "⚠️ Validate this answer first."], ["Console d'administration", "Administration console"], ["Suivi, maintenance et gestion des comptes AKORI", "Monitoring, maintenance and account management for AKORI"], ["← Retour à l'application", "← Back to the app"], ["↻ Actualiser", "↻ Refresh"], ["Utilisateurs", "Users"], ["Activité", "Activity"], ["Maintenance", "Maintenance"], ["Nouveau mot de passe", "New password"], ["Activer / Désactiver", "Enable / Disable"], ["Réinitialiser le mot de passe", "Reset the password"], ["Promouvoir / Rétrograder", "Promote / Demote"], ["Supprimer le compte", "Delete the account"], ["Je confirme la suppression définitive du compte et de ses données", "I confirm the permanent deletion of the account and its data"], ["Sauvegarde complète (ZIP)", "Full backup (ZIP)"], ["Recharger les index", "Reload the indexes"], ["Purger les connexions expirées", "Purge expired sign-ins"], ["Alléger le journal", "Trim the log"], ["Sauvegarde", "Backup"], ["Comptes", "Accounts"], ["Actifs (24 h)", "Active (24 h)"], ["Stockage utilisé", "Storage used"], ["Requêtes IA", "AI requests"], ["Jetons estimés", "Estimated tokens"], ["Erreurs (24 h)", "Errors (24 h)"], ["Sessions en ligne", "Online sessions"], ["Modèle principal", "Main model"], ["Modèles de secours", "Fallback models"], ["Clé API Gemini", "Gemini API key"], ["Disponibilité", "Uptime"], ["Dossier de données", "Data folder"], ["Taille PDF maximale", "Maximum PDF size"], ["✅ configurée", "✅ configured"], ["⛔ Accès réservé à l'administrateur.", "⛔ Reserved for the administrator."], ["⚠️ Sélectionnez un utilisateur.", "⚠️ Select a user."], ["⚠️ Vous ne pouvez pas désactiver votre propre compte.", "⚠️ You cannot disable your own account."], ["⚠️ Vous ne pouvez pas supprimer votre propre compte.", "⚠️ You cannot delete your own account."], ["⚠️ Il doit rester au moins un administrateur.", "⚠️ At least one administrator must remain."], ["⚠️ Cochez la confirmation pour supprimer définitivement ce compte et ses données.", "⚠️ Tick the confirmation to permanently delete this account and its data."], ["Rôle", "Role"], ["Statut", "Status"], ["Créé le", "Created on"], ["Dernière connexion", "Last sign-in"], ["Connexions", "Sign-ins"], ["Cours", "Courses"], ["Échanges", "Exchanges"], ["Quiz", "Quizzes"], ["Stockage", "Storage"], ["Actif", "Active"], ["Désactivé", "Disabled"], ["Date", "Date"], ["Événement", "Event"], ["Détail", "Detail"], ["⚠️ Session expirée : rechargez la page pour vous reconnecter.", "⚠️ Session expired: reload the page to sign in again."], ["Gemini met trop de temps à répondre. Réessayez dans un instant.", "Gemini is taking too long to answer. Try again in a moment."]];
  const PATTERNS = [["^✅ '(.+)' indexé avec succès \\((\\d+) fragments\\)\\.$", "✅ '$1' indexed successfully ($2 fragments)."], ["^✅ '(.+)' est déjà indexé : réutilisé instantanément\\.$", "✅ '$1' is already indexed: reused instantly."], ["^❌ Erreur lors de l'indexation : (.*)$", "❌ Indexing error: $1"], ["^⏳ Indexation (\\d+)\\/(\\d+)…$", "⏳ Indexing $1/$2…"], ["^⚠️ Fichier trop volumineux \\(maximum (\\d+) Mo\\)\\.$", "⚠️ File too large (maximum $1 MB)."], ["^✅ (\\d+) flashcards prêtes !$", "✅ $1 flashcards ready!"], ["^⚠️ Seulement (\\d+) flashcard\\(s\\) obtenue\\(s\\) : il en faut au moins (\\d+)\\. Relancez la génération\\.$", "⚠️ Only $1 flashcard(s) obtained: at least $2 are needed. Generate again."], ["^❌ Erreur : (.*)$", "❌ Error: $1"], ["^⚠️ Génération impossible : (.*)$", "⚠️ Generation failed: $1"], ["^✅ Quiz de (\\d+) questions généré\\.$", "✅ Quiz of $1 questions generated."], ["^🎯 Quiz terminé : (\\d+)\\/(\\d+)\\.$", "🎯 Quiz completed: $1/$2."], ["^Score : (\\d+)$", "Score: $1"], ["^(\\d+)% de bonnes réponses$", "$1% correct answers"], ["^Bonne réponse : (.*)$", "Correct answer: $1"], ["^Le titre du document est : (.*)$", "The title of the document is: $1"], ["^QUIZ \\/ QCM · (.*)$", "QUIZ / MCQ · $1"], ["^(\\d+) fragments indexés · Progression (\\d+)% · RAG local$", "$1 indexed fragments · Progress $2% · Local RAG"], ["^Progression réelle · (.*)$", "Real progress · $1"], ["^Ajouté le (.*)$", "Added on $1"], ["^Indexé · (\\d+) fragments$", "Indexed · $1 fragments"], ["^✅ Compte « (.+) » (activé|désactivé)\\.$", "✅ Account “$1” updated."], ["^✅ Mot de passe de « (.+) » réinitialisé.*$", "✅ Password of “$1” reset (their remembered sign-ins are revoked)."], ["^✅ « (.+) » est maintenant (administrateur|utilisateur)\\.$", "✅ “$1” role updated."], ["^✅ Compte « (.+) » et toutes ses données supprimés\\.$", "✅ Account “$1” and all its data deleted."], ["^✅ Sauvegarde créée \\((.+)\\)\\.$", "✅ Backup created ($1)."], ["^✅ (\\d+) connexion\\(s\\) mémorisée\\(s\\) expirée\\(s\\) purgée\\(s\\)\\.$", "✅ $1 expired remembered sign-in(s) purged."], ["^✅ Journal allégé : (\\d+) ligne\\(s\\) ancienne\\(s\\) supprimée\\(s\\)\\.$", "✅ Log trimmed: $1 old line(s) removed."], ["^✅ Index et cours rechargés.*$", "✅ Indexes and courses will be reloaded from disk on each user's next action."], ["^⏳ Génération des flashcards… \\((\\d+) s\\)$", "⏳ Generating flashcards… ($1 s)"], ["^⏳ Génération du quiz… \\((\\d+) s\\)$", "⏳ Generating the quiz… ($1 s)"], ["^⏳ Génération du résumé… \\((\\d+) s\\)$", "⏳ Generating the summary… ($1 s)"]].map(([re, to]) => [new RegExp(re), to]);
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
      showBanner(en ? '🔄 Unstable connection — your data is safe. Updating automatically…'
                    : '🔄 Connexion instable — vos données sont conservées. Mise à jour automatique…');
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
        else showBanner(window.__akoriLang === 'en' ? 'ℹ️ That action did not complete. Your data is safe — please try again.'
                                                   : "ℹ️ L'action n'a pas abouti. Vos données sont conservées — réessayez.");
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
      setTimeout(() => window.__akBanner && window.__akBanner(window.__akoriLang === 'en' ? '✅ Reconnected — your data is up to date.' : '✅ Reconnecté — vos données sont à jour.'), 900);
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
    name = u.get("username", key)
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
    return (
        gr.update(visible=not logged), gr.update(visible=logged), gr.update(visible=False),
        _user_badge_html(user), gr.update(visible=admin),
        gr.update(choices=names, value=sel), *data,
        status, (gr.update() if token is None else token), gr.Tabs(selected="home"),
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


def do_register(username, password, password2, remember, request: gr.Request):
    if password != password2:
        return _views_for(None, "⚠️ Les mots de passe ne correspondent pas.")
    ok, msg = _create_user(username, password)
    if not ok:
        return _views_for(None, msg)
    return do_login(username, password, remember, request)


def do_logout(stored_token, request: gr.Request):
    key = SESSIONS.pop(request.session_hash, None)
    _revoke_token(stored_token)
    _revoke_token(_cookie_token(request))
    _log("logout", key or "")
    return _views_for(None, "✅ Vous êtes déconnecté.", f"clear:{secrets.token_hex(3)}")


def do_auto_login(token, request: gr.Request):
    key = _user_from_token(token)
    if not key:
        return _views_for(None, "", f"clear:{secrets.token_hex(3)}" if token else None)
    SESSIONS[request.session_hash] = key
    _SESSION_SEEN[request.session_hash] = time.time()
    _log("auto_login", key)
    return _views_for(key)


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
    service = (
        "<div class='admin-service'>"
        f"<div><span>Modèle principal</span><b>{_escape_html(MODEL_NAME)}</b></div>"
        f"<div><span>Modèles de secours</span><b>{_escape_html(', '.join(_models_chain()[1:]) or '—')}</b></div>"
        f"<div><span>Clé API Gemini</span><b>{'✅ configurée' if GEMINI_API_KEY else '❌ manquante (GEMINI_API_KEY)'}</b></div>"
        f"<div><span>Disponibilité</span><b>{_fmt_duration(time.time() - APP_STARTED_AT)}</b></div>"
        f"<div><span>Dossier de données</span><b>{_escape_html(DATA_DIR)}</b></div>"
        f"<div><span>Taille PDF maximale</span><b>{int(MAX_PDF_MB)} Mo</b></div>"
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

    with gr.Column(visible=True, elem_id="login-view") as login_view:
        gr.HTML("<div class='login-hero'><div class='brand-mark big'>A</div><h1>AKORI</h1><p>Assistant Knowledge Organized to Revise Intelligently</p></div>")
        with gr.Column(elem_id="login-card"):
            gr.HTML("<div class='login-welcome'><h2>Bienvenue sur AKORI</h2><p>Connectez-vous pour retrouver vos cours, flashcards, quiz et historiques archivés.</p></div>")
            with gr.Tabs(elem_id="auth-tabs"):
                with gr.Tab("Connexion", id="signin"):
                    li_user = gr.Textbox(label="Nom d'utilisateur", elem_id="li-user")
                    li_pass = gr.Textbox(label="Mot de passe", type="password", elem_id="li-pass")
                    li_remember = gr.Checkbox(label="Rester connecté", value=True)
                    li_btn = gr.Button("Se connecter", variant="primary", elem_id="li-btn")
                with gr.Tab("Créer un compte", id="signup"):
                    su_user = gr.Textbox(label="Nom d'utilisateur", elem_id="su-user")
                    su_pass = gr.Textbox(label="Mot de passe", type="password", elem_id="su-pass")
                    su_pass2 = gr.Textbox(label="Confirmer le mot de passe", type="password", elem_id="su-pass2")
                    su_remember = gr.Checkbox(label="Rester connecté", value=True)
                    su_btn = gr.Button("Créer mon compte", variant="primary", elem_id="su-btn")
            login_status = gr.Markdown("", elem_id="login-status")
            gr.HTML("<div class='login-foot'>Vos données sont enregistrées dans votre compte et restent disponibles à chaque connexion.</div>")

    # ------------------------------------------------------------ vue : application
    with gr.Column(visible=False, elem_id="app-view") as app_view:
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=60, elem_id="sidebar"):
                menu_btn = gr.Button("☰", elem_id="sidebar-toggle")
                gr.HTML("<div class='brand'><div class='brand-mark'>A</div><div><div class='brand-name'>AKORI</div><div class='brand-sub'>Assistant Knowledge Organized to Revise Intelligently</div></div></div>")
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

                    with gr.Tab("À revoir", id="reviewq"):
                        gr.Markdown("## À revoir")
                        gr.Markdown("Cette section regroupera les flashcards marquées « À revoir » et les erreurs de quiz.")
                        review_queue = gr.HTML("<div class='empty-study'>📌 Votre file « À revoir » apparaîtra ici après les interactions.</div>")

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
                admin_file = gr.File(label="Sauvegarde", interactive=False)
        admin_msg = gr.Markdown("", elem_id="admin-msg")

    # Bouton invisible : le navigateur le déclenche seul après une coupure réseau pour réafficher l'état enregistré.
    resync_btn = gr.Button("resync", elem_id="resync-btn", elem_classes="akori-hidden")

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
        prompt = f"CONTEXTE DU COURS:\n{context}\n\nDEMANDE: Fais un résumé synthétique des points clés principaux de ce document."

        def work():
            response = _gemini_structured(prompt, _with_lang("Tu es AKORI. Résume uniquement les informations présentes dans le contexte fourni. Retourne un JSON {\"summary\":\"...\"}.", lang))
            data = _extract_json_from_text(response)
            return data.get("summary", response) if isinstance(data, dict) else response

        try:
            result = yield from _with_heartbeat(work, lambda sec: f"⏳ Génération du résumé… ({sec} s)")
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
    ENTER_OUT = [login_view, app_view, admin_view, user_badge, nav_admin, doc_selector, home_html, docs_html,
                 course_info, review_cards, global_progress_output, progress_output, chatbot, history_box,
                 login_status, token_out, tabs]
    li_btn.click(do_login, [li_user, li_pass, li_remember], ENTER_OUT).then(None, None, None, js=ALIGN_JS)
    li_pass.submit(do_login, [li_user, li_pass, li_remember], ENTER_OUT).then(None, None, None, js=ALIGN_JS)
    su_btn.click(do_register, [su_user, su_pass, su_pass2, su_remember], ENTER_OUT).then(None, None, None, js=ALIGN_JS)
    su_pass2.submit(do_register, [su_user, su_pass, su_pass2, su_remember], ENTER_OUT).then(None, None, None, js=ALIGN_JS)
    nav_logout.click(do_logout, [token_in], ENTER_OUT, js=LOGOUT_JS).then(None, None, None, js=ALIGN_JS)
    token_in.change(do_auto_login, [token_in], ENTER_OUT).then(None, None, None, js=ALIGN_JS)
    token_out.change(None, token_out, None, js=TOKEN_STORE_JS)
    demo.load(None, None, token_in, js=GET_TOKEN_JS)

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

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4, max_size=64).launch(
        share=True,
        debug=False,
        show_error=True,
        head='<meta name="google" content="notranslate"><meta http-equiv="Content-Language" content="fr"><script>document.documentElement.setAttribute("translate","no");document.documentElement.lang="fr";</script>',
        theme=theme_akori,
        css=custom_css,
        js=AKORI_NAV_JS,
    )
