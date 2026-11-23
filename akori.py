import contextvars
import io
import traceback
import atexit
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
import socket
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
def _make_client(key):
    try:
        return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=GEMINI_HTTP_TIMEOUT_MS))
    except Exception:
        return genai.Client(api_key=key)


def _extra_gemini_keys():
    """Clés de secours : GEMINI_API_KEYS (séparées par des virgules) ou GEMINI_API_KEY_2…5 (variables ou secrets Colab)."""
    raw = [os.getenv("GEMINI_API_KEYS") or ""] + [os.getenv(f"GEMINI_API_KEY_{i}") or "" for i in range(2, 6)]
    try:
        from google.colab import userdata  # type: ignore
        for name in ["GEMINI_API_KEYS"] + [f"GEMINI_API_KEY_{i}" for i in range(2, 6)]:
            try:
                raw.append(userdata.get(name) or "")
            except Exception:
                pass
    except Exception:
        pass
    keys = []
    for chunk in raw:
        for part in re.split(r"[,;\s]+", chunk):
            key = _clean_key(part)
            if key and key.isascii() and key != GEMINI_API_KEY and key not in keys:
                keys.append(key)
    return keys[:4]


class _MultiClient:
    """Plusieurs clés Gemini (projets différents = quotas différents) : une clé à bout de quota (429) est mise
    au repos 60 s et l'appel repart aussitôt avec la suivante. Même interface que genai.Client pour .models."""

    def __init__(self, clients):
        self.clients = clients
        self._rest = {}
        self._lock = threading.Lock()
        self.models = self._Models(self)

    def _order(self):
        now = time.monotonic()
        with self._lock:
            ready = [i for i in range(len(self.clients)) if self._rest.get(i, 0) <= now]
            resting = [i for i in range(len(self.clients)) if self._rest.get(i, 0) > now]
        return ready + resting

    def _cool(self, index, seconds=60):
        with self._lock:
            self._rest[index] = time.monotonic() + seconds

    class _Models:
        def __init__(self, owner):
            self.o = owner

        def generate_content(self, **kw):
            last = None
            for i in self.o._order():
                try:
                    return self.o.clients[i].models.generate_content(**kw)
                except APIError as e:
                    if e.code != 429:
                        raise
                    self.o._cool(i)
                    last = e
            raise last

        def generate_content_stream(self, **kw):
            def gen():
                last = None
                for i in self.o._order():
                    try:
                        iterator = iter(self.o.clients[i].models.generate_content_stream(**kw))
                        first = next(iterator)
                    except StopIteration:
                        return
                    except APIError as e:
                        if e.code != 429:
                            raise
                        self.o._cool(i)
                        last = e
                        continue
                    yield first
                    yield from iterator
                    return
                raise last
            return gen()

        def list(self, *a, **kw):
            return self.o.clients[0].models.list(*a, **kw)


_client_key = GEMINI_API_KEY or "missing-key"   # sans clé, l'appli démarre quand même (erreur claire à la 1re question)
client = _make_client(_client_key)
_EXTRA_KEYS = _extra_gemini_keys() if GEMINI_API_KEY else []
if _EXTRA_KEYS:
    client = _MultiClient([client] + [_make_client(k) for k in _EXTRA_KEYS])
    print(f"🔑 {1 + len(_EXTRA_KEYS)} clés Gemini : bascule automatique en cas de quota atteint.")
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

    return (lines[0][:117] + "…") if lines and len(lines[0]) > 120 else (lines[0] if lines else "")


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
  if os.path.isdir("/data") and os.access("/data", os.W_OK):
    print("✅ Données AKORI enregistrées dans /data (disque de l'hébergeur).")
    return "/data", True
  print("⚠️ Aucun stockage persistant (Google Drive ou /data) : les données sont temporaires (perdues au redémarrage).")
  return LOCAL_DATA_DIR, False


DATA_DIR, STORAGE_ON_DRIVE = _resolve_data_dir()
os.makedirs(DATA_DIR, exist_ok=True)


def _writable(path):
  try:
    probe = os.path.join(path, f".akori_write_test_{os.getpid()}")
    with open(probe, "w") as f:
      f.write("ok")
    os.remove(probe)
    return True
  except OSError:
    return False


if not _writable(DATA_DIR):
  # Disque monté en lecture seule ou sans droits (ex. /data mal configuré) : sans ce repli, ni comptes ni PDF ne s'enregistrent.
  print(f"⚠️ Impossible d'écrire dans {DATA_DIR} : repli sur {LOCAL_DATA_DIR} (données temporaires).")
  DATA_DIR, STORAGE_ON_DRIVE = LOCAL_DATA_DIR, False
  os.makedirs(DATA_DIR, exist_ok=True)
STORAGE_KIND = ("Google Drive" if "/drive/" in DATA_DIR.replace("\\", "/").lower()
                else ("Disque persistant" if STORAGE_ON_DRIVE or DATA_DIR.rstrip("/") == "/data" else "Temporaire"))

# ---- Sauvegarde dans un dépôt Hugging Face (facultatif) : les comptes et les cours survivent aux redémarrages du Space,
# même sans « Persistent storage » payant. Activer avec les secrets AKORI_HF_DATASET=<compte>/akori-data et HF_TOKEN (écriture).
HF_DATASET = (os.getenv("AKORI_HF_DATASET") or "").strip()
HF_TOKEN = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()
HF_SYNC = bool(HF_DATASET and HF_TOKEN)
_HF_DIRTY = threading.Event()
_HF_STATE = {"last": "", "error": ""}


def _hf_restore():
  """Au démarrage : si le dossier de données est vide (Space redémarré), on récupère la dernière sauvegarde."""
  if not HF_SYNC:
    return
  try:
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=HF_TOKEN)
    api.create_repo(HF_DATASET, repo_type="dataset", private=True, exist_ok=True)
    if os.path.exists(os.path.join(DATA_DIR, "users.json")):
      return
    snapshot_download(repo_id=HF_DATASET, repo_type="dataset", local_dir=DATA_DIR, token=HF_TOKEN)
    print(f"✅ Données AKORI restaurées depuis le dépôt Hugging Face {HF_DATASET}.")
  except Exception as e:
    _HF_STATE["error"] = f"{type(e).__name__}: {e}"[:200]
    print(f"⚠️ Restauration depuis {HF_DATASET} impossible : {e}")


def _hf_sync_loop():
  """Envoie les changements (comptes, cours, progression) vers le dépôt, au plus toutes les 30 s."""
  from huggingface_hub import HfApi
  api = HfApi(token=HF_TOKEN)
  while True:
    _HF_DIRTY.wait()
    time.sleep(30)
    _HF_DIRTY.clear()
    try:
      api.upload_folder(folder_path=DATA_DIR, repo_id=HF_DATASET, repo_type="dataset", commit_message="AKORI : sauvegarde automatique",
                        ignore_patterns=["*.tmp", ".akori_write_test_*", "LISEZ-MOI.txt", "**/__pycache__/**"])
      _HF_STATE["last"], _HF_STATE["error"] = datetime.now().isoformat(timespec="seconds"), ""
    except Exception as e:
      _HF_STATE["error"] = f"{type(e).__name__}: {e}"[:200]
      print(f"⚠️ Sauvegarde vers {HF_DATASET} impossible : {e}")
      _HF_DIRTY.set()
      time.sleep(120)


_hf_restore()
if HF_SYNC:
  threading.Thread(target=_hf_sync_loop, name="akori-hf-sync", daemon=True).start()
  STORAGE_KIND = f"Hugging Face ({HF_DATASET})"
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
  _io_flush(path)
  try:
    with open(path, "r", encoding="utf-8") as f:
      return json.load(f)
  except (OSError, ValueError):
    return default


def _write_json(path, data):
  _io_drop(path)   # une sauvegarde différée plus ancienne du même fichier ne doit pas passer après celle-ci
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = f"{path}.{os.getpid()}.tmp"
  with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
  os.replace(tmp, path)
  if HF_SYNC:
    _HF_DIRTY.set()


# Écritures différées : sur Google Drive, une écriture peut prendre plusieurs centaines de millisecondes.
# L'historique des cours et les journaux sont donc écrits par un fil dédié, sans retarder les réponses.
# Le contenu est figé (sérialisé) au moment de l'appel ; toute lecture d'un fichier en attente attend son écriture.
_IO_COND = threading.Condition()
_IO_FILES = {}       # chemin -> texte complet (la dernière version remplace les précédentes)
_IO_APPENDS = []     # (chemin, ligne) dans l'ordre d'arrivée
_IO_BUSY = set()     # chemins en cours d'écriture
_IO_THREAD = None


def _io_worker():
  while True:
    with _IO_COND:
      while not _IO_FILES and not _IO_APPENDS:
        _IO_COND.wait()
      files = dict(_IO_FILES)
      appends = list(_IO_APPENDS)
      _IO_FILES.clear()
      _IO_APPENDS.clear()
      _IO_BUSY.update(files)
      _IO_BUSY.update(p for p, _ in appends)
    for path, text in files.items():
      try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
          f.write(text)
        os.replace(tmp, path)
      except OSError as e:
        print(f"⚠️ Écriture impossible ({os.path.basename(path)}) : {e}")
    grouped = {}
    for path, line in appends:
      grouped.setdefault(path, []).append(line)
    for path, lines in grouped.items():
      try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
          f.write("".join(lines))
      except OSError:
        pass
    if HF_SYNC and files:
      _HF_DIRTY.set()
    with _IO_COND:
      _IO_BUSY.clear()
      _IO_COND.notify_all()


def _io_start():
  global _IO_THREAD
  if _IO_THREAD is None:
    _IO_THREAD = threading.Thread(target=_io_worker, name="akori-io", daemon=True)
    _IO_THREAD.start()


def _write_json_later(path, data):
  """Comme _write_json, mais l'écriture disque se fait en arrière-plan."""
  text = json.dumps(data, ensure_ascii=False, indent=2)
  _io_start()
  with _IO_COND:
    _IO_FILES[path] = text
    _IO_COND.notify_all()


def _append_line_later(path, rec):
  line = json.dumps(rec, ensure_ascii=False) + "\n"
  _io_start()
  with _IO_COND:
    _IO_APPENDS.append((path, line))
    _IO_COND.notify_all()


def _io_flush(path=None, timeout=15.0):
  """Attend que les écritures en attente (toutes, ou celles d'un fichier) soient sur le disque."""
  deadline = time.monotonic() + timeout
  with _IO_COND:
    def pending():
      if path is None:
        return bool(_IO_FILES or _IO_APPENDS or _IO_BUSY)
      return path in _IO_FILES or path in _IO_BUSY or any(p == path for p, _ in _IO_APPENDS)
    while pending():
      left = deadline - time.monotonic()
      if left <= 0:
        return False
      _IO_COND.wait(left)
  return True


def _io_drop(prefix):
  """Annule les écritures en attente d'un fichier (ou de tout un dossier) puis attend celle en cours.
  Indispensable avant de supprimer un cours : sinon l'écriture différée recréerait son dossier."""
  with _IO_COND:
    for p in [p for p in _IO_FILES if p == prefix or p.startswith(prefix.rstrip(os.sep) + os.sep)]:
      _IO_FILES.pop(p, None)
    _IO_APPENDS[:] = [(p, l) for p, l in _IO_APPENDS if not (p == prefix or p.startswith(prefix.rstrip(os.sep) + os.sep))]
    deadline = time.monotonic() + 15
    while any(p == prefix or p.startswith(prefix.rstrip(os.sep) + os.sep) for p in _IO_BUSY) and time.monotonic() < deadline:
      _IO_COND.wait(0.5)


atexit.register(_io_flush)


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
  _append_line_later(ACTIVITY_LOG, rec)


def _read_log(limit=300):
  _io_flush(ACTIVITY_LOG)
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
STRICT_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]{1,64}@((?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24})$")
DISPOSABLE_DOMAINS = {"mailinator.com", "yopmail.com", "10minutemail.com", "guerrillamail.com", "tempmail.com", "trashmail.com", "sharklasers.com",
                      "getnada.com", "throwawaymail.com", "maildrop.cc", "dispostable.com", "temp-mail.org", "fakeinbox.com", "mailnesia.com"}
EMAIL_TYPOS = {"gmial.com": "gmail.com", "gmai.com": "gmail.com", "gmail.co": "gmail.com", "gamil.com": "gmail.com", "gnail.com": "gmail.com",
               "gmail.con": "gmail.com", "hotmial.com": "hotmail.com", "hotmail.co": "hotmail.com", "yahooo.com": "yahoo.com", "yaho.com": "yahoo.com",
               "outlok.com": "outlook.com", "outloook.com": "outlook.com"}


def _domain_receives_mail(domain):
  """True si le domaine existe et peut recevoir du courrier. En cas de doute (pas de réseau, délai dépassé), on accepte."""
  result = {"ok": True}

  def probe():
    try:
      try:
        import dns.resolver                                    # dnspython, si présent : on cherche un enregistrement MX
        resolver = dns.resolver.Resolver()
        resolver.lifetime = 3
        try:
          resolver.resolve(domain, "MX")
          return
        except dns.resolver.NoAnswer:
          resolver.resolve(domain, "A")
          return
      except ImportError:
        socket.getaddrinfo(domain, None)
    except Exception as e:
      name = type(e).__name__
      if name in ("NXDOMAIN", "NoAnswer", "NoNameservers") or (isinstance(e, socket.gaierror) and e.errno in (socket.EAI_NONAME, getattr(socket, "EAI_NODATA", -5))):
        result["ok"] = False

  t = threading.Thread(target=probe, daemon=True)
  t.start()
  t.join(4)
  return result["ok"]


def _email_problem(addr):
  """Message d'erreur si l'adresse n'est pas une vraie adresse utilisable, sinon None."""
  addr = str(addr or "").strip()
  m = STRICT_EMAIL_RE.match(addr)
  local = addr.split("@")[0]
  if not m or local.startswith(".") or local.endswith(".") or ".." in local:
    return "⚠️ Adresse e-mail invalide (exemple : prenom.nom@gmail.com)."
  domain = m.group(1).lower()
  if domain in EMAIL_TYPOS:
    return f"⚠️ Vouliez-vous écrire @{EMAIL_TYPOS[domain]} ?"
  if domain in DISPOSABLE_DOMAINS:
    return "⚠️ Les adresses jetables ne sont pas acceptées : utilisez une adresse e-mail que vous consultez."
  if os.getenv("AKORI_EMAIL_CHECK", "1") != "0" and not _domain_receives_mail(domain):
    return f"⚠️ Le domaine « {domain} » n'existe pas ou ne reçoit pas d'e-mails."
  return None
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
  if not allow_username and role != "admin":
    problem = _email_problem(ident)
    if problem:
      return False, problem
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
# Serveur public (Hugging Face, Render, Fly, Docker…) : jamais de mot de passe admin connu d'avance.
PUBLIC_DEPLOY = (os.getenv("AKORI_PUBLIC") == "1" or bool(os.getenv("SPACE_ID")) or bool(os.getenv("RENDER"))
                 or bool(os.getenv("FLY_APP_NAME")) or bool(os.getenv("RAILWAY_ENVIRONMENT")))


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
  if not PUBLIC_DEPLOY and "admin" in users and users["admin"].get("role") == "admin" and os.path.exists(legacy_note):
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
    password = os.getenv("AKORI_ADMIN_PASSWORD") or (secrets.token_urlsafe(12) if PUBLIC_DEPLOY else DEFAULT_ADMIN_PASSWORD)
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
      "answer_cache": data.get("answer_cache", {}),
      "module": data.get("module", ""),
      "oral_bank": data.get("oral_bank", []),
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
      "mindmap_cache": data.get("mindmap_cache", {}),
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
  """Sauvegarde uniquement les métadonnées JSON, sans réécrire l'index FAISS (écriture en arrière-plan)."""
  data = documents_db.get(filename)
  if not data:
    return
  _write_json_later(_doc_metadata_path(filename), _doc_metadata(filename, data))


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
          "answer_cache": metadata.get("answer_cache", {}),
          "module": metadata.get("module", ""),
          "oral_bank": metadata.get("oral_bank", []),
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
          "mindmap_cache": metadata.get("mindmap_cache", {}),
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
  _io_drop(os.path.join(USERS_DIR, key))
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
  progress.setdefault("oral_attempts", [])
  return progress


def _save_progress(filename):
  _save_history(filename)


def _course_stats(filename):
  """Toutes les mesures d'un cours, calculées au même endroit pour que chaque vue affiche les mêmes chiffres.

  Progression du cours = moyenne de 3 activités (flashcards, quiz, colle orale). Une activité non commencée
  compte pour 0 % : on ne peut donc pas avoir « 0 quiz terminé » et une bonne moyenne de quiz."""
  doc = documents_db.get(filename) or {}
  progress = _progress_for(filename)
  flash = progress.get("flashcards", {})
  known = sum(1 for v in flash.values() if v == "known")
  review = sum(1 for v in flash.values() if v == "review")
  series = len(doc.get("flashcards") or [])
  cards_total = max(series, len(flash))
  flash_pct = round(known / cards_total * 100) if cards_total else 0
  attempts = progress.get("quiz_attempts", [])
  q_points = sum(int(a.get("score", 0)) for a in attempts)
  q_total = sum(int(a.get("total", 0)) for a in attempts)
  quiz_pct = round(q_points / q_total * 100) if q_total else 0
  oral = progress.get("oral_attempts", [])
  oral_avg = (sum(float(a.get("score", 0)) for a in oral) / len(oral)) if oral else 0
  oral_pct = round(oral_avg * 10)
  history = doc.get("history", []) or []
  return {
      "flash_known": known, "flash_review": review, "flash_total": cards_total, "flash_answered": len(flash), "flash_pct": flash_pct,
      "quizzes": len(attempts), "quiz_pct": quiz_pct, "quiz_questions": q_total,
      "orals": len(oral), "oral_avg": oral_avg, "oral_pct": oral_pct,
      "questions": sum(1 for m in history if isinstance(m, dict) and m.get("role") == "user"),
      "summary": bool(doc.get("summary_cache")), "pins": len(doc.get("pins", []) or []),
      "started": sum(1 for v in (len(flash), len(attempts), len(oral)) if v),
      "percent": round((flash_pct + quiz_pct + oral_pct) / 3),
  }


def _course_progress_percent(filename):
  if not filename or filename not in documents_db:
    return 0
  return _course_stats(filename)["percent"]


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
    return {"percent": 0, "courses": 0, "started": 0, "flashcards": 0, "known": 0, "quizzes": 0, "quiz_avg": None,
            "orals": 0, "oral_avg": None, "questions": 0}
  all_stats = [_course_stats(name) for name in documents_db]
  q_pts = q_tot = 0
  for name in documents_db:
    for a in _progress_for(name).get("quiz_attempts", []):
      q_pts += int(a.get("score", 0))
      q_tot += int(a.get("total", 0))
  orals = sum(st["orals"] for st in all_stats)
  oral_sum = sum(st["oral_avg"] * st["orals"] for st in all_stats)
  return {"percent": round(sum(st["percent"] for st in all_stats) / len(all_stats)), "courses": len(all_stats),
          "started": sum(1 for st in all_stats if st["started"]),
          "flashcards": sum(st["flash_total"] for st in all_stats), "known": sum(st["flash_known"] for st in all_stats),
          "quizzes": sum(st["quizzes"] for st in all_stats), "quiz_avg": round(q_pts / q_tot * 100) if q_tot else None,
          "orals": orals, "oral_avg": (oral_sum / orals) if orals else None, "questions": sum(st["questions"] for st in all_stats)}


def _pct_or_dash(value, has_data):
  return f"{value}%" if has_data else "—"


def _flash_caption(st):
  if not st["flash_total"]:
    return "Pas encore commencé : générez une série dans Flashcards."
  extra = f" · {st['flash_review']} à revoir" if st["flash_review"] else ""
  return f"{st['flash_known']} maîtrisée(s) sur {st['flash_total']}{extra}"


def _quiz_caption(st):
  if not st["quizzes"]:
    return "Pas encore commencé : terminez un quiz pour obtenir une moyenne."
  return f"{st['quizzes']} quiz terminé(s) · moyenne {st['quiz_pct']} % sur {st['quiz_questions']} questions"


def _oral_caption(st):
  if not st["orals"]:
    return "Pas encore commencé : lancez une colle orale."
  return f"{st['orals']} colle(s) · moyenne {st['oral_avg']:.1f}/10"


def global_progress_html(active_doc=None):
  if not documents_db:
    return """<div class='progress-empty'><b>Votre progression commencera ici.</b><br>Importez un cours pour créer votre premier suivi. Tant qu'aucun cours n'est chargé, la progression reste à 0 %.</div>"""
  st = _global_progress_stats()
  pct = st['percent']
  gp_mood = "wave" if not st["started"] else ("celebrate" if pct >= 80 else ("happy" if pct >= 50 else ("encourage" if pct >= 20 else "sad")))

  # Même présentation que la vue détaillée : une ligne par indicateur, avec sa barre et sa légende.
  def row(icon, title, value, shown, caption):
    on = shown != "—"
    return (f"<div class='act-row {'act-on' if on else 'act-off'}'><div class='act-head'><span class='act-name'>{icon} {title}</span><b>{shown}</b></div>"
            f"<div class='act-bar'><i style='width:{max(0, min(100, value)) if on else 0}%'></i></div>"
            f"<div class='act-foot'><small>{_escape_html(caption)}</small></div></div>")

  started_pct = round(st["started"] / st["courses"] * 100) if st["courses"] else 0
  flash_pct = round(st["known"] / st["flashcards"] * 100) if st["flashcards"] else 0
  rows = (
      row("📚", "Cours commencés", started_pct, f"{st['started']}/{st['courses']}",
          "Un cours est commencé dès qu'une activité y est lancée.")
      + row("🎰", "Flashcards maîtrisées", flash_pct, f"{st['known']}/{st['flashcards']}" if st["flashcards"] else "—",
            f"{flash_pct} % de vos cartes sont marquées « je savais »." if st["flashcards"] else "Aucune série de flashcards pour l'instant.")
      + row("🎯", "Moyenne des quiz", st["quiz_avg"] or 0, f"{st['quiz_avg']}%" if st["quiz_avg"] is not None else "—",
            f"{st['quizzes']} quiz terminé(s), tous cours confondus." if st["quizzes"] else "Aucun quiz terminé pour l'instant.")
      + row("🎤", "Moyenne des colles orales", round((st["oral_avg"] or 0) * 10), f"{st['oral_avg']:.1f}/10" if st["oral_avg"] is not None else "—",
            f"{st['orals']} colle(s) passée(s)." if st["orals"] else "Aucune colle orale pour l'instant."))
  courses = []
  for i, (name, data) in enumerate(documents_db.items()):
    cpct = _course_stats(name)["percent"]
    title = data.get("title") or os.path.splitext(name)[0]
    active = " gp-active" if name == active_doc else ""
    courses.append(f"<div class='gp-course gp-open gp-at-{i}{active}' role='button' tabindex='0' title='Voir le détail de ce cours'>"
                   f"<span class='gp-cname'>{_escape_html(title)}</span><div class='act-bar'><i style='width:{cpct}%'></i></div><b>{cpct}%</b></div>")
  return (_progress_carousel_html()
          + f"<div class='progress-hero gp-hero gp-open' role='button' tabindex='0' title='Voir le détail par cours'>"
            f"<div class='progress-ring' style='--progress:{pct}%;'><span class='cu'>{pct}%</span></div>"
            f"<div class='gp-copy'><div class='progress-title'>Progression globale</div>"
            f"<div class='progress-sub'>Moyenne de vos {st['courses']} cours. Chaque cours compte trois activités : flashcards, quiz et colle orale.</div>"
            f"<span class='gp-hint'>Voir le détail par cours →</span></div><div class='gp-ms'>{mascot_svg(gp_mood)}</div></div>"
          + f"<div class='act-list gp-acts'>{rows}</div>"
          + f"<div class='gp-courses'><span class='eyebrow'>PAR COURS</span>{''.join(courses)}</div>")


GP_COLORS = ["#4f6df5", "#8e6cf4"]   # palette de l'application (bleu → violet), comme les autres barres
FR_MONTHS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]


def _progress_carousel_html():
    """Détail de la progression : une carte par cours, à faire défiler (flèches, glisser, clavier)."""
    try:
        sessions = _read_sessions()
    except Exception:
        sessions = []
    last = {}
    for rec in sessions:
        d = _parse_day(rec.get("t"))
        if d and rec.get("doc"):
            last[rec["doc"]] = max(last.get(rec["doc"], d), d)
    try:
        mods = _space()["modules"]
        plan = _space().get("plan") or {}
    except Exception:
        mods, plan = {}, {}
    exam = _parse_day(plan.get("exam_date")) if plan and plan.get("context") != "free" else None
    exam_names = set(_plan_cfg_names(plan)) if exam else set()
    cards = []
    for i, (name, data) in enumerate(documents_db.items()):
        cs = _course_stats(name)
        pct = cs["percent"]
        color = GP_COLORS[i % len(GP_COLORS)]
        day = last.get(name) or _parse_day(data.get("added_at"))
        when = f"{day.day} {FR_MONTHS[day.month - 1]} {day.year}" if day else ""
        title = data.get("title") or os.path.splitext(name)[0]
        title = title if len(title) <= 46 else title[:44] + "…"
        mod = (mods.get(data.get("module") or "") or {}).get("name") or "Sans module"
        circ = 2 * 3.14159 * 52
        dash = circ * pct / 100
        val = (f"<circle cx='70' cy='70' r='52' class='gp-val' stroke='{color}' stroke-dasharray='{dash:.1f} {circ:.1f}' transform='rotate(-90 70 70)'/>"
               if pct > 0 else "")
        ring = (f"<svg viewBox='0 0 140 140' class='gp-ring' aria-hidden='true'><circle cx='70' cy='70' r='52' class='gp-track-c'/>{val}</svg>"
                f"<b class='gp-pct'>{pct}%</b>")
        bars = "".join(
            f"<div class='gp-bar'><span>{lab}</span><div><i style='width:{v}%'></i></div><em>{shown}</em></div>"
            for lab, v, shown in (
                ("Flashcards", cs["flash_pct"], f"{cs['flash_pct']}%" if cs["flash_answered"] else "—"),
                ("Quiz", cs["quiz_pct"], f"{cs['quiz_pct']}%" if cs["quizzes"] else "—"),
                ("Colle orale", cs["oral_pct"], f"{cs['oral_pct']}%" if cs["orals"] else "—")))
        dots = "".join(f"<span class='gp-dot{' on' if on else ''}' title='{lab}'>{ico}</span>" for lab, ico, on in (
            ("Flashcards", "🎰", cs["flash_answered"] > 0), ("Quiz", "🎯", cs["quizzes"] > 0), ("Colle orale", "🎤", cs["orals"] > 0)))
        if exam and name in exam_names and exam >= _date.today():
            left = (exam - _date.today()).days
            chip = "Examen aujourd'hui" if left == 0 else f"Examen dans {left} j"
        else:
            chip = f"{cs['started']}/3 activités"
        cards.append(f"<article class='gp-card' style='--c:{color}'><div class='gp-glow'></div><div class='gp-date'>{when}</div>"
                     f"<h3 title='{_escape_html(name)}'>{_escape_html(title)}</h3><div class='gp-sub'>{_escape_html(mod)}</div>"
                     f"<div class='gp-ringwrap'>{ring}</div><div class='gp-bars'>{bars}</div>"
                     f"<div class='gp-foot'><div class='gp-dots'>{dots}</div><span class='gp-chip'>{chip}</span></div></article>")
    if not cards:
        return ""
    nav = ("<button type='button' class='gp-nav gp-prev' aria-label='Cours précédent'>‹</button>"
           "<button type='button' class='gp-nav gp-next' aria-label='Cours suivant'>›</button>") if len(cards) > 1 else ""
    pager = "".join(f"<i class='{'on' if i == 0 else ''}'></i>" for i in range(len(cards))) if len(cards) > 1 else ""
    return (f"<div class='gp-modal' role='dialog' aria-modal='true' aria-label='Détail de la progression'>"
            f"<button type='button' class='gp-close' aria-label='Fermer'>✕</button>"
            f"<div class='gp-stage'>{''.join(cards)}</div>{nav}<div class='gp-pager'>{pager}</div></div>")


def progress_detail_html(active_doc=None):
  if not documents_db:
    return """<div class='progress-empty'><b>Aucune progression à afficher.</b><br>Importez un cours puis utilisez les flashcards ou terminez un quiz pour commencer à construire votre progression.</div>"""
  if not active_doc or active_doc not in documents_db:
    active_doc = next(iter(documents_db))
  data = documents_db[active_doc]
  cs = _course_stats(active_doc)
  pct = cs["percent"]
  topics = _progress_topics(active_doc)
  strengths = [(t, round(v["correct"] / v["total"] * 100)) for t, v in topics.items() if v["total"] and v["correct"] / v["total"] >= .75]
  weaknesses = [(t, round(v["correct"] / v["total"] * 100)) for t, v in topics.items() if v["total"] and v["correct"] / v["total"] < .60]
  if not topics:
    strengths_html = "<div class='progress-note'>Pas encore assez de réponses de quiz pour identifier un point fort.</div>"
    weaknesses_html = "<div class='progress-note'>Pas encore assez de réponses de quiz pour identifier un point faible.</div>"
  else:
    strengths_html = "".join(f"<div class='strength-row'><span>{_escape_html(t)}</span><b>{score}%</b></div>" for t, score in sorted(strengths, key=lambda x: -x[1])) or "<div class='progress-note'>Aucun point fort clairement établi pour le moment.</div>"
    weaknesses_html = "".join(f"<div class='weakness-row'><span>{_escape_html(t)}</span><b>{score}%</b></div>" for t, score in sorted(weaknesses, key=lambda x: x[1])) or "<div class='progress-note'>Aucun point faible clairement établi pour le moment.</div>"

  def activity(title, icon, value, started, caption, target):
    shown = f"{value}%" if started else "—"
    state = "act-on" if started else "act-off"
    return (f"<div class='act-row {state}'><div class='act-head'><span class='act-name'>{icon} {title}</span><b>{shown}</b></div>"
            f"<div class='act-bar'><i style='width:{value if started else 0}%'></i></div>"
            f"<div class='act-foot'><small>{_escape_html(caption)}</small>"
            f"<button type='button' class='ak-act ak-go ak-t-{target}'>▶ <span>{'Continuer' if started else 'Commencer'}</span></button></div></div>")

  activities = (
      activity("Flashcards", "🎰", cs["flash_pct"], cs["flash_total"] > 0 and cs["flash_answered"] > 0, _flash_caption(cs), "flash")
      + activity("Quiz / QCM", "🎯", cs["quiz_pct"], cs["quizzes"] > 0, _quiz_caption(cs), "quiz")
      + activity("Colle orale", "🎤", cs["oral_pct"], cs["orals"] > 0, _oral_caption(cs), "oral"))
  if not cs["flash_answered"]:
    nxt = ("Générez vos flashcards pour commencer à mémoriser.", "flash", "Flashcards")
  elif not cs["quizzes"]:
    nxt = ("Terminez un quiz pour mesurer ce que vous retenez.", "quiz", "Quiz / QCM")
  elif not cs["orals"]:
    nxt = ("Passez une colle orale pour vous entraîner à expliquer le cours.", "oral", "Colle orale")
  elif cs["flash_review"]:
    nxt = (f"Revoyez vos {cs['flash_review']} flashcard(s) marquée(s) « à revoir ».", "flash", "Flashcards")
  else:
    nxt = ("Bravo : les trois activités sont lancées. Continuez pour monter vers 100 %.", "assistant", "Assistant IA")
  info = (f"<div class='info-chips'><span><b>{cs['questions']}</b> question(s) posée(s) à l'assistant</span>"
          f"<span>Résumé : <b>{'généré' if cs['summary'] else 'non généré'}</b></span>"
          f"<span><b>{cs['pins']}</b> chapitre(s) épinglé(s) dans À revoir</span></div>")
  return f"""
  <div class='progress-hero'>
    <div class='progress-ring' style='--progress:{pct}%;'><span class='cu'>{pct}%</span></div>
    <div><div class='progress-title'>Progression du cours · {_escape_html(data.get('title') or active_doc)}</div>
    <div class='progress-sub'>{cs['started']}/3 activités commencées · une activité non commencée compte pour 0 %.</div></div>
  </div>
  <div class='act-list'>{activities}</div>
  {info}
  <div class='next-step'><div><span class='eyebrow'>PROCHAINE ÉTAPE</span><p>{nxt[0]}</p></div><button type='button' class='ak-act ak-go ak-t-{nxt[1]}'>▶ <span>{nxt[2]}</span></button></div>
  <div class='progress-columns'>
    <div class='progress-box'><h3>✦ Points forts</h3><small class='progress-src'>d'après vos réponses de quiz sur ce cours</small>{strengths_html}</div>
    <div class='progress-box'><h3>↗ Points faibles</h3><small class='progress-src'>d'après vos réponses de quiz sur ce cours</small>{weaknesses_html}</div>
  </div>
  """

# ---- espace de travail de l'utilisateur : modules (espace semestriel), réglages du plan, journal des séances ----
SEMESTERS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]


def _space_path():
  return os.path.join(USERS_DIR, _require_user(), "space.json")


def _space():
  try:
    data = _read_json(_space_path(), None)
  except PermissionError:                           # avant la connexion : aucun espace
    data = None
  if not isinstance(data, dict):
    data = {}
  data.setdefault("modules", {})
  data.setdefault("plan", {})
  return data


def _space_save(data):
  with _STORE_LOCK:
    _write_json(_space_path(), data)


def _sessions_path():
  return os.path.join(USERS_DIR, _require_user(), "sessions.jsonl")


def _session(kind, doc_name="", **fields):
  """Journal de révision de l'utilisateur : une ligne par séance (quiz, flashcard, colle orale, question, résumé)."""
  try:
    rec = {"t": datetime.now().isoformat(timespec="seconds"), "k": kind, "doc": doc_name or ""}
    rec.update({k: v for k, v in fields.items() if v is not None})
    _append_line_later(_sessions_path(), rec)
  except Exception:
    pass


def _read_sessions(limit=4000):
  try:
    path = _sessions_path()
    _io_flush(path)
    with open(path, "r", encoding="utf-8") as f:
      lines = f.readlines()[-limit:]
  except (OSError, PermissionError):
    return []
  out = []
  for line in lines:
    try:
      out.append(json.loads(line))
    except ValueError:
      continue
  return out


def _module_choices():
  """Choix pour les listes déroulantes : (libellé, identifiant)."""
  mods = _space()["modules"]
  items = sorted(mods.items(), key=lambda kv: (kv[1].get("semester", ""), kv[1].get("name", "").lower()))
  return [("Aucun module", "")] + [(f"{m.get('semester') or '—'} · {m.get('name', '')}", mid) for mid, m in items]


def _doc_module(doc_name):
  return (documents_db.get(doc_name) or {}).get("module", "") or ""


def _scope_docs(doc_name, scope):
  """Cours interrogés : le cours actif, son module, ou tout l'espace."""
  if scope == "all":
    return list(documents_db.keys())
  if scope == "module":
    mod = _doc_module(doc_name)
    if mod:
      return [n for n, d in documents_db.items() if (d.get("module") or "") == mod]
  return [doc_name]


# ---------------------------------------------------------------------------------------------------------
# Mascotte d'AKORI : le « A » du logo, dessiné en SVG pour changer d'expression selon la situation
# (réfléchit, content, fête, encourage, triste, déterminé, salue, montre, surpris, dort, lit).
# ---------------------------------------------------------------------------------------------------------
MASCOT_MOODS = {
    #  mood        bulle  yeux       sourcils   bouche   bras
    "think":     ("?",  "up",      "raised",  "think",  "chin"),
    "happy":     ("✓",  "happy",   "neutral", "smile",  ""),
    "celebrate": ("★",  "happy",   "neutral", "grin",   "up"),
    "encourage": ("!",  "normal",  "neutral", "smile",  "thumb"),
    "sad":       ("…",  "down",    "worried", "frown",  ""),
    "determined": ("",  "normal",  "angry",   "flat",   "flex"),
    "wave":      ("",   "normal",  "neutral", "smile",  "wave"),
    "point":     ("",   "right",   "neutral", "smile",  "point"),
    "surprised": ("!",  "wide",    "high",    "o",      ""),
    "sleep":     ("z",  "closed",  "neutral", "small",  ""),
    "read":      ("",   "down",    "neutral", "smile",  ""),
}
_MASCOT_N = [0]
_MC_INK = "#0b1a44"


def mascot_svg(mood="think", cls=""):
    """Mascotte vectorielle (même dessin que le logo) avec l'expression demandée."""
    if mood not in MASCOT_MOODS:
        mood = "think"
    _MASCOT_N[0] += 1
    u = f"ms{_MASCOT_N[0]}"
    bubble, eyes, brows, mouth, arms = MASCOT_MOODS[mood]

    def eye(cx, dx=0, dy=0, rx=12.0, ry=13.5, r=6.4):
        return (f"<ellipse cx='{cx}' cy='74' rx='{rx}' ry='{ry}' fill='#fff' stroke='{_MC_INK}' stroke-width='2.4'/>"
                f"<circle cx='{cx + dx}' cy='{75 + dy}' r='{r}' fill='#3563e9'/>"
                f"<circle cx='{cx + dx}' cy='{75 + dy}' r='{r / 2:.1f}' fill='{_MC_INK}'/>"
                f"<circle cx='{cx + dx - 2}' cy='{72.5 + dy}' r='1.7' fill='#fff'/>")

    if eyes == "happy":
        eye_svg = (f"<path d='M72 79 Q84 64 96 79 M104 79 Q116 64 128 79' fill='none' stroke='{_MC_INK}' "
                   "stroke-width='4.5' stroke-linecap='round'/>")
    elif eyes == "closed":
        eye_svg = (f"<path d='M73 76 Q84 83 95 76 M105 76 Q116 83 127 76' fill='none' stroke='{_MC_INK}' "
                   "stroke-width='4' stroke-linecap='round'/>")
    else:
        look = {"up": (3, -4), "down": (0, 4), "right": (4, 0), "normal": (0, 0), "wide": (0, 0)}[eyes]
        if eyes == "wide":
            eye_svg = eye(84, rx=13, ry=15.5, r=5) + eye(116, rx=13, ry=15.5, r=5)
        else:
            eye_svg = eye(84, *look) + eye(116, *look)
    brow_d = {
        "neutral": "M73 55 Q84 49 95 55 M105 55 Q116 49 127 55",
        "raised": "M71 49 Q83 38 96 48 M105 57 Q116 53 127 56",
        "worried": "M73 57 Q84 54 95 49 M105 49 Q116 54 127 57",
        "angry": "M73 49 Q85 51 96 58 M104 58 Q115 51 127 49",
        "high": "M72 48 Q84 39 96 47 M104 47 Q116 39 128 48",
    }[brows]
    brow_svg = f"<path d='{brow_d}' fill='none' stroke='{_MC_INK}' stroke-width='5' stroke-linecap='round'/>"
    mouth_svg = {
        "smile": f"<path d='M88 96 Q100 107 112 96' fill='none' stroke='{_MC_INK}' stroke-width='4' stroke-linecap='round'/>",
        "grin": (f"<path d='M86 94 Q100 113 114 94 Z' fill='{_MC_INK}'/>"
                 "<path d='M93 103 Q100 109 107 103 Q100 100 93 103 Z' fill='#ff7aa8'/>"),
        "small": f"<path d='M95 99 Q100 101 105 99' fill='none' stroke='{_MC_INK}' stroke-width='3.5' stroke-linecap='round'/>",
        "flat": f"<path d='M90 100 L110 99' fill='none' stroke='{_MC_INK}' stroke-width='4.5' stroke-linecap='round'/>",
        "frown": f"<path d='M89 104 Q100 95 111 104' fill='none' stroke='{_MC_INK}' stroke-width='4' stroke-linecap='round'/>",
        "o": f"<ellipse cx='100' cy='101' rx='5.5' ry='7' fill='{_MC_INK}'/>",
        "think": f"<path d='M92 101 Q99 97 108 100' fill='none' stroke='{_MC_INK}' stroke-width='4' stroke-linecap='round'/>",
    }[mouth]
    arm_paths = {
        "chin": ("<path d='M66 132 Q66 114 90 110' class='ms-arm'/>", "<circle cx='95' cy='109' r='8' class='ms-hand'/>"),
        "wave": ("<g class='ms-wave'><path d='M134 128 Q158 114 160 88' class='ms-arm'/><circle cx='160' cy='83' r='8' class='ms-hand'/></g>", ""),
        "up": ("<path d='M66 128 Q42 108 40 82' class='ms-arm'/><path d='M134 128 Q158 108 160 82' class='ms-arm'/>",
               "<circle cx='40' cy='77' r='8' class='ms-hand'/><circle cx='160' cy='77' r='8' class='ms-hand'/>"),
        "thumb": ("<path d='M134 130 Q154 126 160 110' class='ms-arm'/>",
                  "<circle cx='161' cy='106' r='8' class='ms-hand'/><rect x='157' y='88' width='7' height='15' rx='3.5' class='ms-hand'/>"),
        "point": ("<path d='M134 126 Q160 122 176 114' class='ms-arm'/>", "<circle cx='180' cy='112' r='7' class='ms-hand'/>"),
        "flex": ("<path d='M66 126 Q38 124 42 98' class='ms-arm'/><path d='M134 126 Q162 124 158 98' class='ms-arm'/>",
                 "<circle cx='42' cy='93' r='9' class='ms-hand'/><circle cx='158' cy='93' r='9' class='ms-hand'/>"),
    }
    arm_svg, hand_svg = arm_paths.get(arms, ("", ""))
    extra = ""
    if mood == "celebrate":
        star = "M0 -8 L2 -2 L8 0 L2 2 L0 8 L-2 2 L-8 0 L-2 -2 Z"
        extra = "".join(f"<path class='ms-spark' style='animation-delay:{d}s' transform='translate({x} {y}) scale({k})' d='{star}' fill='{c}'/>"
                        for x, y, k, c, d in ((24, 46, 1.1, "#ffd25e", 0), (178, 112, .9, "#7fe3ff", .4),
                                              (30, 130, .8, "#c79bff", .8), (150, 18, .7, "#ffd25e", 1.1)))
    elif mood == "sad":
        extra = "<path d='M134 46 Q139 56 134 60 Q129 56 134 46 Z' fill='#8fd8ff' class='ms-drop'/>"
    bubble_svg = ""
    if bubble:
        bubble_svg = (f"<g class='ms-bubble'><path d='M146 56 L140 70 L156 58 Z' fill='url(#{u}b)'/>"
                      f"<circle cx='164' cy='38' r='23' fill='url(#{u}b)' stroke='rgba(255,255,255,.35)' stroke-width='1.5'/>"
                      f"<text x='164' y='47' text-anchor='middle' class='ms-sym'>{bubble}</text></g>")
    return (
        f"<svg class='mascot mood-{mood} {cls}' viewBox='0 0 200 206' role='img' aria-label='AKORI'>"
        f"<defs><linearGradient id='{u}g' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#7fe3ff'/>"
        "<stop offset='.5' stop-color='#4f86ff'/><stop offset='1' stop-color='#7a3cf0'/></linearGradient>"
        f"<radialGradient id='{u}h' cx='40%' cy='32%' r='75%'><stop offset='0' stop-color='#a8efff'/><stop offset='1' stop-color='#4f86ff'/></radialGradient>"
        f"<linearGradient id='{u}b' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='#a46bff'/><stop offset='1' stop-color='#5a2bd8'/></linearGradient>"
        f"<linearGradient id='{u}p' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#d6f6ff'/><stop offset='1' stop-color='#63c6ff'/></linearGradient></defs>"
        "<ellipse cx='100' cy='199' rx='52' ry='5' fill='rgba(0,0,0,.18)'/>"
        "<g class='ms-body'>"
        f"<ellipse cx='100' cy='186' rx='48' ry='10' fill='#7fe3ff' opacity='.28'/>"
        f"<path d='M54 170 L100 64 L146 170' fill='none' stroke='url(#{u}g)' stroke-width='40' stroke-linecap='round' stroke-linejoin='round'/>"
        f"<path d='M74 140 Q100 133 126 140' fill='none' stroke='url(#{u}g)' stroke-width='13' stroke-linecap='round'/>"
        f"<g class='ms-arms'>{arm_svg}</g>"
        f"<ellipse cx='100' cy='77' rx='41' ry='38' fill='url(#{u}h)'/>"
        "<ellipse cx='84' cy='55' rx='15' ry='7' fill='#fff' opacity='.32' transform='rotate(-22 84 55)'/>"
        f"<g class='ms-eyes'>{eye_svg}</g>{brow_svg}{mouth_svg}{hand_svg}"
        f"<path d='M56 180 Q78 169 100 176 L100 191 Q78 184 56 193 Z M144 180 Q122 169 100 176 L100 191 Q122 184 144 193 Z' "
        f"fill='url(#{u}p)' stroke='#3550d4' stroke-width='2' stroke-linejoin='round'/>"
        "<path d='M66 181 Q80 175 92 179 M108 179 Q120 175 134 181' fill='none' stroke='#3f7fe0' stroke-width='1.6' opacity='.7'/>"
        f"</g>{extra}{bubble_svg}</svg>")


def mascot_note(mood, title, text="", cls=""):
    """Bloc « mascotte + message » : états vides, fins de série, résultats."""
    body = f"<span>{text}</span>" if text else ""
    return (f"<div class='empty-study ms-note {cls}'><div class='ms-box'>{mascot_svg(mood)}</div>"
            f"<div class='ms-msg'><b>{title}</b>{body}</div></div>")


def _cheer(pct):
    """Humeur de la mascotte et message selon un résultat (0 à 100)."""
    if pct >= 100:
        return "celebrate", "Tout est maîtrisé, bravo !", "Reviens dans quelques jours pour ancrer ces notions sur la durée."
    if pct >= 70:
        return "happy", "Très bon travail !", "Encore quelques notions à consolider, et ce sera parfait."
    if pct >= 40:
        return "encourage", "Tu progresses !", "Revois les notions ratées puis recommence : ça va venir."
    return "sad", "Pas encore ça… on continue ensemble ?", "Relis les pages indiquées, puis refais une série : chaque essai compte."


def modules_html(active_doc=None):
  mods = _space()["modules"]
  groups = {}
  for mid, m in mods.items():
    groups.setdefault(m.get("semester") or "—", []).append((mid, m))
  loose = [n for n, d in documents_db.items() if not (d.get("module") or "") or d.get("module") not in mods]
  if not mods and not documents_db:
    return mascot_note("point", "Aucun module pour le moment.", "Importez vos PDF puis regroupez-les par module (matière) et par semestre.")
  blocks = []
  for sem in sorted(groups):
    cards = []
    for mid, m in sorted(groups[sem], key=lambda kv: kv[1].get("name", "").lower()):
      names = [n for n, d in documents_db.items() if (d.get("module") or "") == mid]
      chips = "".join(f"<span class='mod-chip{' mod-active' if n == active_doc else ''}'>{_escape_html(os.path.splitext(n)[0][:34])}</span>" for n in names) \
          or "<span class='mod-empty'>Aucun cours dans ce module</span>"
      pct = round(sum(_course_stats(n)["percent"] for n in names) / len(names)) if names else 0
      cards.append(f"<div class='mod-card'><div class='mod-head'><b>{_escape_html(m.get('name', ''))}</b><span>{len(names)} cours · {pct}%</span></div>"
                   f"<div class='mod-bar'><i style='width:{pct}%'></i></div><div class='mod-chips'>{chips}</div></div>")
    blocks.append(f"<div class='sem-block'><div class='sem-title'>{_escape_html(sem)}</div><div class='mod-grid'>{''.join(cards)}</div></div>")
  if loose:
    chips = "".join(f"<span class='mod-chip{' mod-active' if n == active_doc else ''}'>{_escape_html(os.path.splitext(n)[0][:34])}</span>" for n in loose)
    blocks.append(f"<div class='sem-block'><div class='sem-title'>Sans module</div><div class='mod-chips'>{chips}</div></div>")
  return "<div class='modules-wrap'>" + "".join(blocks) + "</div>"


def module_create(name, semester, doc_name=None):
  name = str(name or "").strip()
  if not name:
    return gr.update(), "⚠️ Donnez un nom au module.", modules_html(doc_name), gr.update()
  if len(name) > 60:
    return gr.update(), "⚠️ Nom de module trop long (60 caractères maximum).", modules_html(doc_name), gr.update()
  space = _space()
  if any(m.get("name", "").lower() == name.lower() and m.get("semester") == semester for m in space["modules"].values()):
    return gr.update(), "⚠️ Ce module existe déjà dans ce semestre.", modules_html(doc_name), gr.update()
  mid = secrets.token_hex(4)
  space["modules"][mid] = {"name": name, "semester": semester or ""}
  _space_save(space)
  assigned = ""
  if doc_name in documents_db:                      # le cours actif rejoint aussitôt le nouveau module
    documents_db[doc_name]["module"] = mid
    _save_history(doc_name)
    assigned = f" Le cours « {os.path.splitext(doc_name)[0]} » y a été ajouté."
  _log("module_create", detail=name)
  return gr.update(choices=_module_choices(), value=mid if assigned else ""), f"✅ Module « {name} » créé.{assigned}", modules_html(doc_name), ""


def module_assign(doc_name, module_id):
  if not doc_name or doc_name not in documents_db:
    return "⚠️ Sélectionnez d'abord un cours.", modules_html(doc_name)
  if module_id and module_id not in _space()["modules"]:
    module_id = ""
  documents_db[doc_name]["module"] = module_id or ""
  _save_history(doc_name)
  return "", modules_html(doc_name)


def module_delete(module_id, doc_name=None):
  space = _space()
  if not module_id or module_id not in space["modules"]:
    return gr.update(), "⚠️ Choisissez d'abord un module à supprimer.", modules_html(doc_name)
  name = space["modules"].pop(module_id).get("name", "")
  _space_save(space)
  for n, d in documents_db.items():                  # les cours restent, ils n'ont simplement plus de module
    if (d.get("module") or "") == module_id:
      d["module"] = ""
      _save_history(n)
  _log("module_delete", detail=name)
  return gr.update(choices=_module_choices(), value=""), f"✅ Module « {name} » supprimé (ses cours sont conservés).", modules_html(doc_name)


def module_for_doc(doc_name):
  """Quand on change de cours actif : la liste « Module du cours actif » suit."""
  return gr.update(choices=_module_choices(), value=_doc_module(doc_name) if doc_name in documents_db else ""), modules_html(doc_name)


def _student_hint(doc_name):
  """Adapte la profondeur des réponses au niveau de l'élève et à ses résultats (historique de révision personnalisé)."""
  key = _CUR_USER.get()
  parts = []
  try:
    level = {"college": "collégien", "lycee": "lycéen", "superieur": "étudiant de l'enseignement supérieur"}.get(_profile(key).get("level") if key else "")
    if level:
      parts.append(f"L'élève est un {level} : adapte le vocabulaire et la profondeur.")
    if doc_name in documents_db:
      cs = _course_stats(doc_name)
      if cs["quizzes"] and cs["quiz_pct"] < 50:
        parts.append("Ses résultats de quiz sont fragiles : explique pas à pas, avec un exemple simple tiré du cours.")
      elif cs["quizzes"] and cs["quiz_pct"] >= 80:
        parts.append("Il maîtrise bien le cours : va plus en profondeur et relie les notions entre elles.")
      topics = _progress_topics(doc_name)
      weak = [t for t, v in topics.items() if v["total"] >= 2 and v["correct"] / v["total"] < .6][:3]
      if weak:
        parts.append("Notions à renforcer (ses erreurs de quiz) : " + ", ".join(weak) + ".")
  except Exception:
    pass
  return " ".join(parts)


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
        "answer_cache": dict(keep.get("answer_cache", {})),
        "module": keep.get("module", "") or (previous or {}).get("module", ""),
        "oral_bank": list(keep.get("oral_bank", [])),
        "ocr_pages": ocr_pages,
        "pins": list(keep.get("pins", [])),
        "history": list(keep.get("history", [])),
        "progress": keep.get("progress") or _default_progress(),
        "flashcards": keep.get("flashcards", []),
        "flash_index": keep.get("flash_index", 0),
        "quiz": keep.get("quiz", {}),
        "summary_cache": keep.get("summary_cache", {}),
        "mindmap_cache": keep.get("mindmap_cache", {}),
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
  except PermissionError as e:
    if not _CUR_USER.get():
      yield ("done", None, "⚠️ Session expirée : rechargez la page pour vous reconnecter.")
    else:
      _log("upload_error", detail=f"{filename} · {e}")
      yield ("done", None, f"❌ Écriture impossible dans le dossier de données ({DATA_DIR}) : {e}")
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
  if src.get("doc"):
    meta = f"{_escape_html(os.path.splitext(str(src['doc']))[0])} · {meta}"
  quote = _escape_html(str(src.get("excerpt") or "")).replace("$", "&#36;")
  return (f"<li><b>[{int(src.get('n', 0))}]</b> <span class='ak-src-meta'>{meta}</span>"
          f"<div class='ak-src-quote'>{quote}</div></li>")


def _fmt_dur(ms):
  """Durée lisible : « moins d'une seconde », « 4 s », « 1 min 12 s », « 1 h 5 min »."""
  sec = float(ms or 0) / 1000
  if sec < 1:
    return "moins d'une seconde"
  if sec < 60:
    return f"{sec:.0f} s"
  if sec < 3600:
    return f"{int(sec // 60)} min {int(sec % 60)} s"
  return f"{int(sec // 3600)} h {int(sec % 3600 // 60)} min"


def _time_html(item):
  """Heure d'un message : relative (« il y a 5 min », « il y a 2 h ») quand la date complète est connue."""
  label = _escape_html(item.get("t") or "")
  if not label and not item.get("ts"):
    return ""
  # le filtre HTML du chat retire les attributs data-* : l'horodatage passe par une classe (ts-AAAAMMJJTHHMMSS)
  stamp = re.sub(r"[^0-9T]", "", str(item.get("ts") or ""))[:15]
  ts = f" ts-{stamp}" if len(stamp) == 15 else ""
  timing = item.get("timing") or {}
  trace = item.get("trace") or {}
  bits = []
  if timing.get("total"):
    bits.append(("Réponse instantanée (déjà calculée)" if timing.get("cached") else f"Réponse en {_fmt_dur(timing['total'])}"))
  if trace.get("model"):
    bits.append(trace["model"] + (f", {trace['retries']} reprise(s)" if trace.get("retries") else ""))
  if trace.get("id"):
    bits.append(f"réf. {trace['id']}")
  tip = f" title='{_escape_html(' · '.join(bits))}'" if bits else ""
  return f"<span class='msg-time{ts}'{tip}>{label}</span>"


def _message_actions_html(index, item):
  """Barre d'actions sous une réponse : icônes (épingler, écouter, sources), nom au survol, puis l'heure."""
  sources = item.get("sources") or []
  buttons = [
      f"<button type='button' class='ak-act ak-a-pin ak-i-{index}' aria-label='Épingler comme chapitre'>📌 <span>Épingler comme chapitre</span></button>",
      "<button type='button' class='ak-act ak-a-speak' aria-label='Lire à voix haute'>🔊 <span>Lire à voix haute</span></button>",
  ]
  if sources:
    buttons.append(f"<button type='button' class='ak-act ak-a-sources' aria-label='Sources'>📎 <span>Sources ({len(sources)})</span></button>")
  html_out = "<div class='ak-msg-actions'>" + "".join(buttons) + _time_html(item) + "</div>"
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
      elif not streaming and (item.get("t") or item.get("ts")):
        content += "\n\n" + _time_html(item)
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
  label = f"{head} — {src['section']}" if src.get("section") else head
  return f"{os.path.splitext(src['doc'])[0]} — {label}" if src.get("doc") else label


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


def _stream_gemini(user_prompt, system_instruction, throttle=False, info=None):
  """Flux de texte Gemini. Tant qu'aucun mot n'est reçu : on passe aussitôt au modèle de secours suivant,
  puis on refait un tour complet (jusqu'à 3) après une courte pause, dans la limite du délai global.
  Un quota épuisé (429) ou un modèle inconnu (404) n'est pas réessayé en boucle."""
  _require_gemini_key()
  errors = []
  if info is not None:
    info["errors"] = errors     # traçabilité : modèles essayés et codes d'erreur
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
              if info is not None:
                info["model"] = model_target
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
  stamp = datetime.now()
  now, ts = stamp.strftime("%H:%M"), stamp.isoformat(timespec="seconds")
  historique.append({"role": "user", "content": message_str, "t": now, "ts": ts})
  historique.append({"role": "assistant", "content": "⏳ Génération…", "t": now, "ts": ts})
  if len(historique) > HISTORY_KEEP:
    del historique[:-HISTORY_KEEP]

  # IMPORTANT : aucune sauvegarde FAISS ici.
  return "", _chat_history_for_ui(doc_selectionne), message_str


def _multi_search(query, names, per_doc=3, total=None):
  """Recherche dans plusieurs cours (module ou espace entier) : on fusionne les meilleurs passages de chacun."""
  total = total or (RETRIEVAL_K + 2)
  merged = []
  for name in names:
    data = documents_db.get(name)
    if not data:
      continue
    try:
      scored = data["vector_db"].similarity_search_with_score(query, k=per_doc)
    except Exception:
      continue
    for doc, dist in scored:
      merged.append((doc, max(0.0, min(1.0, 1.0 - float(dist) / 2.0)), name))
  merged.sort(key=lambda x: -x[1])
  return merged[:total]


def repondre_akori_chat_stream(message_str, doc_selectionne, scope="doc"):
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

  rid = secrets.token_hex(3)   # référence de suivi de la question (journal, infobulle de l'heure)
  try:
    t_start = time.monotonic()
    item = historique[-1]
    item["streaming"] = True
    item.pop("sources", None)
    item.pop("timing", None)
    item["trace"] = {"id": rid}
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

    # Réponse déjà calculée (cache sur Drive) : instantanée et sans appel Gemini, même si l'API est saturée.
    scope_names = _scope_docs(doc_selectionne, scope)
    multi = len(scope_names) > 1
    cache_key = (f"{scope}:{_doc_module(doc_selectionne) if scope == 'module' else len(scope_names)}|" if multi else "") + _normalize_query(message_str)
    cached = (doc_data.get("answer_cache") or {}).get(cache_key)
    if cached and cached.get("content"):
      item["content"] = cached["content"]
      item["sources"] = list(cached.get("sources") or [])
      item["timing"] = {"search": 0, "first": 0, "total": round((time.monotonic() - t_start) * 1000), "cached": True}
      item.pop("streaming", None)
      session_usage["requests"] += 1
      _save_history(doc_selectionne)
      _log("chat_cached", detail=f"{doc_selectionne} · {len(item['content'])} car.", total_ms=item["timing"]["total"], rid=rid)
      _session("chat", doc_selectionne, scope=scope, cached=True, q=message_str[:160])
      yield _chat_history_for_ui(doc_selectionne), ""
      return

    best_similarity = None
    names_for = [doc_selectionne] * 0
    if _is_global_query(message_str):
      retrieved = [(doc, None) for doc in _global_docs(doc_data)]
    elif multi:
      merged = _multi_search(message_str, scope_names)
      retrieved = [(doc, sim) for doc, sim, _ in merged]
      names_for = [name for _, _, name in merged]
      best_similarity = max((sim for _, sim in retrieved), default=None)
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
           search_ms=round(t_search * 1000), similarity=round(best_similarity, 3), rid=rid)
      _session("chat", doc_selectionne, scope=scope, refused=True, q=message_str[:160])
      yield _chat_history_for_ui(doc_selectionne), ""
      return

    candidates = []
    for n, (doc, sim) in enumerate(retrieved, start=1):
      entry = _source_entry(n, doc, sim)
      if names_for:
        entry["doc"] = names_for[n - 1]
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
    if multi:
      system_instruction += " Le contexte vient de plusieurs cours : indique de quel cours vient chaque information."
    hint = _student_hint(doc_selectionne)
    if hint:
      system_instruction += " " + hint
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
    # Rafraîchir l'interface à chaque fragment renvoie tout l'historique au navigateur : on regroupe
    # les fragments (au moins 60 ms entre deux envois, 250 ms au plus), le premier mot part tout de suite.
    MIN_STREAM_CHARS = 24
    MIN_STREAM_DELAY = 0.06
    MAX_STREAM_DELAY = 0.25
    gen_info = {}

    try:
      for texte_chunk in _stream_gemini(user_prompt, system_instruction, info=gen_info):
        first = t_first is None
        if first:
          t_first = time.monotonic() - t_start
        reponse += texte_chunk
        buffer += texte_chunk
        now = time.monotonic()
        waited = now - last_ui_update

        if first or (len(buffer) >= MIN_STREAM_CHARS and waited >= MIN_STREAM_DELAY) or waited >= MAX_STREAM_DELAY:
          item["content"] = reponse
          buffer = ""
          last_ui_update = now
          # L'historique précédent reste affiché ; seul le dernier message évolue.
          yield _chat_history_for_ui(doc_selectionne), ""
    except RuntimeError as gemini_error:
      if reponse:
        raise
      # Gemini indisponible : on ne laisse pas l'étudiant sans rien, on affiche les passages du cours trouvés.
      _log("chat_fallback", detail=str(gemini_error)[:120], search_ms=round(t_search * 1000), rid=rid,
           tries=len(gen_info.get("errors") or []))
      intro = ("ℹ️ The AI is temporarily unavailable. Here are the passages of the course closest to your question:"
               if answer_lang == "en" else
               "ℹ️ L'IA est momentanément indisponible. Voici les passages du cours les plus proches de votre question :")
      parts = [intro, ""]
      for c in candidates[:3]:
        where = _source_label(c)
        parts.append(f"**{c['n']}. {where[:1].upper() + where[1:]}**  \n{c['excerpt']}")
        parts.append("")
      item["content"] = "\n".join(parts).strip()
      item["sources"] = []
      item.pop("streaming", None)
      _save_history(doc_selectionne)
      yield _chat_history_for_ui(doc_selectionne), ""
      return

    t_total = time.monotonic() - t_start
    # Sources : les passages cités par la réponse ([1], [2]…), sinon tous les passages utilisés.
    cited = {int(n) for n in re.findall(r"\[(\d{1,2})\]", reponse)}
    shown = [c for c in candidates if c["n"] in cited] or candidates
    item["sources"] = shown
    item["timing"] = {"search": round(t_search * 1000), "first": round((t_first or t_total) * 1000),
                      "total": round(t_total * 1000)}
    item["trace"] = {"id": rid, "model": gen_info.get("model", ""), "retries": len(gen_info.get("errors") or [])}
    item["content"] = reponse or "…"
    item.pop("streaming", None)
    if reponse and len(message_str.split()) >= 3:
      store = doc_data.setdefault("answer_cache", {})
      store[cache_key] = {"content": reponse, "sources": shown, "t": datetime.now().isoformat(timespec="seconds")}
      while len(store) > 150:
        store.pop(next(iter(store)))
    session_usage["output_tokens"] += _estimate_tokens(reponse)
    session_usage["requests"] += 1
    _save_history(doc_selectionne)
    _session("chat", doc_selectionne, scope=scope, ms=round(t_total * 1000), q=message_str[:160])
    _log("chat", detail=f"{doc_selectionne} · {len(reponse)} car. · {_fmt_ms(t_total * 1000)} · {gen_info.get('model', '?')}",
         search_ms=item["timing"]["search"], ttft_ms=item["timing"]["first"], total_ms=item["timing"]["total"],
         similarity=round(best_similarity, 3) if best_similarity is not None else None,
         rid=rid, model=gen_info.get("model", ""), retries=len(gen_info.get("errors") or []))

    # Dernier état garanti après la fin du flux.
    yield _chat_history_for_ui(doc_selectionne), ""

  except Exception as e:
    _log("error", detail=f"chat · {e}", rid=rid)
    last = historique[-1] if historique else None
    if isinstance(last, dict):
      last.pop("streaming", None)
      last["content"] = f"⚠️ {str(e)} (réf. {rid})"
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
      <button type='button' class='doc-del' title='Supprimer ce PDF' aria-label='Supprimer ce PDF'>🗑</button>
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


def delete_document(name, current=None):
    """Supprime un PDF de l'espace de l'utilisateur : index, métadonnées, historique, révisions et fichier d'origine."""
    name = str(name or "").strip()
    if not name or name not in documents_db:
        return "⚠️ Document introuvable.", gr.update(), gr.update(), gr.update(), ""
    try:
        folder = _doc_folder(name)
    except Exception:
        folder = None
    del documents_db[name]
    if folder:
        _io_drop(folder)
        shutil.rmtree(folder, ignore_errors=True)
    _log("doc_delete", detail=name)
    names = list(documents_db.keys())
    sel = current if current in documents_db else (names[-1] if names else None)
    return (f"✅ « {name} » a été supprimé.", gr.update(choices=names, value=sel), documents_html_v16(sel), modules_html(sel), "")


def documents_html_v12(active_doc=None):
    if not documents_db:
        return mascot_note("wave", "Aucun document pour le moment.", "Ajoutez votre premier PDF pour commencer.")
    return "<div class='folder-grid'>" + "".join(_course_card_html(n, d, n == active_doc) for n, d in documents_db.items()) + "</div>"


def documents_html_v16(active_doc=None):
    if not documents_db:
        return mascot_note("wave", "Aucun document pour le moment.", "Ajoutez votre premier PDF pour commencer.")
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
    greeting = f"Bonjour {_escape_html(_who)}" if _who else "Bonjour"
    cards = "".join(_course_card_html(n, d, n == active_doc) for n, d in documents_db.items())
    if not cards:
        cards = "<div class='empty-state'>📚 Aucun cours pour le moment.<br><span>Importez votre premier PDF dans 'Mes dossiers' pour commencer.</span></div>"

    active_chunks = int(active.get("chunks_count", 0)) if active else 0
    stats = _course_stats(active_doc) if active_doc in documents_db else None
    questions_here = stats["questions"] if stats else 0
    questions_all = sum(_course_stats(n)["questions"] for n in documents_db) if documents_db else 0
    pct = stats["percent"] if stats else 0
    # (onglet, icône, titre, description, étapes, astuce, bouton)
    cards_guide = [
        ("courses", "📁", "Dossiers & modules", "Importez vos PDF (même scannés) et rangez-les par module et par semestre.",
         ["Ajoutez un ou plusieurs PDF : AKORI lit le texte, lance l'OCR si la page est une image, puis indexe chaque page.",
          "Créez un module par matière (ex. Réseaux, Algorithmique) et choisissez son semestre.",
          "Rangez directement vos nouveaux fichiers dans un module au moment de l'import."],
         "Un semestre contient souvent plusieurs cours : un module les réunit pour les interroger ensemble.", "Importer un PDF"),
        ("summary", "📚", "Résumé", "Les points clés du cours en un coup d'œil, dans votre langue.",
         ["Choisissez le cours actif en haut de la page.", "Lancez « Générer le résumé » : le texte est conservé avec le cours.",
          "Relisez-le la veille d'un examen, ou au début d'une séance."],
         "Le résumé revient dans votre plan de révision comme première étape.", "Générer le résumé"),
        ("assistant", "✦", "Assistant IA", "Posez vos questions : la réponse s'appuie sur votre cours, avec la page citée.",
         ["Choisissez où chercher : ce cours, tout un module, ou tout votre espace.", "Lisez la réponse et ouvrez « Sources » pour voir les passages et les pages utilisés.",
          "Épinglez une réponse dans « À revoir », ou écoutez-la à voix haute."],
         "Si l'information n'est pas dans vos cours, AKORI le dit au lieu d'inventer.", "Poser une question"),
        ("flash", "🎰", "Flashcards", "Au moins 7 cartes en mode roulette pour mémoriser activement.",
         ["Générez les cartes à partir du cours actif.", "Cliquez pour stopper la roulette, essayez de répondre, puis affichez la réponse.",
          "Marquez « Je savais » ou « À revoir » : les cartes à revoir reviennent dans votre plan."],
         "Mieux vaut 10 minutes chaque jour qu'une heure la veille : c'est la répétition espacée.", "Lancer la roulette"),
        ("quiz", "🎯", "Quiz / QCM", "Testez-vous : correction expliquée, points forts et points faibles.",
         ["Générez un quiz de 5 questions.", "Touchez une réponse : la correction et l'explication s'affichent aussitôt, sans valider.",
          "Glissez vers la gauche pour passer à la question suivante. Vos erreurs alimentent la carte de vos lacunes."],
         "Un quiz raté n'est pas un échec : il indique exactement quelle page relire.", "Passer un quiz"),
        ("oral", "🎤", "Colle orale", "Répondez à l'oral ou à l'écrit, comme devant un examinateur.",
         ["AKORI choisit un passage de votre cours et pose une question ouverte.", "Dictez votre réponse au micro ou écrivez-la.",
          "Recevez une note sur 10, ce qui manquait, et la page à relire."],
         "Dire la réponse à voix haute révèle ce que vous maîtrisez vraiment.", "Passer une colle"),
        ("mindmap", "🕸", "Mindmap", "La carte mentale du cours, d'un seul regard.",
         ["Choisissez le cours actif puis lancez « Générer la mindmap ».", "Le thème est au centre, les grandes notions autour, leurs détails au bout des branches.",
          "Cliquez sur une branche pour la replier ou la déplier et réviser bloc par bloc."],
         "La carte est conservée avec le cours : vous la retrouvez à chaque visite.", "Ouvrir la mindmap"),
        ("plan", "📅", "Plan de révision", "Un assistant en 5 étapes construit votre emploi du temps de révision.",
         ["Contexte et calendrier : examen, contrôle continu ou apprentissage libre, la date et le temps par jour.",
          "Matières et rythme : choisissez vos PDF, l'intensité, les jours de repos et les fonctions IA.",
          "L'emploi du temps s'affiche semaine par semaine : touchez une séance pour voir son détail et la lancer."],
         "Un jour de repos est prévu : la mémoire consolide aussi quand vous ne travaillez pas.", "Créer mon plan"),
        ("reviewq", "📌", "À revoir", "Retrouvez les réponses épinglées, cours par cours.",
         ["Épinglez une réponse de l'assistant depuis le chat.", "Retrouvez-la ici, rangée par cours.", "Relisez-la ou écoutez-la avant l'examen."],
         "C'est votre fiche de révision personnelle, écrite avec vos propres questions.", "Ouvrir À revoir"),
        ("progress", "📈", "Progression", "Un état clair : flashcards, quiz et colle orale, cours par cours.",
         ["Consultez la vue globale, puis le détail du cours choisi.", "Chaque pourcentage précise sur quoi il porte (cartes maîtrisées, quiz terminés…).",
          "Repérez les notions à renforcer."],
         "La progression d'un cours est la moyenne de ses trois activités.", "Voir ma progression"),
        ("history", "🕑", "Historique", "Votre journal de révision et la carte de vos lacunes.",
         ["Chaque séance est notée : quiz, flashcards, colle orale, questions.", "Suivez votre série de jours et vos notions les moins réussies.",
          "L'assistant adapte ses explications à votre niveau et à vos résultats."],
         "Plus vous révisez, plus les réponses s'ajustent à vous.", "Ouvrir l'historique"),
    ]
    guide = "".join(
        f"<details class='guide-card rv' style='--i:{i}'><summary><span class='gc-ico'>{ico}</span>"
        f"<span class='gc-body'><b>{title}</b><span>{desc}</span></span><span class='gc-chev'></span></summary>"
        f"<div class='gc-detail'><ol>{''.join(f'<li>{st}</li>' for st in steps)}</ol>"
        f"<p class='gc-tip'><span class='gc-tipico'>💡</span><span>{tip}</span></p>"
        f"<button type='button' class='ak-act ak-go ak-t-{t}'>▶ <span>{cta}</span></button></div></details>"
        for i, (t, ico, title, desc, steps, tip, cta) in enumerate(cards_guide))

    art = mascot_svg("wave" if not documents_db else ("celebrate" if pct >= 80 else ("happy" if pct >= 40 else "encourage")), "hero-mascot")

    slogan = "Les réponses viennent de vos cours, et chacune indique sa page."
    return f"""
    <div class='home-fit'>
    <div class='hero hero-compact'>
      <div class='hero-glow'></div>
      <div class='hero-text'>
        <div class='eyebrow'>ASSISTANT KNOWLEDGE ORGANIZED TO REVISE INTELLIGENTLY</div>
        <h1>{greeting}<br><span class='hero-grad'>Que révisez-vous aujourd'hui ?</span></h1>
        <p class='home-slogan'>Cours actif : <b>{_escape_html(active_title if len(active_title) <= 70 else active_title[:67] + '…')}</b></p>
      </div>
      <div class='hero-visual'>{art}</div>
    </div>
    <div class='stat-row'>
      <div class='mini-stat'><b class='cu'>{total_docs}</b><span>Cours importés</span></div>
      <div class='mini-stat'><b class='cu'>{active_chunks}</b><span>Passages indexés (cours actif)</span></div>
      <div class='mini-stat'><b class='cu'>{questions_here}</b><span>Questions posées (cours actif)</span></div>
      <div class='mini-stat'><b class='cu'>{pct}%</b><span>Progression du cours actif</span></div>
    </div>
    <div class='guide-box'>
      <h3>Que voulez-vous faire ? <small>Touchez une fonction pour voir comment elle marche.</small></h3>
      <div class='guide-grid'>{guide}</div>
    </div>
    <div class='home-foot'><img src='{LOGO_DATA_URI}' alt='AKORI'><span>{slogan}</span></div>
    </div>
    """

HOME_SLOGAN = "Les réponses viennent de vos cours, et chacune indique sa page."
EDITOR_NAME = os.getenv("AKORI_EDITOR", "Johana")
APP_VERSION = "1.0"
LAND_ICONS = {
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "cal": '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/>',
    "chat": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "dice": '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1"/><circle cx="15.5" cy="15.5" r="1"/><circle cx="12" cy="12" r="1"/>',
    "check": '<polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>',
    "mic": '<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8"/>',
    "mind": '<circle cx="5" cy="12" r="2.6"/><circle cx="19" cy="5" r="2.6"/><circle cx="19" cy="19" r="2.6"/><path d="M7.5 11l9-5M7.5 13l9 5"/>',
    "folder": '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>',
    "book": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V2H6.5A2.5 2.5 0 0 0 4 4.5z"/><path d="M4 19.5V22h16"/>',
    "trend": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/><line x1="7" y1="12" x2="17" y2="12"/>',
}
LANDING_TEMPLATE = r"""<div class='ld'>
<header class='ld-nav'>
  <div class='ld-brand'><img src='__LOGO__' alt='AKORI'><b>AKORI</b></div>
  <nav class='ld-links'><a href='#ld-features'>Fonctionnalités</a><a href='#ld-how'>Comment ça marche</a><a href='#ld-engine'>Sous le capot</a><a href='#ld-faq'>FAQ</a></nav>
  <button type='button' class='ld-btn ld-btn-ghost ld-login'>Se connecter</button>
</header>

<section class='ld-hero'>
  <div class='ld-orb o1'></div><div class='ld-orb o2'></div><div class='ld-grid'></div>
  <div class='ld-pill lr'><span class='ld-dot'></span> Nouveau : plan de révision et mindmap</div>
  <h1 class='lr'>Révisez ce que dit<br><span class='ld-grad'>votre cours.</span> Rien d'autre.</h1>
  <p class='ld-sub lr'>AKORI transforme vos PDF en un espace de révision complet : un assistant qui cite la page, des flashcards, des quiz, des colles orales, des cartes mentales et un plan jour par jour jusqu'à l'examen.</p>
  <div class='ld-cta lr'>
    <button type='button' class='ld-btn ld-btn-primary ld-login'>Commencer gratuitement <span>→</span></button>
    <a class='ld-btn ld-btn-ghost' href='#ld-how'>Voir comment ça marche</a>
  </div>
  <div class='ld-trust lr'><span>Réponses fondées sur vos PDF</span><span>Pages citées</span><span>PDF scannés (OCR)</span><span>Français et English</span></div>

  <div class='ld-mock lr'>
    <div class='ld-win'>
      <div class='ld-bar'><i></i><i></i><i></i><span>AKORI · Réseaux informatiques.pdf</span></div>
      <div class='ld-body'>
        <div class='ld-side'><b></b><b class='on'></b><b></b><b></b><b></b></div>
        <div class='ld-chat'>
          <div class='ld-msg me'>C'est quoi la différence entre TCP et UDP ?</div>
          <div class='ld-msg ai'>TCP établit une connexion fiable (accusés de réception, retransmission) alors qu'UDP envoie sans garantie, mais plus vite [1].
            <div class='ld-src'><span>p. 14 · Couche transport</span><span>p. 15</span></div></div>
          <div class='ld-chips'><span>Épingler</span><span>Écouter</span><span>Sources (2)</span></div>
        </div>
        <div class='ld-cards'>
          <div class='ld-mini'><em>Flashcards</em><b>7 cartes</b><div class='ld-roll'><i></i><i></i><i></i></div></div>
          <div class='ld-mini'><em>Quiz</em><b>4 / 5</b><div class='ld-bar2'><i style='width:80%'></i></div></div>
          <div class='ld-mini'><em>Plan de révision</em><b>J-9</b><div class='ld-bar2'><i style='width:42%'></i></div></div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class='ld-strip'><div class='ld-marquee'><div>
  <span>Assistant avec sources</span><span>Flashcards roulette</span><span>Quiz instantané</span><span>Colle orale</span><span>Mindmap</span><span>Plan de révision</span><span>Résumé</span><span>Modules par semestre</span><span>OCR des scans</span><span>Historique et lacunes</span>
  <span>Assistant avec sources</span><span>Flashcards roulette</span><span>Quiz instantané</span><span>Colle orale</span><span>Mindmap</span><span>Plan de révision</span><span>Résumé</span><span>Modules par semestre</span><span>OCR des scans</span><span>Historique et lacunes</span>
</div></div></section>

<section class='ld-sec'>
  <div class='ld-head lr'><span class='ld-kicker'>Pourquoi AKORI</span><h2>Un assistant qui reste dans votre cours</h2><p>Les assistants généralistes inventent parfois. AKORI ne répond qu'avec ce qui figure dans vos supports, et vous montre d'où vient chaque réponse.</p></div>
  <div class='ld-three'>
    <div class='ld-card lr'><div class='ld-ico'>__I_target__</div><h3>Fidèle au support</h3><p>Chaque réponse est construite à partir des passages retrouvés dans votre PDF. Si l'information n'y est pas, AKORI le dit au lieu d'inventer.</p></div>
    <div class='ld-card lr'><div class='ld-ico'>__I_layers__</div><h3>Mémorisation active</h3><p>Flashcards roulette, quiz corrigé au toucher, colle orale : vous retrouvez l'information au lieu de simplement la relire.</p></div>
    <div class='ld-card lr'><div class='ld-ico'>__I_cal__</div><h3>Un plan qui s'adapte</h3><p>Donnez la date de l'examen : AKORI répartit les séances, rapproche vos points faibles et vous dit quelles pages relire.</p></div>
  </div>
</section>

<section class='ld-sec' id='ld-features'>
  <div class='ld-head lr'><span class='ld-kicker'>Fonctionnalités</span><h2>Tout ce qu'il faut pour réviser, au même endroit</h2></div>
  <div class='ld-bento'>
    <div class='ld-b big lr'><div class='ld-ico'>__I_chat__</div><h3>Assistant IA avec sources</h3><p>Posez vos questions en langage naturel. La réponse cite les pages utilisées, vous pouvez l'épingler dans « À revoir » ou l'écouter à voix haute. Cherchez dans un cours, un module entier ou tout votre espace.</p></div>
    <div class='ld-b lr'><div class='ld-ico'>__I_dice__</div><h3>Flashcards roulette</h3><p>Des séries d'au moins 7 cartes, tirées au hasard comme à la roulette.</p></div>
    <div class='ld-b lr'><div class='ld-ico'>__I_check__</div><h3>Quiz instantané</h3><p>Touchez une réponse : la correction apparaît aussitôt. Glissez pour passer à la suite.</p></div>
    <div class='ld-b lr'><div class='ld-ico'>__I_mic__</div><h3>Colle orale</h3><p>Répondez à voix haute ou à l'écrit. Note sur 10 et page à relire.</p></div>
    <div class='ld-b lr'><div class='ld-ico'>__I_mind__</div><h3>Mindmap</h3><p>Une carte mentale du cours, générée à partir de vos pages, à déplier branche par branche.</p></div>
    <div class='ld-b big lr'><div class='ld-ico'>__I_cal__</div><h3>Plan de révision jour par jour</h3><p>Date d'examen, temps disponible, portée du plan : AKORI combine vos points faibles et une répétition espacée (1, 3, 7 puis 14 jours), puis recalcule le plan après chaque séance.</p></div>
    <div class='ld-b lr'><div class='ld-ico'>__I_folder__</div><h3>Modules et semestres</h3><p>Réunissez plusieurs cours (Systèmes, Réseaux, Algorithmique) et interrogez-les ensemble.</p></div>
    <div class='ld-b lr'><div class='ld-ico'>__I_book__</div><h3>Résumé</h3><p>Les points clés du cours, dans votre langue.</p></div>
    <div class='ld-b lr'><div class='ld-ico'>__I_trend__</div><h3>Progression claire</h3><p>Chaque pourcentage dit sur quoi il porte, cours par cours.</p></div>
    <div class='ld-b lr'><div class='ld-ico'>__I_clock__</div><h3>Historique et lacunes</h3><p>AKORI cartographie vos notions fragiles et adapte le niveau de ses explications.</p></div>
    <div class='ld-b lr'><div class='ld-ico'>__I_scan__</div><h3>PDF scannés (OCR)</h3><p>Même une photo de cours devient interrogeable.</p></div>
  </div>
</section>

<section class='ld-sec' id='ld-how'>
  <div class='ld-head lr'><span class='ld-kicker'>Comment ça marche</span><h2>De vos PDF à l'examen en trois temps</h2></div>
  <div class='ld-steps'>
    <div class='ld-step lr'><span>1</span><h3>Importez</h3><p>Déposez un ou plusieurs PDF, même scannés. AKORI lit, découpe et indexe chaque page, puis les range par module et par semestre.</p></div>
    <div class='ld-step lr'><span>2</span><h3>Révisez</h3><p>Posez des questions, jouez les flashcards, passez un quiz, lancez une colle orale ou ouvrez la mindmap. Tout s'appuie sur votre cours.</p></div>
    <div class='ld-step lr'><span>3</span><h3>Suivez</h3><p>Votre plan se recalcule, votre journal note chaque séance et la carte de vos lacunes montre quoi relire avant le jour J.</p></div>
  </div>
</section>

<section class='ld-sec ld-split'>
  <div class='ld-text lr'><span class='ld-kicker'>Plan de révision</span><h2>Chaque jour, quoi faire et quelles pages relire</h2>
    <p>Fini les « je commence quand ? ». Le plan place d'abord ce qui est faible, espace ce qui est acquis et prévoit un jour de repos. Une mauvaise séance rapproche la prochaine ; une bonne la repousse.</p>
    <ul><li>Pages à relire tirées de vos erreurs de quiz et de vos cartes « À revoir »</li><li>Répétition espacée à 1, 3, 7 puis 14 jours</li><li>Recalcul automatique après chaque séance</li></ul></div>
  <div class='ld-demo lr'>
    <div class='ld-day today'><b>Aujourd'hui</b><span>22 min</span><p>Quiz pour mesurer ta base</p><div class='ld-pg'><i>p. 12</i><i>p. 18 · Routage</i></div></div>
    <div class='ld-day'><b>Demain</b><span>18 min</span><p>Flashcards à revoir</p></div>
    <div class='ld-day'><b>Dimanche 25/10</b><span>24 min</span><p>Colle orale</p><div class='ld-pg'><i>p. 7</i></div></div>
    <div class='ld-day rest'><b>Lundi 26/10</b><span>repos</span><p>La mémoire consolide</p></div>
  </div>
</section>

<section class='ld-sec ld-split rev'>
  <div class='ld-demo lr'>
    <svg viewBox='0 0 460 250' class='ld-mm'>
      <path d='M110 125C160 125 150 50 200 50' stroke='#4f6df5'/><path d='M110 125C160 125 150 125 200 125' stroke='#e86fb0'/><path d='M110 125C160 125 150 200 200 200' stroke='#10b981'/>
      <path d='M290 50C320 50 320 25 350 25' stroke='#4f6df5'/><path d='M290 50C320 50 320 75 350 75' stroke='#4f6df5'/><path d='M290 125C320 125 320 125 350 125' stroke='#e86fb0'/><path d='M290 200C320 200 320 185 350 185' stroke='#10b981'/><path d='M290 200C320 200 320 220 350 220' stroke='#10b981'/>
      <rect x='8' y='103' width='104' height='44' rx='22' fill='#4f6df5'/><text x='60' y='130' text-anchor='middle'>Réseaux</text>
      <rect x='200' y='34' width='90' height='32' rx='16' fill='#4f6df5'/><text x='245' y='55' text-anchor='middle'>Modèle OSI</text>
      <rect x='200' y='109' width='90' height='32' rx='16' fill='#e86fb0'/><text x='245' y='130' text-anchor='middle'>Transport</text>
      <rect x='200' y='184' width='90' height='32' rx='16' fill='#10b981'/><text x='245' y='205' text-anchor='middle'>Routage</text>
      <g class='leaf'><rect x='350' y='10' width='90' height='28' rx='14'/><text x='395' y='29' text-anchor='middle'>7 couches</text><rect x='350' y='60' width='90' height='28' rx='14'/><text x='395' y='79' text-anchor='middle'>Encapsul.</text>
      <rect x='350' y='111' width='90' height='28' rx='14'/><text x='395' y='130' text-anchor='middle'>TCP / UDP</text><rect x='350' y='171' width='90' height='28' rx='14'/><text x='395' y='190' text-anchor='middle'>IP, RIP</text><rect x='350' y='206' width='90' height='28' rx='14'/><text x='395' y='225' text-anchor='middle'>OSPF</text></g>
    </svg>
  </div>
  <div class='ld-text lr'><span class='ld-kicker'>Mindmap</span><h2>Voyez le cours d'un seul regard</h2>
    <p>AKORI dessine la carte mentale de votre document : le thème au centre, les grandes notions autour, leurs détails au bout. Dépliez ou repliez chaque branche pour réviser par blocs.</p>
    <ul><li>Générée uniquement à partir de vos pages</li><li>Branches colorées, repliables</li><li>Conservée avec le cours</li></ul></div>
</section>

<section class='ld-sec' id='ld-engine'>
  <div class='ld-head lr'><span class='ld-kicker'>Sous le capot</span><h2>Une recherche sérieuse avant chaque réponse</h2><p>AKORI ne devine pas : il cherche d'abord dans votre cours, puis rédige.</p></div>
  <div class='ld-flow lr'><span>PDF</span><i></i><span>OCR si besoin</span><i></i><span>Découpage par page</span><i></i><span>Embeddings BGE-M3</span><i></i><span>Index FAISS</span><i></i><span>Gemini</span><i></i><span class='end'>Réponse + pages</span></div>
  <div class='ld-three'>
    <div class='ld-card lr'><h3>Seuil de pertinence</h3><p>Si aucun passage n'est assez proche de la question, AKORI refuse de répondre plutôt que d'inventer.</p></div>
    <div class='ld-card lr'><h3>Sources vérifiables</h3><p>Les passages et les pages utilisés sont affichés sous chaque réponse.</p></div>
    <div class='ld-card lr'><h3>Vos données, à vous</h3><p>Chaque compte a son espace isolé. Vos documents ne sont visibles que de vous.</p></div>
  </div>
</section>

<section class='ld-sec'>
  <div class='ld-head lr'><span class='ld-kicker'>Pour qui</span><h2>Du collège aux études supérieures</h2></div>
  <div class='ld-three'>
    <div class='ld-card lr'><h3>Collège</h3><p>Des explications simples, pas à pas, avec des exemples tirés du cours.</p></div>
    <div class='ld-card lr'><h3>Lycée</h3><p>Des quiz et des colles orales pour tenir le rythme jusqu'au bac.</p></div>
    <div class='ld-card lr'><h3>Études supérieures</h3><p>Plusieurs unités d'enseignement dans un même espace, un plan par semestre.</p></div>
  </div>
</section>

<section class='ld-sec' id='ld-faq'>
  <div class='ld-head lr'><span class='ld-kicker'>Questions fréquentes</span><h2>Bon à savoir</h2></div>
  <div class='ld-faq lr'>
    <details><summary>Mes PDF sont-ils privés ?</summary><p>Oui. Chaque compte dispose de son propre espace, séparé des autres. Seuls la question et les passages retenus sont transmis au modèle pour rédiger la réponse.</p></details>
    <details><summary>Et si la réponse n'est pas dans mon cours ?</summary><p>AKORI vous le dit clairement. Il ne complète pas avec des connaissances extérieures.</p></details>
    <details><summary>Mes cours sont des scans ou des photos. Est-ce que ça marche ?</summary><p>Oui, une reconnaissance de texte (OCR) lit les pages qui sont des images.</p></details>
    <details><summary>Puis-je interroger plusieurs cours à la fois ?</summary><p>Oui. Rangez vos cours dans des modules, puis choisissez « Mon module » ou « Tout mon espace » dans l'assistant.</p></details>
    <details><summary>Faut-il payer ?</summary><p>La création de compte est gratuite.</p></details>
  </div>
</section>

<section class='ld-final lr'>
  <img src='__LOGO__' alt='AKORI'>
  <h2>Essayez avec votre prochain cours</h2>
  <p>Créez votre compte et importez votre premier cours en moins d'une minute.</p>
  <button type='button' class='ld-btn ld-btn-primary ld-login'>Commencer maintenant <span>→</span></button>
</section>

<footer class='ld-foot'>
  <p class='ld-slogan'>__SLOGAN__</p>
  <div><b>AKORI</b> · Assistant Knowledge Organized to Revise Intelligently</div>
  <small>Édité par __EDITOR__</small>
</footer>
</div>
"""


def landing_html():
    """Page de présentation affichée avant la connexion."""
    out = LANDING_TEMPLATE.replace("__LOGO__", LOGO_DATA_URI).replace("__SLOGAN__", HOME_SLOGAN).replace("__EDITOR__", _escape_html(EDITOR_NAME))
    for name, body in LAND_ICONS.items():
        out = out.replace(f"__I_{name}__", f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')
    return out


def about_html():
    return f"""
    <div class='about-card'>
      <div class='about-head'><img src='{LOGO_DATA_URI}' alt='AKORI'><div><b>AKORI</b><span>Assistant Knowledge Organized to Revise Intelligently · version {APP_VERSION}</span></div></div>
      <p>AKORI est un assistant de révision qui part de vos propres cours. Vous importez vos PDF, même scannés ; AKORI les lit, les indexe page par page, puis vous aide à réviser : assistant qui cite ses sources, flashcards, quiz, colle orale, mindmap, résumé et plan de révision jusqu'à l'examen.</p>
      <div class='about-chips'><span>Recherche FAISS</span><span>Embeddings BGE-M3</span><span>Gemini</span><span>OCR</span><span>Gradio</span></div>
      <p class='about-note'>Principe : réviser, c'est reprendre, comprendre, mémoriser et maîtriser. AKORI ne répond qu'avec le contenu de votre support, et indique la page à relire.</p>
    </div>
    <div class='about-card about-editor'>
      <div class='about-av'>{_escape_html(EDITOR_NAME[:1].upper() or 'A')}</div>
      <div><b>{_escape_html(EDITOR_NAME)}</b><span>Éditeur et développeur d'AKORI</span>
      <p>AKORI est né d'un constat simple : on révise mieux quand l'outil parle comme le cours. Il est conçu, développé et amélioré en continu par {_escape_html(EDITOR_NAME)}, avec l'aide de ses utilisateurs.</p></div>
    </div>
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
      <div class='course-overview-meta'>{int(data.get('chunks_count',0))} passages indexés · Progression du cours {progress}%</div>
      <div class='course-overview-actions'>Résumé · Flashcards · Quiz / QCM · Colle orale · Assistant IA</div>
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
# PLAN DE RÉVISION INTELLIGENT + JOURNAL DE RÉVISION
# ============================================================
from datetime import date as _date, timedelta as _timedelta

SPACED_INTERVALS = [1, 3, 7, 14]    # répétition espacée (boîtes de Leitner, en jours)
PLAN_KIND_TAB = {"summary": "summary", "flash": "flash", "quiz": "quiz", "oral": "oral", "final": "oral"}


def _locate(doc_name, text):
    """Page et section du passage du cours le plus proche d'un texte (question, notion, point oublié)."""
    doc = documents_db.get(doc_name)
    if not doc or not str(text).strip():
        return None, ""
    cache = doc.setdefault("_page_cache", {})
    if text in cache:
        return cache[text]
    try:
        found = doc["vector_db"].similarity_search_with_score(str(text)[:300], k=1)
        meta = (found[0][0].metadata or {}) if found else {}
        cache[text] = (meta.get("page"), meta.get("section") or "")
    except Exception:
        cache[text] = (None, "")
    return cache[text]


def _weak_pages(doc_name, limit=12):
    """Pages à relire : erreurs de quiz, flashcards « à revoir » et points oubliés en colle orale, regroupés par page."""
    prog = _progress_for(doc_name)
    items = []
    for attempt in prog.get("quiz_attempts", [])[-3:]:
        for it in attempt.get("items", []):
            if not it.get("correct") and it.get("question"):
                items.append({"text": it["question"], "reason": f"quiz · {it.get('topic', 'Général')}", "weight": 2})
    for question, state in prog.get("flashcards", {}).items():
        if state == "review":
            items.append({"text": question, "reason": "flashcard à revoir", "weight": 3})
    for attempt in prog.get("oral_attempts", [])[-3:]:
        if float(attempt.get("score", 10)) < 8:
            for missed in attempt.get("missed", []):
                items.append({"text": missed, "reason": "colle orale", "weight": 2, "page": attempt.get("page"),
                              "section": attempt.get("section", "")})
    agg = {}
    for it in items[:36]:
        page, section = it.get("page"), it.get("section", "")
        if not page:
            page, section = _locate(doc_name, it["text"])
        if not page:
            continue
        entry = agg.setdefault(page, {"page": page, "section": section, "weight": 0, "reasons": []})
        entry["weight"] += it["weight"]
        if it["reason"] not in entry["reasons"]:
            entry["reasons"].append(it["reason"])
    return sorted(agg.values(), key=lambda e: (-e["weight"], e["page"]))[:limit]


def _parse_day(value):
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


PLAN_INTENSITY = {   # rythme choisi à l'étape 4 : durée des séances et écart entre deux répétitions
    "low": {"mult": 0.8, "intervals": [2, 5, 10, 20], "label": "Faible"},
    "mid": {"mult": 1.0, "intervals": [1, 3, 7, 14], "label": "Moyenne"},
    "high": {"mult": 1.2, "intervals": [1, 2, 4, 8], "label": "Élevée"},
}
PLAN_CONTEXTS = {"exam": "Examen", "cc": "Contrôle continu", "free": "Apprentissage libre"}
PLAN_ACTS = ["notes", "flash", "quiz", "oral"]
PLAN_AI = ["spaced", "prio", "alerts"]
PLAN_FREE_DAYS = 28            # apprentissage libre sans date : plan sur 4 semaines
PLAN_PAUSE = 5                 # minutes de pause entre deux séances d'une même journée


def build_plan(exam_day, names, minutes, today=None, opts=None):
    """Plan jour par jour : points faibles d'abord, puis activités espacées selon ce que l'étudiant a déjà fait.

    Chaque activité revient à intervalles croissants (selon le rythme choisi) ; une séance réalisée repousse la suivante,
    un mauvais résultat la rapproche. Les jours de repos ne reçoivent aucune séance. Le plan est recalculé à chaque
    ouverture, donc après chaque séance. `exam_day=None` : apprentissage libre sur PLAN_FREE_DAYS jours."""
    opts = opts or {}
    today = today or _date.today()
    acts = set(opts.get("acts") or PLAN_ACTS)
    ai = set(PLAN_AI if opts.get("ai") is None else opts.get("ai"))
    spaced, prio, final_review = "spaced" in ai, "prio" in ai, opts.get("final", True)
    rhythm = PLAN_INTENSITY.get(opts.get("intensity") or "mid", PLAN_INTENSITY["mid"])
    intervals = rhythm["intervals"]
    rest = {int(x) for x in (opts.get("rest") or []) if str(x).isdigit()}
    if len(rest) >= 7:
        rest = set()
    if exam_day is None:
        days_left, horizon, final_review = PLAN_FREE_DAYS, PLAN_FREE_DAYS, False
    else:
        days_left = (exam_day - today).days
        if days_left < 0:
            return None
        horizon = max(1, min(days_left, 60))                # jours de révision avant l'examen
    if not names:
        return None

    def is_rest(o):
        return (today + _timedelta(days=o)).weekday() in rest

    work_days = [o for o in range(horizon) if not is_rest(o)] or list(range(horizon))
    last_work = work_days[-1]

    def place(o):
        """Jour de travail le plus proche à partir de `o` (sinon le dernier jour de travail)."""
        o = max(0, o)
        return next((d for d in work_days if d >= o), last_work)

    def scaled(m):
        return max(5, int(round(m * rhythm["mult"])))

    sessions = _read_sessions()
    last_done, recent = {}, {}
    for rec in sessions:
        d = _parse_day(rec.get("t"))
        if not d or rec.get("k") not in ("flash", "quiz", "oral", "summary"):
            continue
        key = (rec.get("doc", ""), rec["k"])
        last_done[key] = max(last_done.get(key, d), d)
        if (today - d).days <= 14:
            recent.setdefault(key, set()).add(d)

    def occurrences(first, box):
        out, day, i = [], first, 0
        while day < horizon:
            out.append(day)
            if not spaced:
                break
            i += 1
            day += intervals[min(box + i, len(intervals) - 1)] if horizon > 4 else horizon
        return out or [min(first, horizon - 1)]

    def first_day(key, offset):
        last = last_done.get(key)
        box = min(len(recent.get(key, ())), 3)
        if not last:
            return min(offset, horizon - 1), 0
        return max(0, (last - today).days + (intervals[box] if spaced else 1)), box

    if prio:   # Priorisation IA : les cours les moins prêts passent en premier
        names = sorted(names, key=lambda n: _course_stats(n)["percent"])
    final_label = "Révision avant le contrôle" if opts.get("context") == "cc" else "Revue finale"
    tasks = []
    for order, name in enumerate(names):
        cs = _course_stats(name)
        pages = _weak_pages(name) if ("notes" in acts or final_review) else []
        if "notes" in acts:
            if not cs["summary"]:
                tasks.append({"kind": "summary", "doc": name, "label": "Lire le résumé du cours", "pages": [], "minutes": scaled(10),
                              "day": min(order, horizon - 1), "prio": 5,
                              "why": "Le résumé n'a pas encore été généré : c'est la base avant de s'entraîner."})
            for i in range(0, len(pages), 3):
                group = pages[i:i + 3]
                reasons = ", ".join(dict.fromkeys(r for pg in group for r in pg["reasons"]))
                why = f"Pages repérées grâce à vos erreurs ({reasons})." if reasons else "Pages repérées grâce à vos erreurs."
                tasks.append({"kind": "reread", "doc": name, "label": "Relire les pages faibles", "pages": group,
                              "minutes": scaled(6 * len(group)), "day": min(order + i // 3, horizon - 1),
                              "prio": (10 - i // 3) if prio else 5, "why": why})
                if horizon > 6 and spaced:
                    tasks.append({"kind": "reread", "doc": name, "label": "Relire les pages faibles (rappel)", "pages": group,
                                  "minutes": scaled(4 * len(group)), "day": min(order + i // 3 + intervals[1], horizon - 1), "prio": 4,
                                  "why": f"Répétition espacée : rappel {intervals[1]} jour(s) après la première relecture."})
        specs = [
            ("flash", cs["flash_answered"] == 0, "Générer puis faire une série de flashcards", cs["flash_review"] > 0,
             f"Refaire les {cs['flash_review']} flashcard(s) « à revoir »", "Entretenir tes flashcards", 12, 1),
            ("quiz", cs["quizzes"] == 0, "Faire un quiz pour mesurer ta base", 0 < cs["quizzes"] and cs["quiz_pct"] < 70,
             f"Refaire un quiz (moyenne actuelle {cs['quiz_pct']} %)", "Faire un quiz d'entretien", 12, 2),
            ("oral", cs["orals"] == 0, "Passer une première colle orale", 0 < cs["orals"] and cs["oral_avg"] < 7,
             f"Repasser une colle orale (moyenne {cs['oral_avg']:.1f}/10)", "Passer une colle orale d'entretien", 10, 3),
        ]
        for kind, never, never_label, weak, weak_label, keep_label, mins, offset in specs:
            if kind not in acts:
                continue
            start, box = first_day((name, kind), offset + order)
            label = never_label if never else (weak_label if weak else keep_label)
            if not never and not weak and last_done.get((name, kind)) is None:
                continue
            for k, day in enumerate(occurrences(start, box)):
                if k == 0:
                    why = ("Première séance : elle mesure votre niveau de départ." if never else
                           "Résultats encore fragiles : la séance revient plus tôt." if weak else
                           "Répétition espacée : on revient sur ce cours avant de l'oublier.")
                else:
                    why = f"Répétition espacée : rappel n° {k} pour ancrer la mémoire à long terme."
                tasks.append({"kind": kind, "doc": name, "label": label if k == 0 else (weak_label if weak else keep_label),
                              "pages": [], "minutes": scaled(mins if k == 0 else max(6, mins - 4)), "day": day,
                              "prio": (8 - offset - (0 if weak or never else 3)) if prio else 4, "why": why})
        if final_review and horizon >= 2:
            tasks.append({"kind": "final", "doc": name, "label": f"{final_label} : résumé, flashcards à revoir et une colle rapide",
                          "pages": pages[:3], "minutes": scaled(15), "day": last_work, "prio": 9,
                          "why": "Dernière séance avant le jour J : on repasse uniquement sur l'essentiel."})

    days = [[] for _ in range(horizon)]
    load = [0] * horizon
    for task in sorted(tasks, key=lambda t: (t["day"], -t["prio"])):
        d = place(task["day"]) if task["kind"] != "final" else last_work
        while task["kind"] != "final" and load[d] > 0 and load[d] + task["minutes"] > minutes:
            nxt = next((x for x in work_days if x > d), None)
            if nxt is None:
                break
            d = nxt
        days[d].append(task)
        load[d] += task["minutes"]
    done_today = {(r.get("doc", ""), r["k"]) for r in sessions if r.get("k") in ("flash", "quiz", "oral", "summary") and _parse_day(r.get("t")) == today}
    for task in days[0]:
        task["done"] = (task["doc"], task["kind"]) in done_today
    alerts = []
    if "alerts" in ai:
        over = [o for o in range(horizon) if load[o] > minutes and len(days[o]) > 1]
        if over:
            first = today + _timedelta(days=over[0])
            alerts.append(f"Surcharge : {len(over)} jour(s) dépassent vos {minutes} min, dès le "
                          f"{PLAN_DAYS[first.weekday()]} {first.strftime('%d/%m')}. Augmentez le temps par jour ou retirez un jour de repos.")
        total, capacity = sum(load), minutes * len(work_days)
        if capacity and total > capacity:
            alerts.append(f"Le plan demande {total} min pour {capacity} min disponibles : choisissez un rythme plus faible ou moins de matières.")
        if exam_day is not None and 0 < days_left <= 3 and len(names) > 2:
            alerts.append(f"Examen dans {days_left} jour(s) pour {len(names)} matières : concentrez-vous sur les pages faibles.")
    return {"days": days, "load": load, "days_left": days_left, "horizon": horizon, "today": today, "rest": rest,
            "alerts": alerts, "exam": exam_day, "cap": minutes, "names": names}


PLAN_KIND_ICON = {"summary": "📚", "reread": "📖", "flash": "🎰", "quiz": "🎯", "oral": "🎤", "final": "🏆"}
PLAN_KIND_NAME = {"summary": "Résumé", "reread": "Relecture", "flash": "Flashcards", "quiz": "Quiz", "oral": "Colle orale", "final": "Revue finale"}
PLAN_DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


def _plan_names(scope):
    scope = str(scope or "all")
    if scope.startswith("mod:"):
        return [n for n, d in documents_db.items() if (d.get("module") or "") == scope[4:]]
    if scope.startswith("doc:"):
        return [scope[4:]] if scope[4:] in documents_db else []
    return list(documents_db.keys())


def _plan_cfg_names(cfg):
    """Cours du plan : les PDF choisis à l'étape « Matières » (anciens plans : portée enregistrée)."""
    if isinstance(cfg.get("docs"), list):
        return [n for n in cfg["docs"] if n in documents_db]
    return _plan_names(cfg.get("scope"))


PLAN_KIND_COLOR = {"summary": "#4f6df5", "reread": "#8e6cf4", "flash": "#e0a43a", "quiz": "#2ea383", "oral": "#e86fb0", "final": "#f0715a"}
PLAN_KIND_GO = {"summary": "summary", "reread": "summary", "flash": "flash", "quiz": "quiz", "oral": "oral", "final": "oral"}
FR_MONTHS_LONG = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]


def _plan_short(name, n=34):
    data = documents_db.get(name, {})
    t = data.get("title") or os.path.splitext(name)[0]
    return t if len(t) <= n else t[:n - 1] + "…"


def _plan_done_pct(cfg, names, plan):
    """Part du plan réalisée : séances faites depuis la création du plan / (faites + restantes)."""
    since = _parse_day(cfg.get("created")) or _date.today()
    done = sum(1 for r in _read_sessions() if r.get("k") in ("flash", "quiz", "oral", "summary") and r.get("doc") in names
               and (_parse_day(r.get("t")) or _date.min) >= since)
    remaining = sum(1 for day in plan["days"] for t in day if not t.get("done"))
    return round(done / (done + remaining) * 100) if (done + remaining) else 0


# ---------------- assistant de création du plan (5 étapes) puis emploi du temps intelligent
PW_STEPS = [
    ("ÉTAPE 1 : CONTEXTE", "Bienvenue ! Dis-moi, c'est pour quoi ?"),
    ("ÉTAPE 2 : CALENDRIER", "Quel est le grand jour, et combien de temps par jour ?"),
    ("ÉTAPE 3 : MATIÈRES", "Qu'est-ce que tu révises ?"),
    ("ÉTAPE 4 : RYTHME", "Quelle intensité veux-tu ?"),
    ("ÉTAPE 5 : INTELLIGENCE", "Presque fini ! Choisis les fonctions IA."),
]
PW_CONTEXT_CHOICES = [("Examen", "exam"), ("Contrôle continu", "cc"), ("Apprentissage libre", "free")]
PW_ACT_CHOICES = [("Notes (résumé et pages à relire)", "notes"), ("Flashcards", "flash"), ("Quiz", "quiz"), ("Colle orale", "oral")]
PW_INTENSITY_CHOICES = [("Faible", "low"), ("Moyenne", "mid"), ("Élevée", "high")]
PW_REST_CHOICES = [("Lun", "0"), ("Mar", "1"), ("Mer", "2"), ("Jeu", "3"), ("Ven", "4"), ("Sam", "5"), ("Dim", "6")]
PW_AI_CHOICES = [("Répétition espacée", "spaced"), ("Priorisation IA", "prio"), ("Alertes de surcharge", "alerts")]
def _clean_hhmm(value, default="18:00"):
  """Heure « HH:MM » valide (sélecteur natif du navigateur), sinon la valeur par défaut."""
  m = re.match(r"^\s*([01]?\d|2[0-3])\s*[:hH]\s*([0-5]\d)", str(value or ""))
  return f"{int(m.group(1)):02d}:{m.group(2)}" if m else default
PW_DEFAULTS = {"context": "exam", "exam_name": "", "exam_date": "", "minutes": 40, "start": "18:00", "acts": list(PLAN_ACTS),
               "intensity": "mid", "rest": ["6"], "ai": ["spaced", "prio"]}


def _pw_mascot(pose):
    """Mascotte de l'assistant du plan : une expression par étape."""
    return mascot_svg(pose, "pw-mascot")


PW_POSES = ["wave", "point", "think", "determined", "celebrate"]


def _pw_head(step):
    step = max(1, min(5, int(step or 1)))
    title, sub = PW_STEPS[step - 1]
    return (f"<div class='pw-head'><div class='pw-bar'><i style='width:{step * 20}%'></i></div>"
            f"<span class='pw-brand'>AKORI Study</span><h2>{title}</h2><p>{sub}</p></div>")


def _pw_doc_choices():
    mods = _space()["modules"]
    out = []
    for name, data in documents_db.items():
        mod = mods.get(data.get("module") or "")
        prefix = f"{mod.get('semester') or ''} · {mod.get('name', '')} — " if mod else ""
        out.append((prefix.lstrip(" ·") + (data.get("title") or os.path.splitext(name)[0])[:70], name))
    return out


def _plan_cfg():
    cfg = dict(PW_DEFAULTS)
    cfg.update(_space().get("plan") or {})
    return cfg


def _plan_ready(cfg=None):
    cfg = cfg if cfg is not None else (_space().get("plan") or {})
    return bool(cfg.get("exam_date") or cfg.get("context") == "free")


def _hm(minutes):
    minutes = int(minutes) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _edt_status(task, offset):
    if offset == 0:
        return ("Fait", "st-ok") if task.get("done") else ("À faire", "st-todo")
    return ("Facultative", "st-opt") if task.get("prio", 5) <= 3 else ("Planifiée", "st-plan")


def _edt_detail(task, when, start, end, offset, multi_ai):
    color = PLAN_KIND_COLOR.get(task["kind"], "#4f6df5")
    status, cls = _edt_status(task, offset)
    pages = "".join(f"<span class='plan-page' title='{_escape_html(', '.join(pg.get('reasons', [])))}'>p. {int(pg['page'])}"
                    f"{(' · ' + _escape_html(pg['section'][:28])) if pg.get('section') else ''}</span>" for pg in task.get("pages") or [])
    launch = ("" if offset < 0 or task.get("done") else
              f"<button type='button' class='pl-go ak-t-{PLAN_KIND_GO.get(task['kind'], 'summary')}' data-doc='{_escape_html(task['doc'])}'>Lancer la séance</button>")
    return (f"<div class='ed-card' style='--k:{color}'><div class='ed-top'><span class='ed-ico'>{PLAN_KIND_ICON.get(task['kind'], '📌')}</span>"
            f"<div><small>{PLAN_KIND_NAME.get(task['kind'], '')}</small><h3>{_escape_html(task['label'])}</h3></div></div>"
            f"<div class='ed-grid'><div><span>Cours</span><b>{_escape_html(_plan_short(task['doc'], 60))}</b></div>"
            f"<div><span>Quand</span><b>{when} · {start} – {end}</b></div><div><span>Durée</span><b>{task['minutes']} min</b></div>"
            f"<div><span>Statut</span><b><i class='pl-st {cls}'>{status}</i></b></div></div>"
            + (f"<div class='ed-why'><b>Pourquoi cette séance ?</b><p>{_escape_html(task.get('why', ''))}</p></div>" if multi_ai and task.get("why") else "")
            + (f"<div class='ed-pages'><b>Pages à relire</b><div class='plan-pages'>{pages}</div></div>" if pages else "")
            + f"<div class='ed-acts'>{launch}</div></div>")


def plan_html():
    """Emploi du temps intelligent : une semaine à l'écran, séances placées à l'heure, détail au toucher."""
    cfg = _plan_cfg()
    if not _plan_ready(cfg):
        return ""
    names = _plan_cfg_names(cfg)
    if not names:
        return mascot_note("think", "Aucune matière dans ce plan.", "Importez un PDF ou cliquez sur « Modifier le plan » pour choisir vos matières."
                           "<button type='button' class='pl-btn pl-primary edt-edit'>Modifier le plan</button>", "edt")
    context = cfg.get("context") or "exam"
    exam = _parse_day(cfg.get("exam_date"))
    if context == "free" and not exam:
        exam = None
    minutes = int(cfg.get("minutes", 40) or 40)
    ai = list(cfg.get("ai") or [])
    plan = build_plan(exam, names, minutes, opts={"acts": cfg.get("acts"), "rest": cfg.get("rest"), "intensity": cfg.get("intensity"),
                                                  "ai": ai, "context": context, "final": context != "free"})
    if plan is None:
        return mascot_note("surprised", "La date du plan est dépassée.", "Cliquez sur « Modifier le plan » pour choisir une nouvelle date."
                           "<button type='button' class='pl-btn pl-primary edt-edit'>Modifier le plan</button>", "edt")
    try:
        sh, sm = (int(x) for x in _clean_hhmm(cfg.get("start")).split(":"))
    except ValueError:
        sh, sm = 18, 0
    day_start = sh * 60 + sm
    today = plan["today"]
    explain = "spaced" in ai or "prio" in ai
    # durée affichée par journée : la plus longue du plan (pauses comprises), arrondie à la demi-heure
    longest = max([sum(t["minutes"] for t in d) + PLAN_PAUSE * max(0, len(d) - 1) for d in plan["days"]] + [45])
    span = ((longest + 29) // 30) * 30
    step = 30 if span <= 180 else 60
    ticks = "".join(f"<span style='top:{(m / span) * 100:.2f}%'>{_hm(day_start + m)}</span>" for m in range(0, span, step))
    last_day = today + _timedelta(days=plan["horizon"] - 1)
    end_day = exam if exam and exam > last_day else last_day
    monday = today - _timedelta(days=today.weekday())
    n_weeks = (end_day - monday).days // 7 + 1
    weeks, labels = [], []
    for w in range(n_weeks):
        cols = []
        week_min = 0
        for i in range(7):
            day = monday + _timedelta(days=w * 7 + i)
            o = (day - today).days
            head_lbl = f"{PLAN_DAYS[day.weekday()][:3].capitalize()} <b>{day.day:02d}</b>"
            full = "Aujourd'hui" if o == 0 else f"{PLAN_DAYS[day.weekday()].capitalize()} {day.strftime('%d/%m')}"
            if exam and day == exam:
                exam_lbl = "Contrôle" if context == "cc" else ("Examen" if context == "exam" else "Fin du plan")
                det = (f"<div class='ed-card' style='--k:#e0a43a'><div class='ed-top'><span class='ed-ico'>🏆</span><div><small>{full}</small>"
                       f"<h3>{_escape_html(cfg.get('exam_name') or exam_lbl)}</h3></div></div>"
                       "<div class='ed-why'><b>Le jour J</b><p>Pas de nouvelle notion : relisez seulement votre résumé et vos flashcards « à revoir ».</p></div></div>")
                cols.append(f"<div class='edt-col exam'><button type='button' class='edt-dh'>{head_lbl}<template>{det}</template></button>"
                            f"<div class='edt-body'><button type='button' class='edt-exam'>🏆<span>{exam_lbl}</span><template>{det}</template></button></div></div>")
                continue
            if o < 0 or o >= plan["horizon"]:
                cols.append(f"<div class='edt-col off'><div class='edt-dh'>{head_lbl}</div><div class='edt-body'></div></div>")
                continue
            tasks = plan["days"][o]
            if not tasks:
                rest = day.weekday() in plan["rest"]
                det = (f"<div class='ed-card' style='--k:#8e6cf4'><div class='ed-top'><span class='ed-ico'>☕</span><div><small>{full}</small>"
                       f"<h3>{'Jour de repos' if rest else 'Journée libre'}</h3></div></div>"
                       "<div class='ed-why'><p>Laissez la mémoire consolider : aucune séance n'est prévue.</p></div></div>")
                cols.append(f"<div class='edt-col {'rest' if rest else 'free'}{' today' if o == 0 else ''}'><button type='button' class='edt-dh'>{head_lbl}<template>{det}</template></button>"
                            f"<div class='edt-body'><span class='edt-rest'>{'Repos' if rest else ''}</span></div></div>")
                continue
            blocks, items, cur = [], [], 0
            for t in tasks:
                a, b = day_start + cur, day_start + cur + t["minutes"]
                color = PLAN_KIND_COLOR.get(t["kind"], "#4f6df5")
                small = " sm" if t["minutes"] / span < .12 else ""
                det = _edt_detail(t, full, _hm(a), _hm(b), o, explain)
                course = _escape_html(_plan_short(t["doc"], 26))
                blocks.append(f"<button type='button' class='edt-blk k-{t['kind']}{small}{' done' if t.get('done') else ''}' "
                              f"style='top:{cur / span * 100:.2f}%;height:{t['minutes'] / span * 100:.2f}%;--k:{color}'>"
                              f"<span class='eb-t'>{PLAN_KIND_ICON.get(t['kind'], '📌')} {_escape_html(PLAN_KIND_NAME.get(t['kind'], ''))}</span>"
                              f"<span class='eb-s'>{course}</span><span class='eb-h'>{_hm(a)} · {t['minutes']} min</span><template>{det}</template></button>")
                items.append(f"<li style='--k:{color}'><em>{_hm(a)} – {_hm(b)}</em><span>{PLAN_KIND_ICON.get(t['kind'], '📌')} {_escape_html(t['label'])}</span>"
                             f"<small>{course}</small></li>")
                cur += t["minutes"] + PLAN_PAUSE
            week_min += plan["load"][o]
            over = plan["load"][o] > plan["cap"] and len(tasks) > 1
            day_det = (f"<div class='ed-card' style='--k:#4f6df5'><div class='ed-top'><span class='ed-ico'>📅</span><div><small>{len(tasks)} séance(s)</small>"
                       f"<h3>{full} · {plan['load'][o]} min</h3></div></div><ul class='ed-day'>{''.join(items)}</ul>"
                       + ("<div class='ed-why warn'><p>⚠️ Journée chargée : elle dépasse votre temps par jour.</p></div>" if over and "alerts" in ai else "")
                       + "</div>")
            cols.append(f"<div class='edt-col{' today' if o == 0 else ''}{' over' if over and 'alerts' in ai else ''}'>"
                        f"<button type='button' class='edt-dh'>{head_lbl}<small>{plan['load'][o]} min</small><template>{day_det}</template></button>"
                        f"<div class='edt-body'>{''.join(blocks)}</div></div>")
        a, b = monday + _timedelta(days=w * 7), monday + _timedelta(days=w * 7 + 6)
        label = f"Semaine {w + 1} · {a.strftime('%d/%m')} – {b.strftime('%d/%m')} · {week_min} min"
        labels.append(label)
        weeks.append(f"<div class='edt-week{' on' if w == 0 else ''}' data-label='{label}' style='--tick:{step / span * 100:.3f}%'><div class='edt-hours'>{ticks}</div>{''.join(cols)}</div>")
    pct = _plan_done_pct(cfg, names, plan)
    circ = 2 * 3.14159 * 20
    ring = (f"<svg viewBox='0 0 48 48' class='pl-ring' aria-hidden='true'><circle cx='24' cy='24' r='20' class='pl-ring-t'/>"
            + (f"<circle cx='24' cy='24' r='20' class='pl-ring-v' stroke-dasharray='{circ * pct / 100:.1f} {circ:.1f}' transform='rotate(-90 24 24)'/>" if pct else "")
            + f"</svg><b>{pct}%</b>")
    ctx = PLAN_CONTEXTS.get(context, "Examen")
    if exam:
        d_label = "Jour J" if plan["days_left"] == 0 else f"J-{plan['days_left']}"
        when = f"{exam.day} {FR_MONTHS_LONG[exam.month - 1]} {exam.year} · {d_label}"
    else:
        when = f"Plan sur {PLAN_FREE_DAYS // 7} semaines"
    title = _escape_html(cfg.get("exam_name") or ("Votre examen" if context == "exam" else ("Votre contrôle" if context == "cc" else "Mes révisions")))
    rest_lbl = ", ".join(PLAN_DAYS[int(r)][:3] + "." for r in sorted(plan["rest"])) or "aucun"
    rhythm = PLAN_INTENSITY.get(cfg.get("intensity") or "mid", PLAN_INTENSITY["mid"])["label"]
    chips = "".join(f"<span style='--k:{PLAN_KIND_COLOR[k]}'>{PLAN_KIND_ICON[k]} {PLAN_KIND_NAME[k]}</span>"
                    for k in ("summary", "reread", "flash", "quiz", "oral", "final"))
    alerts = "".join(f"<div class='edt-alert'>⚠️ {_escape_html(a)}</div>" for a in plan["alerts"])
    if alerts:
        alerts = f"<div class='edt-alerts'>{alerts}</div>"
    ai_lbl = " · ".join(lbl for lbl, v in PW_AI_CHOICES if v in ai) or "aucune fonction IA"
    return (f"<div class='edt' data-weeks='{len(weeks)}'>"
            f"<div class='edt-head'><div class='edt-id'><span class='edt-ctx'>{ctx}</span><b>{title}</b><small>{when}</small></div>"
            f"<div class='edt-kpi'><b>{minutes} min</b><span>par jour · dès {_hm(day_start)}</span></div>"
            f"<div class='edt-kpi'><b>{len(names)} matière(s)</b><span>Rythme {rhythm} · repos : {rest_lbl}</span></div>"
            f"<div class='edt-kpi edt-pct'><span class='pl-ringw'>{ring}</span><span>du plan<br>terminé</span></div>"
            "<div class='edt-acts'><button type='button' class='pl-btn pl-primary edt-edit'>Modifier le plan</button>"
            "<button type='button' class='pl-btn edt-del'>Supprimer</button></div></div>"
            f"{alerts}"
            f"<div class='edt-bar'><button type='button' class='edt-nav edt-prev' aria-label='Semaine précédente'>‹</button>"
            f"<b class='edt-wl'>{labels[0] if labels else ''}</b><button type='button' class='edt-nav edt-next' aria-label='Semaine suivante'>›</button>"
            f"<div class='edt-legend'>{chips}</div></div>"
            f"<div class='edt-weeks'>{''.join(weeks)}</div>"
            f"<p class='edt-note'>Touchez une séance ou un jour pour voir le détail. {_escape_html(ai_lbl)} · "
            "le plan se recalcule après chaque séance.</p></div>")


def _pw_view(step, error=""):
    """Étape affichée de l'assistant (en-tête, colonnes, boutons)."""
    step = max(1, min(5, int(step or 1)))
    return (step, _pw_head(step), _pw_mascot(PW_POSES[step - 1]), *[gr.update(visible=step == i) for i in range(1, 6)],
            gr.update(visible=step > 1), gr.update(visible=step < 5), gr.update(visible=step == 5), error)


def _pw_check(step, context, exam_date, docs, acts, rest):
    """Message d'erreur si l'étape `step` est incomplète, sinon ''."""
    if step == 2:
        day = _parse_day(exam_date)
        if context != "free" and not day:
            return "⚠️ Choisis la date de ton examen." if context == "exam" else "⚠️ Choisis la date de ton contrôle."
        if day and day < _date.today():
            return "⚠️ Cette date est déjà passée : choisis une date à venir."
    if step == 3:
        if not [d for d in (docs or []) if d in documents_db]:
            return "⚠️ Choisis au moins une matière parmi tes PDF." if documents_db else "⚠️ Importe d'abord un PDF dans « Mes dossiers »."
        if not acts:
            return "⚠️ Garde au moins une activité (notes, flashcards, quiz ou colle orale)."
    if step == 4 and len(rest or []) >= 7:
        return "⚠️ Garde au moins un jour de révision dans la semaine."
    return ""


def pw_next(step, context, exam_date, docs, acts, rest):
    step = int(step or 1)
    err = _pw_check(step, context, exam_date, docs, acts, rest)
    return _pw_view(step if err else step + 1, err)


def pw_prev(step):
    return _pw_view(int(step or 1) - 1)


def _pw_values(cfg):
    """Valeurs des champs de l'assistant, pré-remplies avec le plan enregistré."""
    docs = [d for d in (_plan_cfg_names(cfg) if _plan_ready(cfg) else []) if d in documents_db] or list(documents_db.keys())
    return (gr.update(value=cfg.get("context") or "exam"), cfg.get("exam_name", ""), cfg.get("exam_date", ""),
            int(cfg.get("minutes", 40) or 40), gr.update(value=_clean_hhmm(cfg.get("start"))),
            gr.update(choices=_pw_doc_choices(), value=docs), gr.update(value=list(cfg.get("acts") or PLAN_ACTS)),
            gr.update(value=cfg.get("intensity") or "mid"), gr.update(value=[str(r) for r in cfg.get("rest") or []]),
            gr.update(value=list(cfg.get("ai") if cfg.get("ai") is not None else PW_DEFAULTS["ai"])))


def _plan_open_view(force_wizard):
    """Ouverture de l'onglet : l'emploi du temps s'il existe, sinon l'assistant (étape 1, champs pré-remplis)."""
    cfg = _plan_cfg()
    show_edt = _plan_ready(cfg) and not force_wizard
    return (gr.update(visible=not show_edt), gr.update(visible=show_edt), plan_html() if show_edt else gr.update(),
            *_pw_view(1), *_pw_values(cfg))


def plan_open():
    return _plan_open_view(False)


def plan_edit():
    return _plan_open_view(True)


def pw_cancel():
    """Annuler : retour à l'emploi du temps s'il existe, sinon l'assistant revient à l'étape 1."""
    ready = _plan_ready()
    return (gr.update(visible=not ready), gr.update(visible=ready), *_pw_view(1))


def pw_generate(context, exam_name, exam_date, minutes, start, docs, acts, intensity, rest, ai):
    for s in (2, 3, 4):
        err = _pw_check(s, context, exam_date, docs, acts, rest)
        if err:
            return (gr.update(), gr.update(), gr.update(), *_pw_view(s, err))
    day = _parse_day(exam_date)
    space = _space()
    prev = space.get("plan") or {}
    docs = [d for d in docs if d in documents_db]
    space["plan"] = {
        "context": context if context in PLAN_CONTEXTS else "exam", "exam_name": str(exam_name or "").strip()[:60],
        "exam_date": day.isoformat() if day else "", "docs": docs, "minutes": max(10, min(240, int(minutes or 40))),
        "start": _clean_hhmm(start), "acts": [a for a in acts if a in PLAN_ACTS],
        "intensity": intensity if intensity in PLAN_INTENSITY else "mid", "rest": [r for r in rest or [] if r in "0123456"],
        "ai": [a for a in ai or [] if a in PLAN_AI],
        "created": (prev.get("created") if prev.get("exam_date") == (day.isoformat() if day else "") else None) or _date.today().isoformat(),
    }
    _space_save(space)
    _log("plan_save", detail=f"{context} · {day.isoformat() if day else 'libre'} · {len(docs)} matière(s)")
    return (gr.update(visible=False), gr.update(visible=True), plan_html(), *_pw_view(1, ""))


def plan_clear():
    space = _space()
    space["plan"] = {}
    _space_save(space)
    return _plan_open_view(True)


# ---------------- journal de révision
SESSION_LABEL = {"quiz": ("🎯", "Quiz"), "flash": ("🎰", "Flashcards"), "oral": ("🎤", "Colle orale"), "chat": ("✦", "Assistant"),
                 "summary": ("📚", "Résumé"), "mindmap": ("🕸", "Mindmap")}


def _streak(days_set, today):
    day = today if today in days_set else today - _timedelta(days=1)
    count = 0
    while day in days_set:
        count += 1
        day -= _timedelta(days=1)
    return count


def _gaps(limit=6):
    """Cartographie des lacunes : notions les moins réussies aux quiz, tous cours confondus."""
    found = []
    for name in documents_db:
        for topic, v in _progress_topics(name).items():
            if v["total"] >= 2 and v["correct"] / v["total"] < .6:
                found.append((round(v["correct"] / v["total"] * 100), topic, name))
    return sorted(found)[:limit]


def journal_html():
    sessions = _read_sessions()
    today = _date.today()
    if not sessions:
        return mascot_note("sleep", "Votre journal de révision commence ici.",
                           "Chaque quiz, série de flashcards, colle orale, question à l'assistant ou résumé y sera noté, jour après jour.")
    by_day = {}
    for rec in sessions:
        d = _parse_day(rec.get("t"))
        if d:
            by_day.setdefault(d, []).append(rec)
    days_set = set(by_day)
    last30 = sum(1 for d in days_set if (today - d).days < 30)
    kinds = {}
    for rec in sessions:
        kinds[rec.get("k")] = kinds.get(rec.get("k"), 0) + 1
    quiz_scores = [r for r in sessions if r.get("k") == "quiz" and r.get("total")]
    qavg = round(sum(r["score"] / r["total"] for r in quiz_scores) / len(quiz_scores) * 100) if quiz_scores else None
    kpi = (f"<div class='jr-kpis'><div><b class='cu'>{_streak(days_set, today)}</b><span>jour(s) d'affilée</span></div>"
           f"<div><b class='cu'>{last30}</b><span>jour(s) étudié(s) sur 30</span></div>"
           f"<div><b class='cu'>{len(sessions)}</b><span>séance(s) au total</span></div>"
           f"<div><b>{qavg if qavg is not None else '—'}{'%' if qavg is not None else ''}</b><span>moyenne des quiz</span></div></div>")
    strip = "".join(f"<i class='{'jr-on' if (today - _timedelta(days=13 - i)) in days_set else ''}' title='{(today - _timedelta(days=13 - i)).strftime('%d/%m')}'></i>" for i in range(14))
    timeline = []
    for day in sorted(by_day, reverse=True)[:12]:
        agg = {}
        for rec in by_day[day]:
            agg.setdefault((rec.get("k"), rec.get("doc", "")), []).append(rec)
        lines = []
        for (kind, doc), recs in agg.items():
            icon, name = SESSION_LABEL.get(kind, ("📌", kind))
            where = os.path.splitext(doc)[0][:34] if doc else ""
            if kind == "quiz":
                detail = " · ".join(f"{r['score']}/{r['total']}" for r in recs)
            elif kind == "flash":
                detail = f"{len(recs)} carte(s) · {sum(1 for r in recs if r.get('result') == 'known')} sues"
            elif kind == "oral":
                detail = " · ".join(f"{r['score']}/10" for r in recs)
            elif kind == "chat":
                detail = f"{len(recs)} question(s)"
            else:
                detail = ""
            lines.append(f"<li><span class='jr-ico'>{icon}</span><b>{name}</b><span class='jr-doc'>{_escape_html(where)}</span><em>{_escape_html(detail)}</em></li>")
        label = "Aujourd'hui" if day == today else ("Hier" if day == today - _timedelta(days=1) else day.strftime("%d/%m/%Y"))
        timeline.append(f"<div class='jr-day'><div class='jr-date'>{label}</div><ul>{''.join(lines)}</ul></div>")
    gaps = _gaps()
    gap_html = ("<div class='jr-gaps'><div class='jr-sub'>Mes lacunes (notions les moins réussies aux quiz)</div>" + "".join(
        f"<div class='jr-gap'><span>{_escape_html(t)}<small>{_escape_html(os.path.splitext(n)[0][:30])}</small></span><div class='act-bar'><i style='width:{p}%'></i></div><b>{p}%</b></div>"
        for p, t, n in gaps) + "</div>") if gaps else ""
    hint = _student_hint(next(iter(documents_db), None)) if documents_db else ""
    adapt = ("<div class='jr-adapt'>✦ <span>Les réponses de l'assistant s'adaptent à votre niveau et à vos résultats.</span></div>" if hint else "")
    return kpi + f"<div class='jr-strip'><span>14 derniers jours</span><div>{strip}</div></div>" + adapt + gap_html + "<div class='jr-timeline'>" + "".join(timeline) + "</div>"


def history_page_html():
    """Historique : ce que l'étudiant a fait, jour par jour, du plus récent au plus ancien."""
    sessions = _read_sessions()
    if not sessions:
        return mascot_note("sleep", "Aucune activité pour le moment.", "Vos quiz, flashcards, colles orales, questions et résumés apparaîtront ici.")
    today = _date.today()
    days = {}
    for rec in sessions:
        d = _parse_day(rec.get("t"))
        if d:
            days.setdefault(d, []).append(rec)
    blocks = []
    for day in sorted(days, reverse=True)[:45]:
        items, flash = [], {}
        for rec in days[day]:
            kind, doc = rec.get("k"), rec.get("doc", "")
            if kind == "flash":                       # une ligne par série de flashcards et par cours
                f = flash.setdefault(doc, {"n": 0, "known": 0, "t": rec.get("t", "")})
                f["n"] += 1
                f["known"] += 1 if rec.get("result") == "known" else 0
                f["t"] = rec.get("t", "")
                continue
            items.append(rec)
        for doc, f in flash.items():
            items.append({"k": "flash", "doc": doc, "t": f["t"], "_n": f["n"], "_known": f["known"]})
        items.sort(key=lambda r: r.get("t", ""), reverse=True)
        rows = []
        for rec in items:
            kind = rec.get("k")
            icon, name = SESSION_LABEL.get(kind, ("📌", kind or ""))
            course = os.path.splitext(rec.get("doc", ""))[0]
            hour = (rec.get("t", "")[11:16]) or ""
            if kind == "quiz" and rec.get("total"):
                detail = f"Score {rec.get('score', 0)}/{rec['total']}"
            elif kind == "oral" and rec.get("score") is not None:
                detail = f"Note {rec['score']}/10"
            elif kind == "flash":
                detail = f"{rec['_n']} carte(s), {rec['_known']} sue(s)"
            elif kind == "chat":
                detail = (rec.get("q") or "Question à l'assistant") + (" — absent du cours" if rec.get("refused") else "")
            elif kind == "summary":
                detail = "Résumé généré"
            elif kind == "mindmap":
                detail = "Carte mentale générée"
            else:
                detail = ""
            rows.append(f"<li class='hx-item'><span class='hx-time'>{hour}</span><span class='hx-ico hx-{_escape_html(kind or '')}'>{icon}</span>"
                        f"<div class='hx-main'><b>{_escape_html(name)}</b><span class='hx-detail'>{_escape_html(detail)}</span></div>"
                        f"<span class='hx-course' title='{_escape_html(course)}'>{_escape_html(course[:28])}</span></li>")
        label = "Aujourd'hui" if day == today else ("Hier" if day == today - _timedelta(days=1) else
                f"{PLAN_DAYS[day.weekday()].capitalize()} {day.strftime('%d/%m/%Y')}")
        blocks.append(f"<section class='hx-day'><div class='hx-date'>{label}<span>{len(items)} activité(s)</span></div><ul>{''.join(rows)}</ul></section>")
    return "<div class='hx'>" + "".join(blocks) + "</div>"


def after_session_refresh():
    """Après chaque séance : le plan et le journal se recalculent."""
    return plan_html(), history_page_html()


# ============================================================
# MINDMAP — carte mentale du cours (repliable), générée à partir des pages
# ============================================================
MM_COLORS = ["#4f6df5", "#e86fb0", "#10b981", "#f59e0b", "#8e6cf4", "#06b6d4", "#ef4444"]
_MM_STOP = set("dans pour avec cette comme plus sont être avoir leur leurs elles nous vous aussi donc mais ainsi chaque entre sans sous tout tous toute toutes "
               "celui celle fait peut peuvent cours exemple partie premier première deuxième lorsque alors comment pourquoi quand dont ceux celles ces ses mes".split())


def _mm_trim(text, n):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def _mm_clean(data):
    """JSON de Gemini -> {"title", "branches": [{"label", "children": [...]}]} borné et propre, ou None."""
    if not isinstance(data, dict):
        return None
    branches = []
    for b in (data.get("branches") or data.get("children") or [])[:7]:
        if isinstance(b, dict):
            label = _mm_trim(b.get("label") or b.get("title") or b.get("name"), 38)
            kids = [_mm_trim(c if isinstance(c, str) else (c.get("label") or c.get("title") or "") if isinstance(c, dict) else "", 40) for c in (b.get("children") or [])[:5]]
        elif isinstance(b, str):
            label, kids = _mm_trim(b, 38), []
        else:
            continue
        kids = [k for k in kids if k]
        if label:
            branches.append({"label": label, "children": kids})
    if len(branches) < 2:
        return None
    return {"title": _mm_trim(data.get("title") or data.get("root") or "Cours", 40), "branches": branches}


def _mindmap_offline(doc_name):
    """Carte de secours sans IA : sections du document et mots les plus fréquents de chacune."""
    doc = documents_db[doc_name]
    groups = {}
    for d in doc["vector_db"].docstore._dict.values():
        meta = d.metadata or {}
        key = (meta.get("section") or "").strip() or f"Page {int(meta.get('page', 1) or 1)}"
        groups.setdefault(key, []).append(d.page_content)
    ranked = sorted(groups.items(), key=lambda kv: -sum(len(t) for t in kv[1]))[:6]
    branches = []
    for label, texts in ranked:
        freq = {}
        for w in re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\-]{5,}", " ".join(texts)):
            lw = w.lower()
            if lw not in _MM_STOP:
                freq[lw] = freq.get(lw, 0) + 1
        kids = [w.capitalize() for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:3]]
        branches.append({"label": _mm_trim(label, 38), "children": kids})
    return {"title": _mm_trim(doc.get("title") or os.path.splitext(doc_name)[0], 40), "branches": branches}


def _mm_w(text, base=26, per=7.4):
    return int(len(text) * per + base)


def mindmap_html(data):
    if not data or not data.get("branches"):
        return mascot_note("think", "Aucune mindmap pour ce cours.", "Cliquez sur « Générer la mindmap » : AKORI dessine la carte mentale du cours actif.")
    branches = data["branches"]
    X0, ROW = 14, 40
    data = dict(data, title=_mm_trim(data["title"], 24))
    rw = max(150, min(250, _mm_w(data["title"], 40)))
    bw = max(120, min(240, max(_mm_w(b["label"]) for b in branches)))
    bx, lx = X0 + rw + 70, X0 + rw + 70 + bw + 60
    leaf_w = [[max(100, min(270, _mm_w(c))) for c in b["children"]] for b in branches]
    max_leaf = max([w for ws in leaf_w for w in ws] or [100])
    W = lx + max_leaf + 16
    y, by_list, leaf_y = 22, [], []
    for b in branches:
        ys = []
        for _ in (b["children"] or [None]):
            ys.append(y + ROW / 2)
            y += ROW
        leaf_y.append(ys)
        by_list.append(sum(ys) / len(ys))
        y += 14
    H = y + 4
    ry = sum(by_list) / len(by_list)
    out = [f"<svg class='mm-svg' viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg'>"]
    out.append(f"<defs><linearGradient id='mmroot' x1='0' x2='1'><stop offset='0' stop-color='#4f6df5'/><stop offset='1' stop-color='#8e6cf4'/></linearGradient></defs>")
    for i, b in enumerate(branches):
        c = MM_COLORS[i % len(MM_COLORS)]
        by = by_list[i]
        out.append(f"<path class='mm-path' style='--i:{i}' d='M{X0 + rw} {ry:.1f} C{X0 + rw + 40} {ry:.1f} {bx - 40} {by:.1f} {bx} {by:.1f}' stroke='{c}' fill='none' stroke-width='3'/>")
    out.append(f"<g class='mm-root'><rect x='{X0}' y='{ry - 25:.1f}' width='{rw}' height='50' rx='25' fill='url(#mmroot)'/>"
               f"<text x='{X0 + rw / 2}' y='{ry + 5:.1f}' text-anchor='middle' class='mm-t-root'>{_escape_html(data['title'])}<title>{_escape_html(data['title'])}</title></text></g>")
    for i, b in enumerate(branches):
        c = MM_COLORS[i % len(MM_COLORS)]
        by = by_list[i]
        g = [f"<g class='mm-br' style='--i:{i}'>"]
        for k, kid in enumerate(b["children"]):
            ky, kw = leaf_y[i][k], leaf_w[i][k]
            g.append(f"<path class='mm-lp' d='M{bx + bw} {by:.1f} C{bx + bw + 30} {by:.1f} {lx - 30} {ky:.1f} {lx} {ky:.1f}' stroke='{c}' fill='none' stroke-width='2' opacity='.65'/>")
            g.append(f"<g class='mm-leaf'><rect x='{lx}' y='{ky - 15:.1f}' width='{kw}' height='30' rx='15' fill='{c}' fill-opacity='.12' stroke='{c}' stroke-opacity='.55'/>"
                     f"<text x='{lx + kw / 2}' y='{ky + 4.5:.1f}' text-anchor='middle' class='mm-t-leaf'>{_escape_html(kid)}<title>{_escape_html(kid)}</title></text></g>")
        g.append(f"<g class='mm-bnode'><rect x='{bx}' y='{by - 19:.1f}' width='{bw}' height='38' rx='19' fill='{c}'/>"
                 f"<text x='{bx + bw / 2}' y='{by + 5:.1f}' text-anchor='middle' class='mm-t-br'>{_escape_html(b['label'])}<title>{_escape_html(b['label'])}</title></text>")
        if b["children"]:
            g.append(f"<circle cx='{bx + bw}' cy='{by:.1f}' r='9' fill='#fff' stroke='{c}' stroke-width='2'/>"
                     f"<text x='{bx + bw}' y='{by + 4:.1f}' text-anchor='middle' class='mm-minus' fill='{c}'>−</text><text x='{bx + bw}' y='{by + 4:.1f}' text-anchor='middle' class='mm-plus' fill='{c}'>+</text>")
        g.append("</g></g>")
        out.append("".join(g))
    out.append("</svg>")
    zoom = ("<div class='mm-zoom'><button type='button' class='mm-z' data-z='out' aria-label='Dézoomer'>−</button>"
            "<span class='mm-zl'>100 %</span><button type='button' class='mm-z' data-z='in' aria-label='Zoomer'>+</button>"
            "<button type='button' class='mm-z mm-fit' data-z='fit'>Ajuster</button></div>")
    return ("<div class='mm-wrap'><div class='mm-bar'><div class='mm-hint'>Cliquez sur une branche pour la replier ou la déplier. "
            "Zoomez avec + et −, puis faites glisser la carte.</div>" + zoom + "</div><div class='mm-scroll' style='--mmz:1'>" + "".join(out) + "</div></div>")


def mindmap_view(doc_name=None, lang=None):
    cached = ((documents_db.get(doc_name) or {}).get("mindmap_cache") or {}).get(_lang_code(lang)) if doc_name else None
    return mindmap_html(cached), ""


def mindmap_generate_handler(doc_name, lang=None):
    if not doc_name or doc_name not in documents_db:
        yield mindmap_html(None), "⚠️ Sélectionnez d'abord un cours."
        return
    entry = documents_db[doc_name]
    context = "\n\n".join(preparer_contexte_global(entry))[:24000]
    prompt = f"""CONTEXTE DU COURS:\n{context}\n\n
Construis la carte mentale de ce cours : un titre central, 5 à 7 grandes notions (branches) et 2 à 4 sous-notions par branche.
Libellés courts (6 mots maximum), fidèles au vocabulaire du cours. N'invente rien.
Retourne UNIQUEMENT un JSON valide : {{"title": "...", "branches": [{{"label": "...", "children": ["...", "..."]}}]}}"""
    note = ""

    def work():
        raw = _gemini_structured(prompt, _with_lang("Tu es AKORI, assistant académique. Base-toi uniquement sur le contexte fourni.", lang), max_tokens=1800)
        return _mm_clean(_extract_json_from_text(raw))

    data = None
    try:
        data = yield from _with_heartbeat(work, lambda sec: (gr.update(), f"⏳ Construction de la mindmap… ({sec} s)"))
    except Exception as e:
        _log("error", detail=f"mindmap · {e}")
        note = f" (IA indisponible : carte simplifiée à partir des sections — {str(e)[:80]})"
    if not data:
        data = _mindmap_offline(doc_name)
        note = note or " (carte simplifiée à partir des sections du document)"
    entry.setdefault("mindmap_cache", {})[_lang_code(lang)] = data
    _save_history(doc_name)
    _session("mindmap", doc_name)
    _log("mindmap", detail=doc_name)
    yield mindmap_html(data), "✅ Mindmap prête." + note


# ============================================================
# COLLE ORALE — l'étudiant répond à une question ouverte, AKORI le note d'après le cours
# ============================================================
ORAL_SYSTEM = ("Tu es un examinateur bienveillant. Tu t'appuies UNIQUEMENT sur l'extrait de cours fourni, "
               "sans rien inventer ni utiliser d'autres connaissances.")


def _oral_prepare(doc_data):
    """Choisit au hasard un passage assez riche du cours (thread principal : accès à l'index)."""
    docs = [d for d in doc_data["vector_db"].docstore._dict.values() if len(d.page_content.strip()) > 180]
    if not docs:
        docs = [d for d in doc_data["vector_db"].docstore._dict.values() if d.page_content.strip()]
    if not docs:
        return None
    doc = random.choice(docs)
    meta = getattr(doc, "metadata", None) or {}
    return {"excerpt": doc.page_content.strip(), "page": meta.get("page"), "section": meta.get("section") or ""}


def _oral_ask(passage, lang=None):
    """Une question orale + les points clés attendus, tirés du passage (appel Gemini, sans accès aux données)."""
    prompt = (f"EXTRAIT DU COURS (page {passage.get('page') or '?'}):\n{passage['excerpt']}\n\n"
              "Pose UNE question orale ouverte (une ou deux phrases), comme à un examen oral, qui porte sur cet extrait. "
              "Donne aussi les 3 à 5 points clés attendus dans une bonne réponse, courts et tirés strictement de l'extrait.\n"
              'Retourne uniquement un JSON : {"question":"...","key_points":["...","..."]}')
    data = _extract_json_from_text(_gemini_structured(prompt, _with_lang(ORAL_SYSTEM, lang), max_tokens=600))
    question = str((data or {}).get("question") or "").strip() if isinstance(data, dict) else ""
    points = [str(p).strip() for p in (data.get("key_points") or []) if str(p).strip()] if isinstance(data, dict) else []
    if not question or not points:
        raise RuntimeError("Question orale illisible : réessayez.")
    return {"question": question, "key_points": points[:6], "page": passage.get("page"),
            "section": passage.get("section") or "", "excerpt": passage["excerpt"][:700]}


def _oral_words(text):
    import unicodedata
    norm = unicodedata.normalize("NFD", str(text).lower())
    norm = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    return {w for w in re.findall(r"[a-z0-9]{4,}", norm)}


def _oral_grade_offline(state, answer):
    """Correction approximative sans IA : un point clé est couvert si la moitié de ses mots se retrouve dans la réponse."""
    answer_words = _oral_words(answer)
    covered, missed = [], []
    for point in state.get("key_points", []):
        words = _oral_words(point)
        hit = len(words & answer_words) / len(words) if words else 0
        (covered if hit >= 0.5 else missed).append(point)
    total = max(1, len(covered) + len(missed))
    return {"score": round(10 * len(covered) / total), "covered": covered, "missed": missed,
            "feedback": "Correction approximative : l'IA est momentanément indisponible, la note compare seulement les mots-clés du cours à votre réponse.",
            "model_answer": " ".join(state.get("key_points", [])), "offline": True}


def _oral_grade(state, answer, lang=None):
    """Note la réponse de l'étudiant d'après le cours (appel Gemini, sans accès aux données)."""
    points = "\n".join(f"- {p}" for p in state.get("key_points", []))
    prompt = (f"QUESTION : {state['question']}\n\nPOINTS CLÉS ATTENDUS (tirés du cours) :\n{points}\n\n"
              f"EXTRAIT DU COURS (page {state.get('page') or '?'}) :\n{state.get('excerpt', '')}\n\n"
              f"RÉPONSE DE L'ÉTUDIANT :\n{answer}\n\n"
              "Évalue la réponse UNIQUEMENT par rapport au cours. Retourne uniquement un JSON :\n"
              '{"score": entier de 0 à 10, "covered": ["points clés bien couverts"], "missed": ["points clés oubliés ou faux"], '
              '"feedback": "deux phrases : un encouragement et un conseil précis", '
              '"model_answer": "réponse modèle courte (3 à 4 phrases) fondée sur l\'extrait"}')
    data = _extract_json_from_text(_gemini_structured(prompt, _with_lang(ORAL_SYSTEM, lang), max_tokens=900))
    if not isinstance(data, dict) or "score" not in data:
        raise RuntimeError("Correction illisible : réessayez.")
    try:
        score = max(0, min(10, int(round(float(data.get("score", 0))))))
    except (TypeError, ValueError):
        score = 0
    return {"score": score, "covered": [str(x) for x in data.get("covered", []) if str(x).strip()],
            "missed": [str(x) for x in data.get("missed", []) if str(x).strip()],
            "feedback": str(data.get("feedback", "")).strip(), "model_answer": str(data.get("model_answer", "")).strip()}


def oral_card_html(state):
    if not state or not state.get("question"):
        return mascot_note("wave", "Prêt pour votre colle ?", "Cliquez sur « Nouvelle question » : AKORI vous interroge sur votre cours, comme à l'oral.")
    return ("<div class='oral-card'><span class='oral-tag'>Question orale</span>"
            f"<h3>{_escape_html(state['question'])}</h3></div>")


def oral_result_html(state, result):
    score = int(result["score"])
    band = "good" if score >= 8 else ("mid" if score >= 5 else "low")
    verdict = "Très bien" if score >= 8 else ("Correct, à compléter" if score >= 5 else "À reprendre")
    where = ""
    if state.get("page"):
        where = f"page {int(state['page'])}" + (f" · {_escape_html(state['section'])}" if state.get("section") else "")
    covered = "".join(f"<li>{_escape_html(x)}</li>" for x in result["covered"]) or "<li class='oral-none'>Aucun point clé repéré.</li>"
    missed = "".join(f"<li>{_escape_html(x)}</li>" for x in result["missed"]) or "<li class='oral-none'>Rien d'oublié, bravo.</li>"
    reread = f"<p class='oral-reread'>📚 <span>À relire</span> : {where}</p>" if (where and result["missed"]) else ""
    model = (f"<details class='oral-model'><summary>Réponse modèle</summary><p>{_escape_html(result['model_answer'])}</p>"
             f"<p class='oral-src'>Source : {where or 'cours'}</p></details>") if result.get("model_answer") else ""
    note = "<p class='oral-offline'>ℹ️ Correction approximative (IA indisponible).</p>" if result.get("offline") else ""
    return (
        f"<div class='oral-result'><div class='oral-top'><div class='ms-box oral-ms'>{mascot_svg(_cheer(score * 10)[0])}</div>"
        f"<div class='oral-score oral-{band}'><b>{score}</b><span>/10</span></div>"
        f"<div><div class='oral-verdict'>{verdict}</div><p class='oral-feedback'>{_escape_html(result.get('feedback', ''))}</p></div></div>{note}"
        f"<div class='oral-cols'><div><h4>✅ Points maîtrisés</h4><ul>{covered}</ul></div>"
        f"<div><h4>🎯 À retravailler</h4><ul>{missed}</ul>{reread}</div></div>{model}</div>"
    )


def oral_history_html(doc_name):
    attempts = (_progress_for(doc_name).get("oral_attempts", []) if doc_name and doc_name in documents_db else [])
    if not attempts:
        return ""
    avg = sum(a.get("score", 0) for a in attempts) / len(attempts)
    rows = "".join(
        f"<li><span class='oral-mini oral-{'good' if a.get('score', 0) >= 8 else ('mid' if a.get('score', 0) >= 5 else 'low')}'>{int(a.get('score', 0))}/10</span>"
        f"<span>{_escape_html(str(a.get('question', ''))[:110])}</span></li>" for a in attempts[-5:][::-1])
    return (f"<div class='oral-history'><div class='oral-history-head'><b>Mes dernières colles</b>"
            f"<span>{len(attempts)} séance(s) · moyenne {avg:.1f}/10</span></div><ul>{rows}</ul></div>")


ORAL_EMPTY = mascot_note("read", "La correction s'affichera ici.", "Note sur 10, points manquants et page à relire.")


def oral_reset(doc_name):
    return {}, oral_card_html(None), "", ORAL_EMPTY, "", oral_history_html(doc_name)


def oral_new_handler(doc_name, lang, last_state):
    """Nouvelle question : tirée de la banque préparée (instantané) ou générée par Gemini."""
    if not doc_name or doc_name not in documents_db:
        yield {}, oral_card_html(None), gr.update(), ORAL_EMPTY, "⚠️ Sélectionnez d'abord un cours."
        return
    doc = documents_db[doc_name]
    bank = doc.setdefault("oral_bank", [])
    last_q = (last_state or {}).get("question")
    fresh = [b for b in bank if b.get("question") != last_q]
    if len(bank) >= 3 and fresh:
        state = random.choice(fresh)
        yield state, oral_card_html(state), "", ORAL_EMPTY, "Répondez à l'écrit ou à la voix, puis cliquez sur « Corriger »."
        return
    passage = _oral_prepare(doc)
    if not passage:
        yield {}, oral_card_html(None), gr.update(), "", "❌ Document vide."
        return
    try:
        state = yield from _with_heartbeat(
            lambda: _oral_ask(passage, lang),
            lambda sec: (gr.update(), gr.update(), gr.update(), gr.update(), f"⏳ Préparation de la question… ({sec} s)"))
    except Exception as e:
        if fresh:                      # IA indisponible : on pioche dans la banque
            state = random.choice(fresh)
            yield state, oral_card_html(state), "", ORAL_EMPTY, "Répondez à l'écrit ou à la voix, puis cliquez sur « Corriger »."
        else:
            yield gr.update(), gr.update(), gr.update(), gr.update(), f"⚠️ {e}"
        return
    bank.append(state)
    del bank[:-30]
    _save_history(doc_name)
    _log("oral_question", detail=doc_name)
    yield state, oral_card_html(state), "", "", "Répondez à l'écrit ou à la voix, puis cliquez sur « Corriger ma réponse »."


def oral_grade_handler(state, answer, doc_name, lang):
    if not state or not state.get("question"):
        yield gr.update(), "⚠️ Cliquez d'abord sur « Nouvelle question ».", gr.update()
        return
    answer = str(answer or "").strip()
    if len(answer.split()) < 3:
        yield gr.update(), "⚠️ Écrivez ou dictez une réponse avant de la faire corriger.", gr.update()
        return
    reply_lang = "English" if _guess_lang(answer) == "en" else (lang or "Français")
    try:
        result = yield from _with_heartbeat(
            lambda: _oral_grade(state, answer, reply_lang),
            lambda sec: (gr.update(), f"⏳ Correction en cours… ({sec} s)", gr.update()))
    except Exception as e:
        _log("error", detail=f"oral · {e}")
        result = _oral_grade_offline(state, answer)
    if doc_name in documents_db:
        _progress_for(doc_name)["oral_attempts"].append(
            {"question": state["question"], "score": result["score"], "t": datetime.now().isoformat(timespec="seconds"),
             "page": state.get("page"), "section": state.get("section", ""), "missed": list(result.get("missed", []))[:4]})
        _session("oral", doc_name, score=result["score"])
        del _progress_for(doc_name)["oral_attempts"][:-100]
        _save_progress(doc_name)
        _log("oral_graded", detail=f"{doc_name} · {result['score']}/10")
    yield oral_result_html(state, result), f"✅ Note : {result['score']}/10", oral_history_html(doc_name)


ORAL_MIC_JS = r"""
() => {
  const area = document.querySelector('#oral-answer textarea');
  const btn = document.getElementById('oral-mic');
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const en = window.__akoriLang === 'en';
  if (!area || !btn) return;
  if (!SR) { if (window.__akBanner) window.__akBanner(en ? 'Voice dictation is not supported by this browser (use Chrome).' : "La dictée vocale n'est pas disponible sur ce navigateur (utilisez Chrome)."); return; }
  if (window.__akRec) { window.__akRecStop = true; try { window.__akRec.stop(); } catch (e) {} return; }
  const clean = (t) => String(t || '').replace(/\s+/g, ' ').trim();
  const setValue = (v) => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(area, v);
    area.dispatchEvent(new Event('input', { bubbles: true }));
    area.scrollTop = area.scrollHeight;
  };
  // Fusion des segments reconnus. Chrome sur ordinateur renvoie des segments distincts ; Chrome Android renvoie
  // souvent chaque segment final en répétant les précédents (texte cumulé) : on remplace au lieu d'ajouter.
  const merge = (a, b) => {
    a = clean(a); b = clean(b);
    if (!a) return b; if (!b) return a;
    const la = a.toLowerCase(), lb = b.toLowerCase();
    if (lb.startsWith(la)) return b;
    if (la.endsWith(lb)) return a;
    return a + ' ' + b;
  };
  // Indicateur « écoute en cours » sous la zone de réponse, avec le texte en train d'être reconnu
  const box = area.closest('.block') || area.parentElement;
  let live = document.getElementById(btn.id + '-live');
  if (!live) { live = document.createElement('div'); live.id = btn.id + '-live'; live.className = 'mic-live'; box.insertAdjacentElement('afterend', live); }
  const showLive = (txt) => {
    live.innerHTML = '';
    const dot = document.createElement('b'); dot.textContent = en ? 'Listening… tap the microphone again to stop' : 'Écoute en cours… touchez à nouveau le micro pour arrêter';
    live.appendChild(dot);
    if (txt) { const t = document.createElement('span'); t.textContent = txt; live.appendChild(t); }
    live.classList.add('on');
  };
  let base = clean(area.value);
  let empty = 0;
  window.__akRecStop = false;
  const finish = () => { window.__akRec = null; btn.classList.remove('rec-on'); live.classList.remove('on'); setValue(base); };
  const start = () => {
    const rec = new SR();
    rec.lang = en ? 'en-US' : 'fr-FR';
    rec.continuous = true;
    rec.interimResults = true;
    rec.maxAlternatives = 1;
    let session = '';
    rec.onresult = (e) => {
      let fin = '', interim = '', lastFinal = -1;
      for (let i = 0; i < e.results.length; i++) if (e.results[i].isFinal) { fin = merge(fin, e.results[i][0].transcript); lastFinal = i; }
      // texte provisoire : seulement ce qui suit le dernier segment final (le reste est déjà compris dedans)
      for (let i = lastFinal + 1; i < e.results.length; i++) interim = merge(interim, e.results[i][0].transcript);
      session = merge(fin, interim);
      setValue(clean(base + ' ' + session));
      showLive(interim || fin.split(' ').slice(-12).join(' '));
    };
    rec.onerror = (e) => { if (e.error === 'not-allowed' || e.error === 'service-not-allowed' || e.error === 'audio-capture') window.__akRecStop = true; };
    rec.onend = () => {
      if (session) { base = clean(base + ' ' + session); empty = 0; } else { empty += 1; }
      setValue(base);
      // Chrome coupe la reconnaissance après un silence : on la relance tant que l'étudiant n'a pas arrêté
      if (!window.__akRecStop && empty < 4) { try { start(); return; } catch (err) {} }
      finish();
    };
    window.__akRec = rec;
    btn.classList.add('rec-on');
    showLive('');
    rec.start();
  };
  start();
}
"""


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
    _session("flash", doc_name, result="review" if result == "review" else "known")


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
    "fr": {"card": "CARTE", "running": "Roulette active", "stop_hint": "Cliquez sur la carte pour l'arrêter.",
           "stopped": "Roulette arrêtée", "show_hint": "Cliquez sur le bouton pour afficher la réponse.",
           "next_hint": "Passez à la carte suivante.", "answer": "Réponse :",
           "click_btn": "Cliquez sur le bouton « Afficher la réponse »."},
    "en": {"card": "CARD", "running": "Roulette running", "stop_hint": "Click the card to stop it.",
           "stopped": "Roulette stopped", "show_hint": "Click the button to show the answer.",
           "next_hint": "Go to the next card.", "answer": "Answer:",
           "click_btn": "Click the “Show the answer” button."},
}


def flashcard_view(cards, index=0, revealed=False, doc_name=None, results=None, lang=None):
    cards = normalize_flashcards(cards)
    t = _FC_TEXT[_lang_code(lang)]
    if not cards:
        return mascot_note("think", "Aucune flashcard générée.", "Choisissez un cours puis cliquez sur « Générer les flashcards ».")

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
        answer_html = f"<div style='margin-top:20px;padding:15px;background:#eef2ff;border-radius:12px;border-left:4px solid #4f46e5;color:#1e293b;'><b class='ans-title'>{t['answer']}</b><br><br>{answer}</div>"
    else:
        answer_html = f"<div style='margin-top:20px;text-align:center;color:#94a3b8;'>{t['click_btn']}</div>"

    html_body = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body {{ font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:10px;overflow:hidden; }}
.card {{ cursor:pointer;position:relative;min-height:280px;padding:40px;border-radius:24px;background:#fff;border:1px solid #e2e8f0;box-shadow:0 10px 30px rgba(15,23,42,.07);display:flex;flex-direction:column;justify-content:center;align-items:center;transition:box-shadow .2s; }}
.card:hover {{ box-shadow:0 15px 40px rgba(15,23,42,.12); }}
.badge {{ position:absolute;top:20px;left:20px;color:#6366f1;font-size:12px;font-weight:800;display:flex;align-items:center;gap:6px; }}
.badge::before {{ content:"";width:15px;height:15px;background:currentColor;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='2'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Ccircle%20cx='12'%20cy='12'%20r='10'/%3E%3Ccircle%20cx='12'%20cy='12'%20r='6'/%3E%3Ccircle%20cx='12'%20cy='12'%20r='2'/%3E%3C/svg%3E") center/contain no-repeat;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='2'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Ccircle%20cx='12'%20cy='12'%20r='10'/%3E%3Ccircle%20cx='12'%20cy='12'%20r='6'/%3E%3Ccircle%20cx='12'%20cy='12'%20r='2'/%3E%3C/svg%3E") center/contain no-repeat; }}
.ans-title {{ display:inline-flex;align-items:center;gap:7px; }}
.ans-title::before {{ content:"";width:16px;height:16px;background:#4f46e5;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='2'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M9%2018h6M10%2022h4'/%3E%3Cpath%20d='M12%202a7%207%200%200%200-4%2012.7c.6.5%201%201.3%201%202.3h6c0-1%20.4-1.8%201-2.3A7%207%200%200%200%2012%202z'/%3E%3C/svg%3E") center/contain no-repeat;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='2'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M9%2018h6M10%2022h4'/%3E%3Cpath%20d='M12%202a7%207%200%200%200-4%2012.7c.6.5%201%201.3%201%202.3h6c0-1%20.4-1.8%201-2.3A7%207%200%200%200%2012%202z'/%3E%3C/svg%3E") center/contain no-repeat; }}
.status::before {{ content:"";display:inline-block;width:13px;height:13px;margin-right:6px;vertical-align:-2px;background:currentColor;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='2'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Crect%20x='3'%20y='3'%20width='18'%20height='18'%20rx='3'/%3E%3Ccircle%20cx='8.5'%20cy='8.5'%20r='1'/%3E%3Ccircle%20cx='15.5'%20cy='15.5'%20r='1'/%3E%3Ccircle%20cx='12'%20cy='12'%20r='1'/%3E%3C/svg%3E") center/contain no-repeat;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='2'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Crect%20x='3'%20y='3'%20width='18'%20height='18'%20rx='3'/%3E%3Ccircle%20cx='8.5'%20cy='8.5'%20r='1'/%3E%3Ccircle%20cx='15.5'%20cy='15.5'%20r='1'/%3E%3Ccircle%20cx='12'%20cy='12'%20r='1'/%3E%3C/svg%3E") center/contain no-repeat; }}
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
    mood, title, text = _cheer(mastery)
    return f"""
    <div class='quiz-result'>
      <div class='ms-result'><div class='ms-box'>{mascot_svg(mood)}</div><div><b>{title}</b><span>{text}</span></div></div>
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
    """Le choix se fait en touchant la réponse dans la carte : le Radio d'origine reste caché."""
    return gr.update(choices=[], value=None, visible=False)


def quiz_view(questions, index=0, validated=False, answers=None):
    if not questions:
        return mascot_note("think", "Aucun quiz généré.", "Choisissez un cours puis cliquez sur « Générer le quiz ».")
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
        last = index == total - 1
        swipe = (f"<div class='quiz-swipe'><span class='qs-arrow'>←</span><span>Glissez vers la gauche pour "
                 f"{'voir le résultat' if last else 'passer à la question suivante'}</span>"
                 f"<button type='button' class='ak-act ak-quiz-next'><span>{'Résultat' if last else 'Suivant'}</span> →</button></div>")
        body = f"{verdict}<div class='quiz-options'>{rows}</div><div class='quiz-explanation'><b>Explication :</b> {_escape_html(q.get('explanation', ''))}</div>{swipe}"
    else:
        rows = "".join(f"<button type='button' class='option opt-click ak-o-{i}'><span>{chr(65 + i)}</span>{_escape_html(option)}</button>"
                       for i, option in enumerate(q["options"]))
        body = f"<div class='quiz-options'>{rows}</div><div class='quiz-tap'>Touchez une réponse : la correction s'affiche aussitôt.</div>"

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
    if choice is None or choice == "":
        return False, gr.update(), "⚠️ Touchez une réponse.", answers, gr.update(), gr.update()
    idx = int(index)
    answers = list(answers or [None] * len(questions))
    answers.extend([None] * (len(questions) - len(answers)))
    answers[idx] = int(choice)
    ok = int(choice) == int(questions[idx]["answer"])
    status = ""   # le verdict est déjà affiché dans la carte
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
    review = f"<div class='quiz-miss-list'><div class='global-list-title'>À revoir</div>{wrong}</div>" if wrong else ""
    mood, title, text = _cheer(pct)
    return f"""
    <div class='quiz-result'>
      <div class='ms-result'><div class='ms-box'>{mascot_svg(mood)}</div><div><b>{title}</b><span>{text}</span></div></div>
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
        items.append({"topic": q.get("topic", "Général"), "question": str(q.get("question", ""))[:240],
                      "correct": a is not None and int(a) == int(q["answer"])})
    _progress_for(doc_name)["quiz_attempts"].append(
        {"score": score, "total": len(questions), "items": items, "timestamp": datetime.now().isoformat(timespec="seconds")})
    _save_progress(doc_name)
    _session("quiz", doc_name, score=score, total=len(questions))
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
LOGO_DATA_URI = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAQDAwMDAgQDAwMEBAQFBgoGBgUFBgwICQcKDgwPDg4MDQ0PERYTDxAVEQ0NExoTFRcYGRkZDxIbHRsYHRYYGRj/2wBDAQQEBAYFBgsGBgsYEA0QGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBj/wgARCAEAAQADASIAAhEBAxEB/8QAHAABAAIDAQEBAAAAAAAAAAAAAAUGAwQHAgEI/8QAGgEAAgMBAQAAAAAAAAAAAAAAAAQCAwUBBv/aAAwDAQACEAMQAAAB4sH1QAAAAAAAAAAAAAAAAAAAAAAAAAAACR6DyPLXW6QFbHZAAA92bozLXNpO1azzlSrPWpKHOGOgc/zs8IVAAAAAAAuVQ/VIvuvvq3AxZ/PjseSch/Xf5lq3K8OOrLXOsOXZYHWn97S+2XrcPinJoL9K8aZnXojzOt95EkI/zmIHOPn0AB8+gD0HTO2xMpPCy5POaMMcHFcDHetxuDHOdEh7ZU4OzVsr0puN47VUMrrXXKBG9bUan4K6cHzKYP3q4/Qd90q/UHz6IJrAAAAOjRdkkr1/3ybanldWw/nLuMbq3F9K5FfKx7uvHWXZuIdy5KrZ6nKbedd+Mya3t5iW/Sv5m7FmNc3r21GaXcvrWk5VatOko3zygLVAAAkY4ctOneb3erznU/UvL03uT92/Nv6F0MOHlax1BVqb5RdPzfVpykzVumP4/wCe5/Sja9O1625H+ivlu3fn8nbLRfnYa75y4a0tHyMFUAAABcKf3y1Otdh5vHtr/oD8w3oo1zbtFQt7eKj5CEhHqNQrGNXe0J/S9aWHWuXd+4bRpzGpDYXNiZzQBWM3oaarnz6KVgAAAAFx7DD7ep5WH29fDp3yO1HbFS8l9hagpy04fGxWxP8AjQ8OKe9PzgYc3oGa2qLOCpCP8/vAdAAAAAABbKn3C1WZ1tf3veU09DDgZ1pHVzQyjFg3IvUhXMx0Td6iqfJqpFth8xevrWWnzj8RXjeYdv47g6OkEGQAAAAAAsvV4j3tYu18xamgloZ8Ud16l7/ja8/o9tqkxMN+c3KBc4TneNx9psKXoq5tbWl6G6X361PtU5Yzd3Vucdbep5ZkOAAAACwV/rN9ctpa3v1ORvQ+3oynnz6fqXdisz+lRZPSnNLOvXZIyNgeGWerVhvdkN+vfLO5oqT2BnR+ecbE9OhdS5t5yOsMvgAAAEx0uJ8eko2dqPzaa2uxfZ2ZvsToKds3qlaicrnE1rCpZbJPn/on01zndYLt8rcw7ZteWJy7e08Yt9Rm7lpOeM+DxqQ+cPoBJxl1vLBoYPXto7ur61yOasWCiY3fKYu/n6tqb2/ej5mLpHTfHJ/n6yTvQF9iKkJD7etX+N96rlOjzCyx9esevPiO3fT6Xrzj+X26VauNO8rmhlLgC407dv7b/WXU9xf6+Y/UuROhpz/jlJnNn8M52fNFR/I2OTpWlLtmn6Ji7fc/NZVX2L7EbspQUdeKrCyAlYCTuamPGrl9Hq5KpLRXlsz4MpX78A+/AJGcqR2y3aEAYlnslUKL2P5XVnLGrjhZPlcyd79mcGjK2b+VxAsSujti91p2Upq6qmuS+Ry+b78IVAAAAAAAAAAAAAAAAAAAAAAAAf/EAC0QAAEEAgEDAwMDBQEAAAAAAAQBAgMFABEGEhMUFSEwECAiFjFAIyQmNGBB/9oACAEBAAEFAv8Al2tc98PH7aZP0pY6l43axpLDNBJ/ADAMsJoOC2T2ycCMRthQ2lYn3VdNKfjH11Sx9sS/PPMxlmU3POEMjs+Pdtvzceon3JsLBwhlVy57piSrrk/G4oofsp67zSDDenGpg4c07m8esVaVWlC45usFNdAt7WtVvyMY6SSvBZVU2azX0ZrLuu9LuvoiK5y9NdUpleK4ssUWAEV9/XMka6Eoa+rmhGLgE3XGcN4h/wAfC63yLRV6nJms1h9oBW5JzLTibKl5DlnWz1ZmVMfctj5OopMpz2V5xvJmEgI5yv46yRtTy17ehy+8EnbK5DH/AFfia1zn1gLaqkTExM5FeJUjLLOSQJxAmSIviM7Wukk9Iyk/3yl/vd4i5H+S0lE1YjrAauHs7GQ4tVzfvfe4fxcPrvJt1d1PTEyMhHT8l4+TaFVoDajJJdOgIlZnKhWn02VUnRYmf7W83g6/1GnAIyyNkJNV2byNOqa8k38QdPA0WLk9bXDs5tHt3Nheg7kdqYVAssxpUqxjTTG2diPwOF0ZnED65KG0Q3LQFa61Y5Y5SNTC7zeCtdLOnE/7YuN8JG83gbfzPm753wA9nzo4Dr41a5z7D9E2bo3cNumNoxBmWKO9r8hR6TiVW0Cj/ZL3kFtVG2hzJSeTCIbVZXl9OTQ9td4JOsBCcwA8U0lSS94xiyOMJQeD4kT0ji3EqmMep9mt5ar0pPMncRC7cHJ2q/jNc9rq+9cSygvT7KzLcHLHx4DS09gIoNjk7tDtRXL0vbnWufk7GwrkxjIGucr3fDx8Dyzb9qvpamVq1i9MkZ9MU2aatcHSD/gIREwkTjR72CpOipLThJJeJHK9+m5yILyK2JNzvRHq8uOJG2C56hHi2GSFTSfJXh+mcaljaSNXn3IQKE8lmySK7mZILZmTdWbw4BDHpYXQjXnXpmCAxVrXOyBWvYaK8CyfLI/+Bxqv826nf3ZEXS/hI1IZG4jXYjsV6NbNfVkKx8hrZHNlVMfNI7HOxzsRdJyUVCa/+BUh+mUG8Ib0yNdiOxHYUWwQRALO7fFS0QqeDSyY3ojjV+Ofifk/eQKx6GiuCP8Am49X+fckS9yfeGS9Ef8A4j8lsxB3wFwEM695puF2IwLWfqIuEQ9xCKuDzbn3nVpeRC+QD81ON6Zx/eJ7uIk7s7HK1z9LnHFFTLiojjgjMY8OOY61IApQKrLS6ZCgDFbks/czq1nc7ke8HVr2mDPDN+SiA8+4Km7pG8lf0C5GiJkxUkh4JMi3Q8idEPEhEnYyIYaxKKMOl8gKwlnmnkHm7kO8Ek9951Ki8gHQgP5Kgf02g3jducS9Hz5aSPYBFUnS4wN4V0O/8Gy46X2IlZGyZYy75/GAZsTj1oE73RUVUd19bcHVr2lDuFM+KmB8+2Mm7hO8R3bgxFRF7q51rlivaOil0iT4s+SydWWr0mJ3kRZEGOLENQqpdHGO/N5tUW8gSYb4qiD06j39CXfl9d4VEk0AxbwljnhkSQkeJJbSSbAxe3J9N5HNJC97YTVdtrsiVr2kQOHJ+CrD86zNn7hG8iXoTf039j4kcjq6BVaBA3GDO0jelPfN5vN5vO4krV2i70tvF3YfgqIvBpt5vJndLcVzWtfYCMx1umOtSVxTy3YpBC51vXOp2dx6YhRLcZZmMxlzkZYc+K1U+m83tN4xWubNGsM/3AC+ZYGS9U28j11K5XOT3W0l6iPoFxcHw049QJknGqORLDjRQjMraQ2zwfilQxq8b4/p3E6aRSYHimQGTj5CZCVi7TN5vN5ZR9cf3cfb05vN452o8dIkUKuVznV0rRIR457h5PdmSXO7neVq2AA0nJWyojUlxZsdNlroawJqum6buOSEn2XN5vHfmN91E/P2xP3V21yym9gBfLOYW1S686uDqm28S56rPiXYSu7jXRku/wAhY5XY+5roXesTOxbfWLaUxQSSOJprJjX5C7aRS9OLm8V3TB9whCilyakTftiKjUlessrOoSr8Ibw0dEPkhzVxTIM8tXp3FFmefDLZSlqc9s6jp5sCrGfEmd9CMhEEhl7XRP8AlHKi7SN/thUv4/eMZKMqGhyZ3xcLJjUeHt+ROoBEvarM7NVnZq87NVnZqscILJArHtcOG3s9mqzs1edmrzs1edqsxrK5rz3xSlRvRE7rUx5Pt/DZI+N6WMupSJZv+R//xAAuEQACAgEDAwMCBAcAAAAAAAABAgADEQQSIRMgMQUiQTJREEBCYRQjM1JxgbH/2gAIAQMBAT8B/LFgPMDg+OynSl+TwIukp/eNoaz9BltLVHDd+ou6S5js9hy0G5OVmj1HVXB8/hSmTkxPd5guY8VrK7RZ7SMGMBYvTaMCpwe7WXdR8CKsp0ob3N4m/TqcA4MQn5lZ9sreI61rmV5ZzaYX5mr/AKp7bbSwK1DMGiuzkiWUGoAmVvZaOmI3p/HmaVyM1P5EVsRWntZRuMtfnEByZc25yewjImnc2E7fpEKYE9RBVgZ6ZgpmWPjiau4KyWrFYEZE8QOMYaElzHfYMDz2628VV/5mhtCLtMt1tRGA+DNfqFbCg5mg1opO1vBj3I6+6ay4WsFTwJ6dbxsPxHtH6YLmEa1j3eqajqWYHgQDeodZYGMq0e/3McCGnSgfMFf2ldOOZS3TfdAc89+rt6dZMZcnM06bExNiFS5nQVhur8Rq0qGbT/qVrvXJXH2nSyMQJNM3G3v1zb2xK6stAu47YSmNoiX2UEqsps/mbiMmdRVxug2tyJZVzmBcc91jbRGXMSvAmyVDa0u0gJzKdJjxCnxBXFJ8NAkK47beZ08zbOmYKjApmJiDECRUj1ZHaVgSBeY+oVZf6hqyx2nE0/qWqU+/mXa4hAaxyYtmpb5lJsH1yqz5iYabZqU2t+KnBhWbJaeOI21htxBplP6f+z+CX44i0cDMAx4m5pVz7ZU5ERgwzNU+58DsWwrDcSMYhXM2/vNv7wCe0TcPtN4+06o/thfLboLmAwPyn//EACsRAAEDAwMEAQMFAQAAAAAAAAEAAgMEERITITEFFCBBMhBRUyIzQGGxcf/aAAgBAgEBPwH+NdX8IaV8iHTU/pxHCfG5hs7z52QAatjyiLbfSjg1Xb8BbMbco9UmkdaBiouo67tOQWKqIBI2ycMTY+TB7TimrH6dPbaK6e0PaWu4KOhQRkhUEb56juCLBFV7bTHxx+6LgmjJQhrBm9d807EKRoG7eF012URZ9kCupNhfjqusm2DQG8IfqNlWSakziPGUCNu/KZJk6yxFlUk2sqePbO1wtIbtCgmdBJkFKLOsFUUrJrag4TGOOzQqyqbC3TjN3eMLMnhVsBcdlDE6K4cNimtv6UsOQTGGM3aVTRH5FVDNOQOPCqOrX/ZFv7TeqVA9qWvnk2c7ypIrDJWuLpwKldjsOUBOfSbH901iqYdRiItt5wx5usg3EWTW2CqJSLCPlU82pcO2IUtVvjFuVEXteA53/VjcIBV8Gm+/nQQWbkVjc2U79NhcmNlyDj91PDcqKIjjZFkjr24VHNqR78q26q4NWOyIsbHxp4jK+ybHiLKNvtFqqIc2EBNeyTnYp0kce3JVNE4Xe/kp1Ox2/tDNny3Csup0+m/Me/HpdNi3Mpzb7I7KStgZy5P6tD6upOowv+Ud0zqUbPjGm9WjPIUVbDJwfoG2VVTiaMtTgWmx+tNHqSBqazEWCA9rrM+LREPaEZQoABujQA8KOkLiQfS7IKek0wCFTVcsBF/iopA8bKy6vCI5ch7+tNLpyByY4PbcI7BSv7md0h4XbSMIcnPd7f8A4mVTmfLcJtSxmZG9yjI48uQyeLZJ9MQ3F3v/AH0qGc2t7CjeHtuurziSXFvrwp6+WDZp2UvV5ZGFluVDU6TS3EFd438YXdt/GE2qjJs+MW/pDto/1A5f0u7b+MLu2/jC7xv4wu5OoZQj1Ga2Ldv4n//EAEQQAAECAwQGBA0CBAUFAAAAAAECAwAEERIhMUEFEBMiUXEjMkJhFCAwM1JicoGRkrHB0TShJEBDglNgY+HwFYOisvH/2gAIAQEABj8C/wArhKElSjkBWK+DbMf6irMeclvmP4ioZQ57C4sPtLbVwUKfyOyk5dbys7OXMxV+Zl2e69cdFpBhR4KSRFqaljs/8RG8nxw64S2x6WauUbOWaFvOzj7zG4EI/ePPqjeKV8xGxm2k0OS7xBf0fVScS1ifdx8ubZKJVvzix9BAlpJlLTaeEXk6qL3hCtKaMRRsXusjs948W26OgRj6x4R4PL7tLiR9BqstNqWeCRFfB6c1COmZUnvy1WF3t/SDpCWHe4B/7eVS2hNpSjQDiYZkUdYCqzxVn4pQoVSq4gw7KjzfWb9k6wlIqTcIQw318K9+Z1NsIxUaQGmkhKRiePOLNtSvWSm6LSSlxtY+MdH5pYqnu7tSpZy8ZcuEOMZA7vLyitIODo5bq+3/ALRXxKTLlXMmkYx0WjkU9ZcJb0ihySeTch4GoHONg9RQN6HE4LHEam64J3oCfRGoTC2y5QEUBpDjCJdSCsUtWouiq6gKVVMS6O1vHUhXfDL3EFJ8mEpFVG4DjDMkOvSrh4qz8QNM0M06N31R6Udt11w8yowFz0zsCf6aBaMVkpoOn0HBZPxh/Rc+hSXZbpGbeKfSTqX7H3hfiJm5xNQb0N/cxadVf2UDEwp5z3DgNbXt/byfhrg6KWvHt5RXWtkiytN9OI4w3NyakWwmwpCzSCQUuTZuLowT3J/MdK9Q+sqKhdtPOsJ0iynpGOt7Ofw1D1gRFeI1iABOS9B64hx1Sq2j+2tKe+Gm+avJJntMTHgzCuogddceDaN0c7s61310r9Y6TRh/tc/2jo9HOlXrLFID3hK5cJ6rbJoB+Yl3XBRwS42vM5QaYm4QdHaI7PnHuEVm5h5a8zWkGY0RNLNP6as4ck5puw9Si2znDsr2QaoPFOUJcTik1hLyOfu1pbRepRoI/WHa09HdhbTgopBsnXtOFwhSh1RujyKVzAq0jfUn0qZQ7MLcSLOKlYJ7hAkpNSph7OgoBAU26yo5ih+sEqQzd62P7QRNtq8KZVXYuC6KjOHngd4C7mboaUU9K6Noo6kJMvKIYX1ColVr8Qzp2Va2EwhQS+2MDwUPpDekmRVTYv70HV4Os3HqxaHV+mpDqesghQi0WXdr6GXxhx9XWWoqOqg+MbFvrEfDyZyeOPtGETLiavP75rwyipwj/qLOkXkIBTZbbXRK68oani4tamSAbV5pCaYUh+mRSf3hgpwLafpD7smopdbosEdxqYbVNiqEjo9mm7/7DheFkhsEg5VWKCJdp0VSplII90OyxwSd08RlqSisUArGB1XAxvRYaoVfSCpRqT5JcwsdFLptnvOQg0yUDEuU4bNP0gpOYpHganLK2bkpcVuqHEQ+pSkrWaVs4J4CGknEIA/aHZdzquJKTB0XNbszKblDmnI6iuWddlwewjCGNAyNS46sLdUbz7/rFhHVTcITPoG+yaL9mEDvgWjhFlsVP7RvN/COoqN1v4mKFVBwHlAwR0zu+5zhbCsFikJkm5BKlNkgOuromkX6TkJfuQ3aijumZJ7udla/aJaXnlSfgjS9oRLizaPLWh9l4y0231Hh9DFmc0St/wD1ZU1BixI6KcYr/VmMoWvbeETjvnHuHcNSmHBVDgoRDkuewd08RkYopX8glaxVpjpFc8hB4YRSKL+aN1xBEbzg92q0ogAZmKeEbQ/6YrFC6tv200i0heOYMby1H36xDekWxvN7q+X8ihKhR97fXqtjPxFPrqQnIZxtZ10Ssvkk/ZP5jpW3ZlXFxdB8BG5o1ke9UBttISlIoAMvFXLuiqHBSkOSyuybjxGXl0WxVprpF/YQeAuGpKczFpOGqw69vcBfFplxK/tGOreNV+gnGNvL6PS21ltLifjC0Ot7N5s0UnVs8lYaqiG9ItjeRur5eXFoUff3ld3/AAaqQVZYCLs8oUlChbpWkOpmJVh561ftk2oOl9EI2Cmr3WRhTiITME2U0vrlBldEtE+k8q4J/EbZ2k3N4l1zqp5QVKXaOQ4w7PzhDe2NwMUFyYugOjtY89S5Z29DgpSHJdfZOPEeVQlQq03vr/EGmAuGonNe6PvqLirkphU0lRSom4jKG3VG9ZsqpnFFCqVpsqHGKvTrzjANQ0BT94DEs0llodlEKkNHt21p6yjclPOHEzTYXMJOK96kbR1xSlcTAVnnqUye1eOeqoxhvSDYvTur5eV2ihR+Y3uXDUEjExZT1Ubo1CWaSSt3EJGUease2aRLNuKSok2t2BrK1KCUi8mETWkWXES0yaJNaGmAMfwekSk8HKGD0aX2zm0a/tFCKHgYBGIgLGepcs6KocFKQ4wrsn4+TQ0odGnfXyig6qbhqW9n1U89d0YxLTWSTfAobtdCKxKyiDVaVWld2ro3VDuyixpCWBP+KjERt5Nzwlnu6wix7xqqIRPIF6d1fky+oUemL+Qy/OsMjBv65+LZVHg80Ds+ysRVuZbP90Vdmmx3A1MFrR7ZHF1WP+0WibSsVK8S20spPdFtNlmZ/wDFf4MWVAgjEHUqXcvQsQtlWKT5Ftk9TrL5RZT1UXDUp49jDn49FJtCOqoco6hVzi5NkRQDxrDp3h1V/YxQxWETaReLleRVNK869hyy/OtLPC9XPVaWsJHfHXK/ZEdGwf7jG6G0+6PPkcovfc+aL3FfGOsfjFy1fGN2Yc+aPOBXtCOmYH9pjddsq4Ku8ZTS70qhTasj47bHZxVyjZp6qLtVpWCb4KjidQaGCPrrZc0i/MB1abRaboLMfp5hfN2Nzwpg8Qu19YL0s4JtoY2RRQ92q23ZbZGLrmHu4x/ETEzMK9XcEfpXx37YxRqZm2ieNFQ7LOjebVZMbq6p9FUU6i+HipmE5XHx5l/MADWEcbzqU6chBUrExLu2hbmDRtrM98S2ixfKyCdo8fSVCnOOuoN8SqQmyiZvWlMBCAEoTclIwGvGJXTqUhSF9DMp4/8APtDkm04kJItsk9sReLxcRFly9PGLsNbiO7x5hj0hXVfFdSWBzMJa7OKz3Q/pdQ6OXGylk98KD8yTMPqtuWE1PKP4fRc293m6L9BPU7lGLDzUxKq9YVEbRtxLiPSTGjT3QaYDEnKLAW5Mr4Mi74x0egn1D1lH8R0+hZpscUKh6UXMOtB0Uo8jqnjdCVJP8VIG4jNENz7Q3H8e5UWYsnq61q7vHS8MsR3Rt2jVKr9ZWcBCnDnGzSOnmruSYaYWLSW8q0qY6JEkz3qvMb+lzyabi7Sk9FkaRafHoTTf3jaSwMus4t1tNr5GJKY6gQDbHowErthjsS7eK+cUTMysiPRaG0X7zG/piePIUjc009/3W6xRxyQmR33GNs20G1UoQlVQYf0Wrza99o8IoRRSTeIqIsnVsh7/ACFE3ozSYqqrZ748+I2bS7VcYRtTuVvjaOTF+Fxi+YV80efPzR+oPzR+oPzR+oPzQrwN0qUMqxZUgg8oU5NlTaeFaR+oPzR+oPzR58/NHnz80fqFfNAUJhVReN6No0qtRfSKGK2ooj4/ylpCqGOqmN9V3D/KP//EACoQAQACAQIFBAMAAwEBAAAAAAEAESExQVFhcYGREKGx0TDB8CDh8UBg/9oACAEBAAE/If8A5fTCRePaAMpuXs1gtknhImB8pfDU56i6/wDDXXNjHtDvD9/22Hij3lk/6Y+Ymq+/5U071/mpbPZ/LnGPCTJx0xeRb3mW/aPqJYTh9c4LxR7+0fcXLQc/w16/ntw03C7c59jtNHdQffm82I27qw2FOjDoxykeEFRjDgcPY4m2v+Iq63H+HWV7wHR6+CWMB8/xPCwB+Zoj8R8jEZ5h1n3en6h0M8DTg/fn8qUhD6o0EqZwHvavOOhA9DCTMMbYGMccr3fS8advW+ggOLAs1TFvq/jl6cTWc4cWHCC115lKLPdZEHOqYCYnqle6HmDoCaO+6E1y4uL0/IOK7Voox4W9yXsiEMAW3LyH14HWEx5P9iFFE5ftuTqdyYK1rnpQaq7fEx71GscD5c/UUpuCaw7zL/5Fo39peop+jJ4UZ8wVs+Dg9A1npR6OIQYz2Fk+X8aJ2gNU6Eob0JuZ+nQ/wCSpG5ZpW/R9TA+vXPlhYMjAerp4uddBi6aPNR7jAjWcOijZ09Kjb3rwnmf1CFJYS5XqNjRP4xKJin6acInYLgdNomSYWIxG/wB34wfiw0V+mXxL7btFLpYMSuLI3ykUrLnMFRHu4hxYV1jvf77olgLy33ZScstHmbKjrqvR3U9L9AO/yv6gp7Sf1CCQyaXmDmAB/wBoh9LWdNh4isYo7cwyTunx+Jsed7JNjt4ly65IXu7qmmI5f7g7duhvZcFuRs/fdZeNNUoCv3bRB+/TGfx9GzXY2xl2l8ryIz8L5Y8HzUO29Hokq6BH0bXDpwSN+20NzK/XaacAEDLYFOqEEDJQFxVog6w4Qy4ce8ueKPBGMMWtaeRBS34U/wB3+ECn14QLO5o7wGDulexHxNI4vfi9t2Dwk0EOlMzHkLC8Ni3QYb1un0wHZZXxlLJY+FA+YFakrW0wdjHmOpL6A3xXUdLZMc9YMl8t26jcQV2lVgmv8Kc930s6qydnhGpc4IWukh4jcxG00Bl3aSm3CGlr6NKBvwSw9VsbOPX8WdvaKUoNv+NP1DzAF1cD99WZCAC7dpXkhUVajku+rglypA+gXcqyCtXodNo0Z3PSsUYuCuFJrQ+y0PsDMECDTnV8dkdrqxshc5SC2W46fEsqbCburxAzFcS4Fd6JZkuUV59JLMrtHNujgazoWZp1RKyrV/FnZyjTR82e0ac773P3H9L1eESfhlK2UKQ2O3EVpe7zVZuuYHVzutJrFVcL37axj7SuxnE/1KbYjtFexag9o6S5VpMAYt0vkA4xV4ROmYJlzwLdfT8s6E4qJWyUBcr9oHSebqcT2YVZ5A+Bv5MdRdJbdijzHPrP8OD5mr10pcdd+0zwnE07owwNMhRGSBw6Rgf5vNV+nr2D57pDFW+Saz+oJtUtRz00fMfe5eltfs9I5TjN8UniN4376eCC2E4GD/waEas0ftZ7Q13wIjvUxDabsNSNd9NQntJCqI+1LJQRTiNl9+kOMHRcecwQUAs1UNoTn6lZg0mZwQM8b+n5/wDD55CFmnYx1X0cBPn/AIPi/oGonQhzXo7p7nWA5AOUVV3C75g2cMYcPWHUbasYBo4lvjPtL/39wPH5+dFmjT7j8MtA36AoQFbzwINlLXpVYA1FR1qKghqGvURY1Xxm65moI01HXhOIQYg8rF+PSfQ1Rbo3YD6MduO0YsCUjZCxK0nE/p+fz2nuWsYO3uYRcuKUP9UlHFmDxQNTetycGpvAhtByvTNz9PRPnDifULClksLcleO4Acb/AOuBAb9OWXkflz0jAX6nQbHOWSuxbzf/ACJh07cesEhVJoym1mHDdGB2vUumfaZS2o8B8fl9w8YadzXvMoX/ALyEf6ujd+pUabRtWWC8MpGx4h8c9FZYv4YlXsRsTM1eCM9G3uE0o1lR34y2Oafb5nflHS26tFxDR5bTnNo06cJk7g6vRnDHgvs9AG1BsZhOdJ4eH5/LW6wOIrDsZ7wjUClEvBZdXi930ZR2QXi/RKa63cvZrDGg1qMvHpPCTBrOImY9lwHOIzIakK+BjnMVwPfSxd1BmDuzG24NQpI6dJYwGABuuDvLlvCEumZkDah4Nnx+PPx4lt3aILLwb3hG3HuW/YgwiuQnCVLdVHqLoeV/cPKFkeJMWsxawMBfZLmiMp46x4I2Zo2cTfgwzowYf3RQDdfek1U4fQBCUmScF9HDbw/P49JQC9f9C+6EDbRlh3uCnnvfrtLly/QmLHHhzglwaW6P2fEpS+gfE/tCVRMiPj4KCuz5EuX6BvEPd1ldNXjDwDMNtQMuA5ZCTI3QvibP4UANGzl186d4GK0o0vf67QgdADuen32lltbZcIuDNSnSOSLsly9dwjnvkADkdemh5H/AMVeAUv8AFc9pcBSYSDAOTMHhk/Dbw/P4RMVXnt/D4QgirNfKfRLnM6yqaGzgjC4Rzp+J7WFvma10UJqo7o6498/6KGifRzSD3TVA8ljLSjj+phFF1p7znRylxhqXqaMYAqiiTVuqvibP+bIam3gdfrvDOkOgaX/YhDGyr8+BHBtLZoJbfHnq9Kxe0ERvWFtC01qF11znxHt9qAeBM9+sJ8/r6CVrVhfuXSGncOH2+8ZU5JZDOIxcviVwH9t1mP70H+ppjnnXpCtMYYuNjmAQz4e3+YWDDfd/RLLblcvoqzX/AEpc0n7usbi0tecsu0YNXFwNJTbwR67ffHmMpuxyJg1jzTScN43raYXbdcLqAJTQQmPWcyNtD04lrNlvU+EbKY5pcgPH6iyoSxFAl2m4nGrW/oYAhov4/wAxS7Ij3P2TKp1IzkbxHW8u2pQv/eIreO2HX6hqOVOw++8oI425dtOfeYYri09hlJZOK18T+qh3GVTjo9n+pbw7+WJKhkRR1M/vG1bXtKF8eigeriH5J4g3UbW0YTtsuvP71XjrMKhWTa1JleppEzcvaLn6DY4oaf5DZZhxFrCQBsj5jWvGXFJobWarDvpMwr6m/wDD7yx/laLeWYck3vq+ZoW5I+5b1XGv9RsweAuOLwCt/jD5m7Fjm2f4maGfJlt1e3Gf0dHN5jmD40viaHXgQ+J4cD7pe8FVU7JANN1btv7lKXhBAD0GfAJcHJ5/X8DrJZ/S4Qmzeg+o7PllAluRsTYjU8pbiVphA5YnEX85T+p9T+d9T+l9T+N9TXA62L7TlARgKcmMjv8AUqP5e0/lfU/jfU4q/wA5SvQP5ygbz2cfiUwdNXHb1uQVg7RWgnNNW3/xknSDVZ42xk0mhwH/AMj/AP/aAAwDAQACAAMAAAAQAAAAAAAAAAAAAAAAAAAAAANAYAgCqdqAAAAAAJjxMAVOlDLLIQAgE3VeRg4tDhEBAAAAMCP9Qu2w7OKKgAACLuSRBS6obm0AIAAOFU4BofVNQbkAAAAE8N7RI6CCAAAAgQDuhFktHO/UAgQAAA3KsJteLQAHIIAAArrWTvtS5Nv3OIEAg0TbIuPQHpxgUjEAFrGKQbSBVeIyZyAAPSy1eFKTAeWtM6MIUqiSQzHMCTAzssgAwYAgQAAAAAAQwwwQ/8QAKBEBAAIBAgQGAwEBAAAAAAAAAQARITFBIFFhsRBxgaHR8ECRweHx/9oACAEDAQE/EPxtRVMGuAO+ACm0Jyj1lF/94yXfLUQvZUXYXgW2kVrWCOtoPNijQI69B5MUaxxBsxCZ9lMpeqC++kQVrPfrDKHOACOjKOwd2Vaoz8SixCWG9duG6g0vbrn4nMh1IzGWCnwa+URsfsRrA7EsV4DVVcovQJVErjguCIWCUHlvMpcRLY+0uN99qgVtVzaSUepCWYcxLVKwNJZAgo9rXpwoM5wI9UKbzyZXgEIu+VTrgPaOVrc5Mo1E8yvCSjfvEU2K6zcoPS8S2/0++0LXE1LtKrT93sG8pxfriPaaIQtGAwADR46Y1cERFKZeszho2NYnta9mEPLbmApz+0UqKjmsTNt243o6EoL2iAN+28e0Fp/UUChjvSK+fOFwKU9OpBlblde8dg2hnhsWXLZmOcS5YYLpLsgOxBAO0qbMMUa3X58BMeEOEbKjWDaHgjbwfOANYVC28HQIlNPiFtS+b0otCLMmb6B/uIgVtsQRYHUmMQ3naZ9r6QZcvmKBrGMMQkbjfxoFmOEKI1uD59NYGhVjpg0N3XLjWf8AU/iW6beqe4J7zGjQ/rFGPf8AhUqKTHr/AGZEGHTzmcNSEzC5BwHUZIwoLhK7nnS/NKXKwMl3DlYcrAdnvEQFXEyUP4n/xAAoEQEAAgECBAcBAQEBAAAAAAABABEhMUFRYXGBECCRobHR8MFA4fH/2gAIAQIBAT8Q/wAyDWAfJlKol5lh1uVo89KBMEEdES/4MrAXsAFsTWg5K+2kf7I1qcmPkZiprHmpF7ymWS3SJSwiVA5tZo5FMQBQumqveKMxXu4o482KBX3p9vLhwSnRBtfCLMpsS/UqNbiDuSvs+DUVrdUXrU24ArpDQIGgro7Y8g03D6csyuVBxzGqHqbkATLLRZSNyzYUNcOUKa5qceUCt126xCxfZDTCPSKLBhTQOBz8rEOMCIu49b9S8aJiNwAwE0wjSnMzd4wRWoWWHOd2vY0IHWfUIDYrgY+PNnWsawitZbovgmYqEFpGZVllgbkRK187GIYDQjDExCr15bdYVXuobS5m0uZXo4H8zDElpmJR0fPjrLCj3eChgkEa8VlqkBz6oJlQ/llc6GHwhTcR21DymTAM6Ew8/wAbQXWJrvU6mSYN06jqMSo2bGWAkpbrhwI7hXEYZpXzNzqf09IBJWBj5eXG+X9+6zE4/iANuAmGHtn4gMBdv+z5kV9TbTon1Ncz0YhR3zx8wp0l/GkYuu3WKNY8be0WGe1L1i1Wcnofb8ROE4e8CZrANQx+KtUz7+sbQTCZ3as/zhLdBAnQPufjxJ/QYIbWIudCbMvwYDv/AGWCgmeOXXgY0M7QDu9R/WM2vSv1FPiIZLB6GYrdXRD5b9iBhuOzT8N+0G0bB5P6MekNa/R+xCBhVLDXffyH5nAx8gBVl38y9SLeb27xXX3/ALn/AI79yjAtW1nMtivYtkUd3ftPzv3P3v3Asnu/cpIBdtpcpR1qLf8Aj//EACoQAQACAQMEAgEFAQEBAQAAAAEAESExQVFhcYGREKGxIDDB0fBA8VDh/9oACAEBAAE/EP8A4Ffqr5r9uv1V81/1JnSmldBaw1vKGVdy/SdyGL7IPkNn3axH5bul2sz4v/hTV62M7pQ9xF1QtXuSLOyj4XLAr0o19S4+KSl7XjhcTX9OnD4aJyFxWy8cCwISlUX9bPj6Iim0/wAvh9S/0AZ6gfysQvmkFS9I1eiLfXHeUio5E16g6ux0RsUcP6q/YblANTsrcVZXqaoTPRVOLunVt2Vl+L1IEt7lCMlMaKrUbwnRnKwTUztF6ehQIB81KWo6hvBeKyuKN5YUCWAhWDQGFNNCMRzGIEuww5a0OrCzGH8GQLb9ovp4LvtEgKjK81WvV6OfSUldBMrSjudVnL9V/rL9ftgAdVQizvvfvnwdAmGggpgnTibaQVglInCWPeeS7Sib3cn1+bLAx3Gg9pBA4LUX8ta/pFpvHoAwyw6ocAL4mBEJxQzvH6NqIwLLF98cKdQZjCUaHMIj6R0gBVBzdxq7cFE6JxKEQOgGyxMN0z6WCNba9Q/TT1H9wWzkDlVv/EYo9otBwbfDglsP0BWG2RYNyi9rgQAcKrPSfcO6Z6/FSjNBqozqMW5x3baJ6s1HkRY1hYx5/mIuSrA8j6gxIZEMhAhkjoX7mM58FkpQC3Y8wAil6RGMvaupTohTnXeYbCw3HV7R9Q71BBQHqZH0zLAi5sfR8Ffsiuy+0qA5VDzDGfa6rfF0ej8dcpBCAJBtg3bw2FFcK6++pTvKn+xATYAfSuhemDmLh3Q/Qgt3ijrEhONaZpnF2lusVHWXHo962Ouu08UhTTH4ZVkMK5ByjqcMvE0ABqYsBsdTg+pQsF+xaP3buqy9NxYagiRC6V93fx+2apAY4Rp429SX380OA0+AOWDKhdQGSgALou5smoyhU5dSHpeJVbJuR68rUWKDwpegA153UFvshTtn8Ayp8yzsSIzgJWuB6ozvFeo87VPuE2obyhb8SqdaKBggW4XMP6FFYBRCjmrWCULgKURlV+FMyqPa8/Vy241nBQPy/s7XCoxV2eo0KhpSUbwpj8ZwQEOsKBq6BggAc5UngPzKcu1TdUKvEWUOF5kUG+ZvelBiC/MFR9hs2XhH1SpjUdSdi4OVcA1C1EaHVZbMC4OBNB+VD9ogsioBBsAdpDzLPYQlLczYKnRTbT/3hEXFZdVPeZUjdedPMoyTVuGfT+GdX4FHj/sge0hSMbUo19F49q2iEHZbtBOuT4epGL4Gzdavg/MxuYmzqPlL1+nb5xM0ju1L/HhQvcC63ujOAHCtLW2OjSgLzW0qNxQaZYFvLajiyd1EIQsDajYWgvVDrBLDowCMeUGnBYWZgZbA7w1etRihZbhHi74lcBfZWrOqDy3YhFgCXBKgNUSJCxAlakwfCx86RTM5Evdi2ZHQzQb8x2Q0mA1xZhNX0dTrfMSKV9uHpxKTWBJJJYGFnFkQdVaq6fuVfSNlRvSUodC6j1zTY6xgf30g/HQrL6rqzXlm2Pi/m/i5cLNBVxWp6QPTfW4B1rR5w0qnWFl3gERui6EV4KWgAthwNgRoBCm6wswazcNCO9blkFrQpGvoV8rP0SEqogcAfyRYhP0WY/MGqQzApWiZg4SyHIfX13MrJIG3YoqHqsSqVwKW1tSbTa4EMgV3VRue7W34D1GIgDLg7zPpeLdQr3hFB22f1K0YNws9kqKVOkNRF4UHdTVr+oiAmNRuVu9Pcvooa1f2nN8drS/lH065eEO2XJ7Esshq4qTUMQOaSn8xo9MGXMbFnBujSQE2mapBUtKrgdhZgjBXgT9jE4rVBUUeSvCW74NknswQvi+8YsBSsicVFSqP9daqnoNGwQtqGyMv0lRQYDdRStRsZ9BM3RUMvMvfvtxQ1NGzsNv4gFFyjFry7RINYp0Hru8e4UD3J+jf5ghhvFr+ZjEuzSegi2yanVOrq+5Ur9mrwC9Ay9oZOzbqAb9ruxZ/ha7ufAB8Q032NcAWNVojsNpSNW/ejJe4iladXo2TxCJVmnIWHUs0HdUbK2uWd8LKlCaDQzOVBygoiNRugKCn6JS//BFmxR7wGw5rtjYiItQVpctuqy0aABq5mhtRIvsLHqAn3dFXA7ovrZB/DMJ3DX5qY+dv2Q0vRYTk7m3RwHWY3bnPlVmMFVeIxEn/AKm06Q0q/WY6LK6/2wAwBtBthZ4eVcEuDVSNHsr7RGgV5UhB5gXYYkEsRGk6x3x8yeoAOZrZiSqQTZ1giq6Bl1j3/T4/er40ywMbbDlCvXgAI9BvPTd/c0cw+YNGYM4jdtQ9VTO2WMOLeibMgrrF7YxAgMypnpVR3uPkQ2g/NhDCXVHQXtCpzNa2JqoydJ1poNRusgebeQj2pW5r58qPN/tbfO3wOgQI5S8N9Ib36Oacvlv6ndHxMeLr8VuPDD40G48MfRfcYJinV4ooejmOFMuUm1pZ3qMMS1tmBrlOq7xIVi6zxOw9XwMpGozOgjrcgHaOkttYDaWDkyIjo942KZaaBHEgL9hyvOTyShRKTUguoyGyaQfVAzOkXv126f3sUq0BbFu9lhRZPve2YaIRKFtXQN3wQbrTBxiPevmHbUEaf0Y2c4A5xE1VZr0mcoQlXAaApdC7S3SYdZTV7tt4KWgOWKJ5YsI9ETriKycDdWIocLagmCNUb9vWnW6tEFnLk/iup+5Ub+zcugOW1Aa0XvFFjMLV8/0jFVCGomjAeBoGzg+89mYNYePcsWgefsEseXVvZPuh9/s1+izShcYWWeN2jcS6tGnLy/QTqxTGmec1eivKOEJOjpRQq+C2PRqrDwSdB9yqY4E12QYuxK3LhF9ANTB9ssR6aycttQgXkgTG12d1ap3W13WMDLIVN3SEXjPsuJpvYGSzeIinA24Y7LWu0aDoEBwLO2dfevmd80xBk2LT2HgjbMY2EDZNJiQwjNnJ7qdjx81+yoFrQasCwaWlJ+0vdJhAcUBcrFGSF0P5B9SpU+8AjTiG79kbQ/5VP0jxfzgDBYLZFz5pKiMEqLbCqlDVdftxFf2PEK0LhEoyWLqXWxZm+LPxMptPVwaKRmYwO0KK+mEdRySlKBNkbIFZgDwPDceuDM3UF0Hk+wjdq7Tq+dD7/a2l1bbawufe7C8QDjcGAu4847BOrMqYfE5Hke6QAraEeoujmOWi+Lu2I9FwYipQVXB/AwwCYG2JkTxKxcm1g9JIawUbMMHomyuzydRF4xLhTdc5jBQ+ZewlRCUPef0tOkL+Mgw9PWuQvkJRnPYbnk/EYtasQ2TSUOAB5XJ7idhx+0axbkdFVFr4SdRNHMczUaA3eIQwsY0bPsqBJJ1w6o9K+KavYdRio8pRcQfRr1EEs5ogd0ieo1GmxTtYzhBRUOjTubeAjFJZNvN0LnLquWMssEXu5o4GidGDLu/JXJzo7xPrUVboSMmYQXqUnk+wje2Rr1vIIysfG8r5v4cqrbZxHlQi64Go7g6YB0jggHDZD0B8NwuFI2rqu7CcGsLxesUULRSM40HJ5P5iY9OtekYKpHDJ6KhBB0Fa7BKesGquV5YDnFyFxyrSJ5jNh0iDBeXIbJvw1dmlmjrGoylkgHhhBqAuVye4nYS8frIYm6qznp+SE3V6zqMRdQ4tyweg7rO+dJQ8e2GIY2R7aPuLPBAfQP5lpS9VnlfxL4CdvwZGXP8AI/hj3sl/mZLz/wCt41a/I/5lXQm1h6VlVTNrTyUwVvlmf46wYsaHm4ND4Zmjv/1jzYHMcTsa5Dh7beonMN8qm9lJ6+yLkrh5XkKf1VLiCvOp83R1ECzYnsBrsB4TqRqV1l4fkaJYwBuVcxIB1mXCwTSjPoo8stZqIUaoaQQjEkZqh0L63xPPg96oQySaE+6L9w3RrKNvc2G63tKtwxlZQTcagMDjA3SG6Bi3PQBpFyr/AJypEoxQbXTUv3K/+SsXQdiUnRI8OHW7x7+DBLMqG8Lq3+mUR7PPwdSUdQNjDmBAnJy8Nnkm36ijFZ41f8bQYfUDuusOU0FKdnQfl8/HNYoHLY8tER++7dNsYtrsWKG0Rab5uNFUGGhtb3XLasTum8eg9QYAkAK8wDEEoOYpOvdVuqawYlyqPAASqGCN4CaJtD+FgwVgb0++WJm0egq5wst5RbcgSkRpHqJ9Re4AXIbXyfcQAXMA3FQZmBpO9v5CGS+f1NQBNzQsGwdYEdkjBbHPYTW7V9ukEA1YWByqvR/L6jaI8Uzle14HeZ3bwzZaHQX/APCMLkqGW1AHKs6si3AVukerVhEOqY0jvVU8Q/Bl47y6MS4h3wAQguUwEz0is63GD7I3PGjh9Q/fGBfAn3LBFz6DJVLAc8Tlt3MhTsXoilDgSVN3pe48y8fdtuceI7nJv/uII4sOjHLWMLQfif5YKB+jeViMuYm1xB33OoQtzl1f8s2bhUXOrt8bBSI2At+phLTR2bHgolbSQGNEHS/8VAVdam6aKrbWs4GE9KiOu6PXig7NP7gW+QT1SaetNX0YY74hScrN3INp7l8BxLO1qwgUhyN91hrNer1lx2w5srY5jbMriP8AgzLHuL0wgo6yG7qX7gQWsa97f4w68FY7KMJ4NI9UV7m08FPfzFYkGbI0kSKwsYzeyZXTiOccLzYhtufyfENPm4yyOY+A2wxfK3fTuS0MaiF97D9S7n/50g9Vmq05p7teBhWCy0WUzT0Wh6MFkhCstho1ZeV3Se5S80k+JTcdU+Ej6kIb00pSmjH4O1cG4W1gVovXFxoGqFAU9BLfiLkx603/AGgZ2EdKyXauSQNkkHRLGjge6VfaElBkfsgYWGTJj9a6652IqiKrau/w6/Br8Hwnxbz85lszyzMz8AgJVmicJuTsRefVzMwtltc1u9WW8y3rMy2ZvWZ+KlSvh/Vc1/5rxLz8X8PxXzt+i/2L+b+b+b+b+b+Np//Z"


CHAT_MIC_JS = ORAL_MIC_JS.replace("#oral-answer textarea", "#assistant-input textarea").replace("'oral-mic'", "'chat-mic'")


def _logo_favicon_path():
    """Écrit le logo dans un fichier temporaire pour l'icône de l'onglet du navigateur."""
    try:
        import base64
        path = os.path.join(tempfile.gettempdir(), "akori_logo.jpg")
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
        return mascot_note("think", "Aucun cours sélectionné")
    pins = documents_db[doc_name].get("pins", [])
    if not pins:
        return mascot_note("read", "Aucun chapitre épinglé pour ce cours.", "Dans l'assistant, cliquez sur « Épingler comme chapitre » sous une réponse utile.")
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
  background:#1b2438!important;color:#e6ebf7!important;border-color:var(--ak-field-border)!important; }
html:root:root:root:root body.akori-dark .gradio-container .wrap, html:root:root:root:root body.akori-dark .gradio-container .input-container { background:#1b2438!important;border-color:var(--ak-field-border)!important; }
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

/* NB : Gradio recopie chaque règle avec un préfixe ".gradio-container … .contain" qui lui donne plus
   de poids. Les états (replié / survol) passent donc par des variables CSS posées sur <body>. */
:root { --sb-open:264px; --sb-rail:72px; --sb-left:max(16px, calc((100vw - 1280px)/2 + 32px)); --sb-ease:cubic-bezier(.4,0,.2,1); --sb-dur:.32s; }
body { --sb-w:var(--sb-open); --main-ml:calc(var(--sb-open) + 16px); }
body.sidebar-collapsed { --sb-w:var(--sb-rail); --main-ml:calc(var(--sb-rail) + 16px); }
/* survol du rail : le menu s'ouvre et la page glisse en même temps (même durée, même courbe) */
body.sidebar-collapsed.sb-hover { --sb-w:var(--sb-open); --main-ml:calc(var(--sb-open) + 16px); }

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
  width:100%!important;min-height:clamp(24px, calc((100vh - 345px) / 16), 40px)!important;margin:0!important;padding:0 12px!important;
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
body.sidebar-collapsed:not(.sb-hover) #sidebar .nav-title { opacity:0; }
#sidebar .brand { display:flex!important;align-items:center;gap:11px;width:100%;margin:2px 0 14px!important;padding-left:3px;overflow:hidden;white-space:nowrap; }
#sidebar .brand-mark { flex:none!important;width:42px!important;min-width:42px!important;height:42px!important; }
#sidebar .brand > div:last-child { flex:none;width:170px; }
#sidebar .brand-name { white-space:nowrap; }
#sidebar .brand-sub { white-space:normal!important;overflow-wrap:normal;word-break:normal;font-size:10px!important;line-height:1.35!important; }
#sidebar .nav-title { white-space:nowrap;overflow:hidden;padding-left:3px; }

/* écrans étroits : rail fixe, ouverture en surimpression (pas la place de pousser la page) */
@media (max-width:900px) {
  :root { --sb-left:10px; }
  body, body.sidebar-collapsed, body.sidebar-collapsed.sb-hover { --sb-w:var(--sb-rail); --main-ml:calc(var(--sb-rail) + 12px); }
  body:not(.sidebar-collapsed) { --sb-w:var(--sb-open); }
  body:not(.sidebar-collapsed) #sidebar, body.sidebar-collapsed.sb-hover #sidebar { box-shadow:0 14px 40px rgba(30,41,80,.2)!important; }
  body.sidebar-collapsed.sb-hover { --sb-w:var(--sb-open); }
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
/* Icônes du menu (traits fins, couleur du texte) */
#sidebar #nav-home::before { content:""!important;width:20px!important;height:20px!important;font-size:0!important;background:currentColor!important;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M3%209.5L12%203l9%206.5V20a1%201%200%200%201-1%201h-5v-7H9v7H4a1%201%200%200%201-1-1z'/%3E%3C/svg%3E") center/contain no-repeat!important;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M3%209.5L12%203l9%206.5V20a1%201%200%200%201-1%201h-5v-7H9v7H4a1%201%200%200%201-1-1z'/%3E%3C/svg%3E") center/contain no-repeat!important; }
#sidebar #nav-courses::before { content:""!important;width:20px!important;height:20px!important;font-size:0!important;background:currentColor!important;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M22%2019a2%202%200%200%201-2%202H4a2%202%200%200%201-2-2V5a2%202%200%200%201%202-2h5l2%203h9a2%202%200%200%201%202%202z'/%3E%3C/svg%3E") center/contain no-repeat!important;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M22%2019a2%202%200%200%201-2%202H4a2%202%200%200%201-2-2V5a2%202%200%200%201%202-2h5l2%203h9a2%202%200%200%201%202%202z'/%3E%3C/svg%3E") center/contain no-repeat!important; }
#sidebar #nav-review::before { content:""!important;width:20px!important;height:20px!important;font-size:0!important;background:currentColor!important;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpolygon%20points='12%202%202%207%2012%2012%2022%207%2012%202'/%3E%3Cpolyline%20points='2%2017%2012%2022%2022%2017'/%3E%3Cpolyline%20points='2%2012%2012%2017%2022%2012'/%3E%3C/svg%3E") center/contain no-repeat!important;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpolygon%20points='12%202%202%207%2012%2012%2022%207%2012%202'/%3E%3Cpolyline%20points='2%2017%2012%2022%2022%2017'/%3E%3Cpolyline%20points='2%2012%2012%2017%2022%2012'/%3E%3C/svg%3E") center/contain no-repeat!important; }
#sidebar #nav-flash::before { content:""!important;width:20px!important;height:20px!important;font-size:0!important;background:currentColor!important;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Crect%20x='3'%20y='5'%20width='18'%20height='14'%20rx='2'/%3E%3Cline%20x1='3'%20y1='10'%20x2='21'%20y2='10'/%3E%3C/svg%3E") center/contain no-repeat!important;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Crect%20x='3'%20y='5'%20width='18'%20height='14'%20rx='2'/%3E%3Cline%20x1='3'%20y1='10'%20x2='21'%20y2='10'/%3E%3C/svg%3E") center/contain no-repeat!important; }
#sidebar #nav-quiz::before { content:""!important;width:20px!important;height:20px!important;font-size:0!important;background:currentColor!important;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M9%2011l3%203L22%204'/%3E%3Cpath%20d='M21%2012v7a2%202%200%200%201-2%202H5a2%202%200%200%201-2-2V5a2%202%200%200%201%202-2h11'/%3E%3C/svg%3E") center/contain no-repeat!important;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M9%2011l3%203L22%204'/%3E%3Cpath%20d='M21%2012v7a2%202%200%200%201-2%202H5a2%202%200%200%201-2-2V5a2%202%200%200%201%202-2h11'/%3E%3C/svg%3E") center/contain no-repeat!important; }
#sidebar #nav-summary::before { content:""!important;width:20px!important;height:20px!important;font-size:0!important;background:currentColor!important;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cline%20x1='4'%20y1='6'%20x2='20'%20y2='6'/%3E%3Cline%20x1='4'%20y1='12'%20x2='16'%20y2='12'/%3E%3Cline%20x1='4'%20y1='18'%20x2='12'%20y2='18'/%3E%3C/svg%3E") center/contain no-repeat!important;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cline%20x1='4'%20y1='6'%20x2='20'%20y2='6'/%3E%3Cline%20x1='4'%20y1='12'%20x2='16'%20y2='12'/%3E%3Cline%20x1='4'%20y1='18'%20x2='12'%20y2='18'/%3E%3C/svg%3E") center/contain no-repeat!important; }
#sidebar #nav-oral::before { content:""!important;width:20px!important;height:20px!important;font-size:0!important;background:currentColor!important;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M12%201a3%203%200%200%200-3%203v8a3%203%200%200%200%206%200V4a3%203%200%200%200-3-3z'/%3E%3Cpath%20d='M19%2010v2a7%207%200%200%201-14%200v-2M12%2019v4M8%2023h8'/%3E%3C/svg%3E") center/contain no-repeat!important;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M12%201a3%203%200%200%200-3%203v8a3%203%200%200%200%206%200V4a3%203%200%200%200-3-3z'/%3E%3Cpath%20d='M19%2010v2a7%207%200%200%201-14%200v-2M12%2019v4M8%2023h8'/%3E%3C/svg%3E") center/contain no-repeat!important; }
#sidebar #nav-assistant::before { content:""!important;width:20px!important;height:20px!important;font-size:0!important;background:currentColor!important;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M12%203l1.9%205.1L19%2010l-5.1%201.9L12%2017l-1.9-5.1L5%2010l5.1-1.9z'/%3E%3C/svg%3E") center/contain no-repeat!important;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M12%203l1.9%205.1L19%2010l-5.1%201.9L12%2017l-1.9-5.1L5%2010l5.1-1.9z'/%3E%3C/svg%3E") center/contain no-repeat!important; }
#sidebar #nav-progress::before { content:""!important;width:20px!important;height:20px!important;font-size:0!important;background:currentColor!important;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpolyline%20points='23%206%2013.5%2015.5%208.5%2010.5%201%2018'/%3E%3Cpolyline%20points='17%206%2023%206%2023%2012'/%3E%3C/svg%3E") center/contain no-repeat!important;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpolyline%20points='23%206%2013.5%2015.5%208.5%2010.5%201%2018'/%3E%3Cpolyline%20points='17%206%2023%206%2023%2012'/%3E%3C/svg%3E") center/contain no-repeat!important; }
#sidebar #nav-reviewq::before { content:""!important;width:20px!important;height:20px!important;font-size:0!important;background:currentColor!important;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M19%2021l-7-5-7%205V5a2%202%200%200%201%202-2h10a2%202%200%200%201%202%202z'/%3E%3C/svg%3E") center/contain no-repeat!important;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M19%2021l-7-5-7%205V5a2%202%200%200%201%202-2h10a2%202%200%200%201%202%202z'/%3E%3C/svg%3E") center/contain no-repeat!important; }
#sidebar #nav-history::before { content:""!important;width:20px!important;height:20px!important;font-size:0!important;background:currentColor!important;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Ccircle%20cx='12'%20cy='12'%20r='10'/%3E%3Cpolyline%20points='12%206%2012%2012%2016%2014'/%3E%3C/svg%3E") center/contain no-repeat!important;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Ccircle%20cx='12'%20cy='12'%20r='10'/%3E%3Cpolyline%20points='12%206%2012%2012%2016%2014'/%3E%3C/svg%3E") center/contain no-repeat!important; }
#sidebar #nav-admin::before { content:""!important;width:20px!important;height:20px!important;font-size:0!important;background:currentColor!important;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M12%2022s8-4%208-10V5l-8-3-8%203v7c0%206%208%2010%208%2010z'/%3E%3C/svg%3E") center/contain no-repeat!important;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M12%2022s8-4%208-10V5l-8-3-8%203v7c0%206%208%2010%208%2010z'/%3E%3C/svg%3E") center/contain no-repeat!important; }
#sidebar #nav-settings::before { content:""!important;width:20px!important;height:20px!important;font-size:0!important;background:currentColor!important;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Ccircle%20cx='12'%20cy='12'%20r='3'/%3E%3Cpath%20d='M19.4%2015a1.7%201.7%200%200%200%20.3%201.8l.1.1a2%202%200%201%201-2.8%202.8l-.1-.1a1.7%201.7%200%200%200-1.8-.3%201.7%201.7%200%200%200-1%201.5V21a2%202%200%201%201-4%200v-.1a1.7%201.7%200%200%200-1.1-1.5%201.7%201.7%200%200%200-1.8.3l-.1.1a2%202%200%201%201-2.8-2.8l.1-.1a1.7%201.7%200%200%200%20.3-1.8%201.7%201.7%200%200%200-1.5-1H3a2%202%200%201%201%200-4h.1a1.7%201.7%200%200%200%201.5-1.1%201.7%201.7%200%200%200-.3-1.8l-.1-.1a2%202%200%201%201%202.8-2.8l.1.1a1.7%201.7%200%200%200%201.8.3H9a1.7%201.7%200%200%200%201-1.5V3a2%202%200%201%201%204%200v.1a1.7%201.7%200%200%200%201%201.5%201.7%201.7%200%200%200%201.8-.3l.1-.1a2%202%200%201%201%202.8%202.8l-.1.1a1.7%201.7%200%200%200-.3%201.8V9a1.7%201.7%200%200%200%201.5%201H21a2%202%200%201%201%200%204h-.1a1.7%201.7%200%200%200-1.5%201z'/%3E%3C/svg%3E") center/contain no-repeat!important;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Ccircle%20cx='12'%20cy='12'%20r='3'/%3E%3Cpath%20d='M19.4%2015a1.7%201.7%200%200%200%20.3%201.8l.1.1a2%202%200%201%201-2.8%202.8l-.1-.1a1.7%201.7%200%200%200-1.8-.3%201.7%201.7%200%200%200-1%201.5V21a2%202%200%201%201-4%200v-.1a1.7%201.7%200%200%200-1.1-1.5%201.7%201.7%200%200%200-1.8.3l-.1.1a2%202%200%201%201-2.8-2.8l.1-.1a1.7%201.7%200%200%200%20.3-1.8%201.7%201.7%200%200%200-1.5-1H3a2%202%200%201%201%200-4h.1a1.7%201.7%200%200%200%201.5-1.1%201.7%201.7%200%200%200-.3-1.8l-.1-.1a2%202%200%201%201%202.8-2.8l.1.1a1.7%201.7%200%200%200%201.8.3H9a1.7%201.7%200%200%200%201-1.5V3a2%202%200%201%201%204%200v.1a1.7%201.7%200%200%200%201%201.5%201.7%201.7%200%200%200%201.8-.3l.1-.1a2%202%200%201%201%202.8%202.8l-.1.1a1.7%201.7%200%200%200-.3%201.8V9a1.7%201.7%200%200%200%201.5%201H21a2%202%200%201%201%200%204h-.1a1.7%201.7%200%200%200-1.5%201z'/%3E%3C/svg%3E") center/contain no-repeat!important; }
#sidebar #nav-logout::before { content:""!important;width:20px!important;height:20px!important;font-size:0!important;background:currentColor!important;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M9%2021H5a2%202%200%200%201-2-2V5a2%202%200%200%201%202-2h4'/%3E%3Cpolyline%20points='16%2017%2021%2012%2016%207'/%3E%3Cline%20x1='21'%20y1='12'%20x2='9'%20y2='12'/%3E%3C/svg%3E") center/contain no-repeat!important;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Cpath%20d='M9%2021H5a2%202%200%200%201-2-2V5a2%202%200%200%201%202-2h4'/%3E%3Cpolyline%20points='16%2017%2021%2012%2016%207'/%3E%3Cline%20x1='21'%20y1='12'%20x2='9'%20y2='12'/%3E%3C/svg%3E") center/contain no-repeat!important; }
.akori-hidden { display:none!important; }
html:root:root:root:root body.sb-instant #sidebar, html:root:root:root:root body.sb-instant #main-column,
html:root:root:root:root body.sb-instant #sidebar * { transition:none!important; }
html:root:root:root:root { --sb-dur:.16s; }
#user-badge { cursor:pointer; } #user-badge:hover .ub-text b { color:var(--ak-blue); }
#user-badge .ub-avatar { transition:transform .2s ease; } #user-badge:hover .ub-avatar { transform:scale(1.08); }
img.ub-avatar, img.ad-avatar { object-fit:cover; }
.prof-card { display:flex;align-items:center;gap:20px;border:1px solid var(--ak-line);border-radius:18px;background:var(--ak-card);padding:20px 24px;margin-bottom:14px;animation:akfade .4s ease both; }
.prof-avatar { width:84px;height:84px;border-radius:50%;object-fit:cover;background:linear-gradient(135deg,#4f6df5,#8e6cf4);color:#fff;font-size:34px;font-weight:800;display:flex;align-items:center;justify-content:center;flex:none; }
.prof-card b { display:block;font-size:19px;color:var(--ak-text); } .prof-card small { color:var(--ak-muted); }
/* ===== Animations d'accueil ===== */
@keyframes akfade { from { opacity:0;transform:translateY(14px); } to { opacity:1;transform:none; } }
@keyframes akgrow { from { width:0; } }
@keyframes akfloat { 0%,100% { transform:translateY(0) rotate(0); } 50% { transform:translateY(-8px) rotate(4deg); } }
@keyframes akglow { 0%,100% { box-shadow:0 0 0 0 rgba(79,109,245,.35); } 50% { box-shadow:0 0 0 10px rgba(79,109,245,0); } }
.hero { animation:akfade .5s ease both; }
.hero-orb { animation:akfloat 4s ease-in-out infinite; }
.mini-stat { animation:akfade .5s ease both;transition:transform .2s ease,box-shadow .2s ease; }
.stat-row .mini-stat:nth-child(2) { animation-delay:.07s; } .stat-row .mini-stat:nth-child(3) { animation-delay:.14s; } .stat-row .mini-stat:nth-child(4) { animation-delay:.21s; }
.mini-stat:hover { transform:translateY(-3px);box-shadow:0 10px 24px rgba(42,55,90,.1); }
.mini-stat small { display:block;font-size:11.5px;color:var(--ak-muted);margin-top:3px;line-height:1.35; }
.guide-grid { grid-template-columns:repeat(auto-fit,minmax(250px,1fr))!important; }
.guide-card { display:flex;gap:14px;align-items:flex-start;text-align:left;width:100%;padding:16px;border-radius:16px;border:1px solid var(--ak-line)!important;
  background:var(--ak-card)!important;cursor:pointer;animation:akfade .5s ease both;animation-delay:calc(var(--i) * 70ms);
  transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease; }
.guide-card:hover { transform:translateY(-4px);box-shadow:0 14px 30px rgba(79,109,245,.16);border-color:var(--ak-blue)!important; }
.gc-ico { flex:none;width:42px;height:42px;border-radius:13px;background:linear-gradient(135deg,#4f6df5,#8e6cf4);color:#fff!important;display:flex;align-items:center;justify-content:center;font-size:18px; }
.gc-ico .ak-ico { margin:0;width:21px;height:21px; }
.gc-body { display:flex;flex-direction:column;gap:4px; }
.gc-body > span { font-weight:400!important; }
.global-metrics { grid-template-columns:repeat(auto-fit,minmax(190px,1fr))!important; }
.gc-body b { font-size:15px;color:var(--ak-text)!important; } .gc-body > span { font-size:12.8px;line-height:1.45;color:var(--ak-muted)!important; }
.gc-body em { font-style:normal;font-size:12.5px;font-weight:700;color:var(--ak-blue)!important;margin-top:4px; }
.act-list { display:grid;gap:12px;margin:16px 0; }
/* Vue d'ensemble : liste des cours, même langage visuel que les activités */
.gp-courses { display:grid;gap:8px;border:1px solid var(--ak-line);border-radius:16px;background:var(--ak-card);padding:14px 16px 10px; }
.gp-courses .eyebrow { margin-bottom:4px; }
.gp-course { display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr) 3.2rem;align-items:center;gap:12px;padding:8px 10px;border-radius:12px; }
.gp-course .gp-cname { font-weight:600;color:var(--ak-text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
.gp-course b { text-align:right;color:var(--ak-text);font-variant-numeric:tabular-nums; }
.gp-course:hover, .gp-course:focus-visible, .gp-course.gp-active { background:var(--ak-soft); }
.gp-course.gp-active .gp-cname::before { content:"● ";color:#8e6cf4; }
.act-row { border:1px solid var(--ak-line);border-radius:16px;background:var(--ak-card);padding:14px 16px;animation:akfade .45s ease both; }
.act-row:nth-child(2) { animation-delay:.08s; } .act-row:nth-child(3) { animation-delay:.16s; }
.act-head { display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;color:var(--ak-text); }
.act-name { font-weight:700;display:inline-flex;align-items:center; } .act-head b { font-size:18px; }
.act-bar { height:9px;border-radius:99px;background:var(--ak-soft);overflow:hidden; }
.act-bar i { display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#4f6df5,#8e6cf4);animation:akgrow .9s ease both; }
.act-off .act-head b, .act-off .act-name { color:var(--ak-muted); }
.act-foot { display:flex;justify-content:space-between;align-items:center;gap:10px;margin-top:9px;flex-wrap:wrap; }
.act-foot small { color:var(--ak-muted);font-size:12.5px; }
.info-chips { display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 14px; }
.info-chips span { background:var(--ak-soft);color:var(--ak-text);border-radius:999px;padding:6px 14px;font-size:12.8px; }
.next-step { display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap;border-radius:16px;padding:14px 18px;margin-bottom:16px;
  background:linear-gradient(120deg,rgba(79,109,245,.1),rgba(142,108,244,.12));border:1px solid var(--ak-line);animation:akfade .5s ease both .2s; }
.next-step p { margin:4px 0 0;color:var(--ak-text);font-weight:600; }
.next-step .ak-act { animation:akglow 2s ease-in-out infinite; }
.progress-src { display:block;color:var(--ak-muted);font-size:11.5px;margin:-2px 0 8px; }
.gc-detail { display:block;color:var(--ak-muted);margin-top:2px; }
.global-course-bar div { animation:akgrow .9s ease both; }
.global-metrics small { display:block;font-size:11px;color:var(--ak-muted);margin-top:3px; }
/* ===== Colle orale ===== */
#sidebar #nav-plan::before { content:""!important;width:20px!important;height:20px!important;font-size:0!important;background:currentColor!important;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Crect%20x='3'%20y='4'%20width='18'%20height='18'%20rx='2'/%3E%3Cpath%20d='M16%202v4M8%202v4M3%2010h18'/%3E%3C/svg%3E") center/contain no-repeat!important;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Crect%20x='3'%20y='4'%20width='18'%20height='18'%20rx='2'/%3E%3Cpath%20d='M16%202v4M8%202v4M3%2010h18'/%3E%3C/svg%3E") center/contain no-repeat!important; }

.modules-wrap { display:flex;flex-direction:column;gap:14px; }
.sem-title { font-size:12px;font-weight:800;letter-spacing:.08em;color:var(--ak-blue);text-transform:uppercase;margin-bottom:8px; }
.mod-grid { display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px; }
.mod-card { border:1px solid var(--ak-line);border-radius:16px;background:var(--ak-card);padding:14px 16px;animation:akfade .45s ease both; }
.mod-head { display:flex;justify-content:space-between;gap:8px;align-items:baseline; } .mod-head b { color:var(--ak-text); } .mod-head span { color:var(--ak-muted);font-size:12.5px;white-space:nowrap; }
.mod-bar { height:7px;border-radius:99px;background:var(--ak-soft);overflow:hidden;margin:9px 0; } .mod-bar i { display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#4f6df5,#8e6cf4);animation:akgrow .9s ease both; }
.mod-chips { display:flex;flex-wrap:wrap;gap:6px; }
.mod-chip { padding:3px 10px;border-radius:99px;background:var(--ak-soft);color:var(--ak-text);font-size:12.5px; } .mod-chip.mod-active { background:var(--ak-blue);color:#fff; }
.mod-empty { color:var(--ak-muted);font-size:12.5px; }
.ak-datepick { width:100%;height:40px;border:1px solid var(--ak-line);border-radius:10px;padding:0 12px;background:var(--ak-card);color:var(--ak-text);font:inherit; }
.plan-head { display:flex;justify-content:space-between;align-items:center;gap:14px;border:1px solid var(--ak-line);border-radius:18px;background:var(--ak-card);padding:16px 20px; }
.plan-head b { display:block;font-size:19px;color:var(--ak-text); } .plan-head span { color:var(--ak-muted);font-size:13px; }
.plan-count { font-size:26px;font-weight:800;color:#fff!important;background:linear-gradient(135deg,#4f6df5,#8e6cf4);padding:8px 18px;border-radius:14px; } .plan-count.plan-past { background:#94a3b8; }
.plan-kpis,.jr-kpis { display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:12px 0; }
.plan-kpis > div,.jr-kpis > div { border:1px solid var(--ak-line);border-radius:16px;background:var(--ak-card);padding:12px 16px;animation:akfade .45s ease both; }
.plan-kpis b,.jr-kpis b { display:block;font-size:26px;color:var(--ak-blue-dark); } .plan-kpis span,.jr-kpis span { color:var(--ak-muted);font-size:12.5px; }
.plan-note { color:var(--ak-muted);font-size:13px;margin:4px 2px 12px; }
.plan-day { border:1px solid var(--ak-line);border-radius:16px;background:var(--ak-card);padding:12px 16px;margin-bottom:10px;animation:akfade .5s ease both;animation-delay:calc(var(--i,0) * 40ms); }
.plan-day.plan-today { border-color:var(--ak-blue);box-shadow:0 0 0 3px rgba(79,109,245,.14); }
.plan-day.plan-exam { border-style:dashed; }
.plan-day-head { display:flex;justify-content:space-between;margin-bottom:6px; } .plan-day-head b { color:var(--ak-text); } .plan-day-head span { color:var(--ak-muted);font-size:13px; }
.plan-rest { color:var(--ak-muted);font-size:13.5px;padding:4px 0; }
.plan-task { display:flex;align-items:center;gap:12px;padding:8px 0;border-top:1px solid var(--ak-line); } .plan-task:first-of-type { border-top:0; }
.plan-task-done { opacity:.55; }
.plan-ico { width:28px;height:28px;display:inline-flex;align-items:center;justify-content:center;flex:none;color:var(--ak-blue); } .plan-ico .ak-ico svg { width:20px;height:20px; }
.plan-main { flex:1;min-width:0; } .plan-main b { color:var(--ak-text);font-size:14.5px; }
.plan-doc { margin-left:8px;padding:1px 9px;border-radius:99px;background:var(--ak-soft);color:var(--ak-muted);font-size:12px; }
.plan-pages { display:flex;flex-wrap:wrap;gap:6px;margin-top:5px; }
.plan-page { padding:2px 10px;border-radius:8px;background:rgba(245,158,11,.14);color:#b45309;font-size:12.5px;font-weight:600; }
.plan-min { color:var(--ak-muted);font-size:13px;white-space:nowrap; } .plan-done { color:#16a34a;font-size:13px;white-space:nowrap; }
.plan-more summary { cursor:pointer;color:var(--ak-blue);font-weight:600;padding:8px 2px; }
.jr-strip { display:flex;align-items:center;gap:12px;margin:6px 0 12px;color:var(--ak-muted);font-size:13px; } .jr-strip > div { display:flex;gap:4px; }
.jr-strip i { width:16px;height:16px;border-radius:5px;background:var(--ak-soft);display:block; } .jr-strip i.jr-on { background:linear-gradient(135deg,#4f6df5,#8e6cf4); }
.jr-adapt { margin:0 0 12px;padding:10px 14px;border-radius:12px;background:var(--ak-soft);color:var(--ak-text);font-size:13.5px; }
.jr-sub { font-weight:700;color:var(--ak-text);margin:6px 0; }
.jr-gap { display:grid;grid-template-columns:minmax(120px,1fr) 2fr 44px;gap:12px;align-items:center;margin:6px 0; } .jr-gap small { display:block;color:var(--ak-muted); } .jr-gap b { text-align:right;color:#dc2626; }
.jr-timeline { margin-top:12px; } .jr-day { border-left:3px solid var(--ak-blue);padding:2px 0 2px 14px;margin:0 0 14px; }
.jr-date { font-weight:700;color:var(--ak-text);margin-bottom:4px; } .jr-day ul { list-style:none;margin:0;padding:0; }
.jr-day li { display:flex;align-items:center;gap:10px;padding:4px 0;color:var(--ak-text);font-size:14px; } .jr-ico { color:var(--ak-blue);width:20px;display:inline-flex; }
.jr-doc { color:var(--ak-muted);font-size:12.5px; } .jr-day li em { margin-left:auto;font-style:normal;color:var(--ak-muted);font-size:13px; }

/* ===== accueil : héros illustré, guide détaillé, pied de page ===== */
@keyframes akrise { from { opacity:0;transform:translateY(22px) scale(.98); } to { opacity:1;transform:none; } }
@keyframes akblob { 0%,100% { transform:translate(0,0) scale(1); } 50% { transform:translate(14px,-10px) scale(1.06); } }
@keyframes akflow { 0% { background-position:0% 50%; } 100% { background-position:200% 50%; } }
@keyframes aktwink { 0%,100% { opacity:.25;transform:scale(.7) rotate(0); } 50% { opacity:1;transform:scale(1.15) rotate(20deg); } }
@keyframes akbob { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-9px); } }
@keyframes akpulse { 0%,100% { transform:scale(1); } 50% { transform:scale(1.07); } }
@keyframes aksheen { from { transform:translateX(-120%) skewX(-18deg); } to { transform:translateX(320%) skewX(-18deg); } }
.rv { animation:akrise .7s cubic-bezier(.2,.8,.2,1) both;animation-delay:calc(var(--i,0) * 60ms); }
.hero { position:relative;overflow:hidden!important;gap:10px;--mx:70%;--my:30%;
  background:radial-gradient(420px circle at var(--mx) var(--my),rgba(142,108,244,.20),transparent 60%),linear-gradient(120deg,#eef3ff 0%,#faf7ff 100%)!important; }
.hero-glow { position:absolute;width:300px;height:300px;right:-60px;top:-90px;border-radius:50%;background:radial-gradient(circle,rgba(79,109,245,.22),transparent 68%);animation:akblob 9s ease-in-out infinite;pointer-events:none; }
.hero-text { position:relative;z-index:1;flex:1 1 55%;min-width:0; }
.hero h1 { font-size:clamp(26px,3.1vw,38px)!important; }
.hero-grad { background:linear-gradient(90deg,#4f6df5,#8e6cf4,#e86fb0,#4f6df5);background-size:200% 100%;-webkit-background-clip:text;background-clip:text;color:transparent!important;-webkit-text-fill-color:transparent;animation:akflow 6s linear infinite; }
.hero-chips { display:flex;flex-wrap:wrap;gap:8px;margin-top:14px; }
.hero-chips span { font-size:12px;font-weight:700;padding:5px 12px;border-radius:99px;background:rgba(255,255,255,.75);border:1px solid #dfe6fb;color:#4f5fc0;animation:akrise .6s both;transition:transform .2s ease,background .2s ease; }
.hero-chips span:hover { transform:translateY(-3px);background:#fff; }
.hero-chips span:nth-child(1){animation-delay:.15s}.hero-chips span:nth-child(2){animation-delay:.22s}.hero-chips span:nth-child(3){animation-delay:.29s}.hero-chips span:nth-child(4){animation-delay:.36s}.hero-chips span:nth-child(5){animation-delay:.43s}.hero-chips span:nth-child(6){animation-delay:.5s}
.hero-visual { position:relative;z-index:1;flex:0 1 380px;min-width:0;display:flex;justify-content:flex-end; }
.hero-art { width:100%;max-width:380px;height:auto;overflow:visible; }
.art-blob { transform-origin:center;transform-box:fill-box;animation:akpulse 7s ease-in-out infinite; }
.art-card { transform-box:fill-box;transform-origin:center;animation:akbob 5s ease-in-out infinite; }
.art-card.c2 { animation-delay:-1.2s; } .art-card.c3 { animation-delay:-2.4s; } .art-card.c4 { animation-delay:-3.4s; }
.art-fly { transform-box:fill-box;transform-origin:center;animation:akbob 3.4s ease-in-out infinite; } .art-fly.f2 { animation-delay:-1.4s; }
.art-lid { transform-box:fill-box;transform-origin:0% 100%;animation:aklid 4s ease-in-out infinite; }
@keyframes aklid { 0%,100% { transform:rotate(0); } 50% { transform:rotate(-9deg); } }
.art-star { transform-box:fill-box;transform-origin:center;animation:aktwink 2.6s ease-in-out infinite; } .art-star.s2 { animation-delay:-.9s; } .art-star.s3 { animation-delay:-1.7s; }
.art-person { transform-box:fill-box;transform-origin:50% 100%;animation:akbob 6s ease-in-out infinite; }
.mini-stat { position:relative;overflow:hidden; }
.mini-stat::after { content:"";position:absolute;top:0;left:0;width:30%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.55),transparent);transform:translateX(-120%) skewX(-18deg);pointer-events:none; }
.mini-stat:hover::after { animation:aksheen .9s ease; }
.stat-row .mini-stat:hover b { color:var(--ak-blue)!important; }
.guide-grid { align-items:start; }
details.guide-card { display:block;padding:0;overflow:hidden; }
details.guide-card[open] { grid-column:1/-1;border-color:var(--ak-blue)!important;box-shadow:0 14px 34px rgba(79,109,245,.16); }
details.guide-card > summary { display:flex;gap:14px;align-items:center;padding:16px;cursor:pointer;list-style:none; }
details.guide-card > summary::-webkit-details-marker { display:none; }
.gc-chev { margin-left:auto;flex:none;width:9px;height:9px;border-right:2px solid var(--ak-muted);border-bottom:2px solid var(--ak-muted);transform:rotate(45deg);transition:transform .25s ease; }
details[open] > summary .gc-chev { transform:rotate(-135deg); }
details.guide-card:hover .gc-ico { transform:rotate(-8deg) scale(1.08); } .gc-ico { transition:transform .25s ease; }
.gc-detail { padding:2px 18px 18px 72px;animation:akrise .4s ease both; }
.gc-detail ol { margin:0 0 10px;padding-left:20px;color:var(--ak-text);line-height:1.55;font-size:13.5px; } .gc-detail li { margin:4px 0;animation:akrise .45s both; }
.gc-detail li:nth-child(2){animation-delay:.07s}.gc-detail li:nth-child(3){animation-delay:.14s}
.gc-tip { display:flex;gap:8px;align-items:flex-start;margin:8px 0 12px;padding:10px 12px;border-radius:12px;background:var(--ak-soft);color:var(--ak-text);font-size:13px; }
.gc-tipico { flex:none;color:#d99a06;display:inline-flex; }
.home-footer { margin:34px 0 8px;padding:32px 20px 24px;text-align:center;border-top:1px solid var(--ak-line);position:relative;animation:akrise .8s both; }
.hf-logo { display:block;margin:0 auto;width:64px;height:64px;border-radius:18px;box-shadow:0 10px 26px rgba(79,109,245,.28);animation:akpulse 4.5s ease-in-out infinite; }
.hf-name { margin-top:8px;font-size:22px;font-weight:850;letter-spacing:.14em;color:var(--ak-text); }
.hf-slogan { max-width:640px;margin:10px auto 0;font-size:17px;line-height:1.55;font-weight:600;font-style:italic;color:var(--ak-text);background:linear-gradient(90deg,#4f6df5,#8e6cf4);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent; }
.hf-sub { margin-top:10px;font-size:10px;letter-spacing:.12em;color:var(--ak-muted);text-transform:uppercase; }
@media (prefers-reduced-motion: reduce) { .rv,.hero *,.home-footer,.home-footer *,.mini-stat { animation:none!important; } }
@media (max-width:820px) { .hero { flex-direction:column;align-items:flex-start!important; } .hero-visual { width:100%;justify-content:center; } .gc-detail { padding-left:18px; } }

/* ===== menu : page active, encoche concave ===== */
@keyframes aknavin { from { opacity:0;transform:translateX(-14px); } to { opacity:1;transform:none; } }
html:root:root:root:root body #sidebar .navbtn.nav-active {
  position:relative!important;overflow:visible!important;z-index:2;
  width:calc(100% + 12px)!important;margin-right:-12px!important;
  border-radius:14px 0 0 14px!important;border-color:transparent!important;
  background:var(--ak-soft)!important;color:var(--ak-blue-dark)!important;-webkit-text-fill-color:var(--ak-blue-dark)!important;
  box-shadow:inset 3px 0 0 var(--ak-blue)!important;transform:none!important;
  animation:aknavin .32s cubic-bezier(.2,.8,.2,1) both;
}
html:root:root:root:root body #sidebar .navbtn.nav-active * { color:var(--ak-blue-dark)!important;-webkit-text-fill-color:var(--ak-blue-dark)!important; }
html:root:root:root:root body #sidebar .navbtn.nav-active::before { color:var(--ak-blue)!important;transform:scale(1.12);transition:transform .25s ease; }
html:root:root:root:root body #sidebar .navbtn.nav-active::after {
  content:"";position:absolute;right:0;top:-14px;bottom:-14px;width:14px;pointer-events:none;
  background:radial-gradient(circle at 0 0,transparent 13.5px,var(--ak-soft) 14.5px) top/14px 14px no-repeat,
             radial-gradient(circle at 0 100%,transparent 13.5px,var(--ak-soft) 14.5px) bottom/14px 14px no-repeat;
}
html:root:root:root:root body.akori-dark #sidebar .navbtn.nav-active { background:var(--ak-soft)!important; }

#sidebar #nav-mindmap::before { content:""!important;width:20px!important;height:20px!important;font-size:0!important;background:currentColor!important;-webkit-mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Ccircle%20cx='5'%20cy='12'%20r='2.6'/%3E%3Ccircle%20cx='19'%20cy='5'%20r='2.6'/%3E%3Ccircle%20cx='19'%20cy='19'%20r='2.6'/%3E%3Cpath%20d='M7.5%2011l9-5M7.5%2013l9%205'/%3E%3C/svg%3E") center/contain no-repeat!important;mask:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='black'%20stroke-width='1.9'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Ccircle%20cx='5'%20cy='12'%20r='2.6'/%3E%3Ccircle%20cx='19'%20cy='5'%20r='2.6'/%3E%3Ccircle%20cx='19'%20cy='19'%20r='2.6'/%3E%3Cpath%20d='M7.5%2011l9-5M7.5%2013l9%205'/%3E%3C/svg%3E") center/contain no-repeat!important; }

/* ===== menu replié : seule l'icône reste visible ===== */
html:root:root:root:root body.sidebar-collapsed:not(.sb-hover) #sidebar .navbtn.nav-active {
  width:100%!important;margin-right:0!important;overflow:hidden!important;border-radius:12px!important;box-shadow:none!important;animation:none!important;
}
html:root:root:root:root body.sidebar-collapsed:not(.sb-hover) #sidebar .navbtn.nav-active::after { display:none!important; }
@media (max-width:900px) {
  html:root:root:root:root body:not(.sb-hover) #sidebar .navbtn.nav-active { width:100%!important;margin-right:0!important;overflow:hidden!important;border-radius:12px!important;box-shadow:none!important; }
  html:root:root:root:root body:not(.sb-hover) #sidebar .navbtn.nav-active::after { display:none!important; }
}

/* ===== quiz : toucher = correction, glisser = suite ===== */
.quiz-card { touch-action:pan-y;transition:transform .2s ease,opacity .2s ease;animation:akrise .35s ease both; }
.quiz-card.swipe-out { transform:translateX(-60px);opacity:0; }
.quiz-options { display:flex;flex-direction:column; }
button.option.opt-click { width:100%;text-align:left;font:inherit;background:var(--ak-card);transition:transform .15s ease,border-color .15s ease,background .15s ease,box-shadow .15s ease; }
button.option.opt-click:hover { border-color:var(--ak-blue);background:var(--ak-soft);transform:translateX(4px);box-shadow:0 6px 16px rgba(79,109,245,.12); }
button.option.opt-click:active { transform:scale(.99); }
.option.correct,.option.wrong { animation:akrise .35s ease both; }
.quiz-tap { margin-top:10px;color:var(--ak-muted);font-size:12.5px;text-align:center; }
.quiz-swipe { display:flex;align-items:center;gap:10px;margin-top:14px;padding:10px 14px;border-radius:12px;background:var(--ak-soft);color:var(--ak-text);font-size:13px; }
.quiz-swipe .qs-arrow { font-size:20px;color:var(--ak-blue);animation:qsnudge 1.2s ease-in-out infinite; }
@keyframes qsnudge { 0%,100% { transform:translateX(0); } 50% { transform:translateX(-7px); } }
.quiz-swipe .ak-quiz-next { margin-left:auto; }

/* ===== mindmap ===== */
.mm-wrap { border:1px solid var(--ak-line);border-radius:18px;background:var(--ak-card);padding:14px 16px;animation:akrise .5s ease both; }
.mm-hint { color:var(--ak-muted);font-size:12.5px;margin-bottom:8px; }
.mm-scroll { overflow-x:auto; }
.mm-svg { width:100%;height:auto;display:block; }
.mm-svg text { font-family:inherit;font-size:13px;pointer-events:none; }
.mm-t-root { fill:#fff;font-weight:800;font-size:15px!important; } .mm-t-br { fill:#fff;font-weight:700; } .mm-t-leaf { fill:var(--ak-text);font-weight:600;font-size:12.5px!important; }
.mm-path,.mm-lp { stroke-dasharray:600;stroke-dashoffset:600;animation:mmdraw 1s ease forwards;animation-delay:calc(var(--i,0) * 120ms + 150ms); }
.mm-lp { animation-delay:.6s; }
@keyframes mmdraw { to { stroke-dashoffset:0; } }
.mm-root { animation:mmpop .6s cubic-bezier(.2,.9,.3,1.3) both; transform-box:fill-box;transform-origin:center; }
.mm-br { animation:mmpop .6s cubic-bezier(.2,.9,.3,1.3) both;animation-delay:calc(var(--i,0) * 120ms + 300ms);transform-box:fill-box;transform-origin:left center; }
@keyframes mmpop { from { opacity:0;transform:scale(.7); } to { opacity:1;transform:scale(1); } }
.mm-bnode { cursor:pointer; } .mm-bnode:hover rect:first-child { filter:brightness(1.08); }
.mm-leaf { transition:transform .2s ease; } .mm-leaf:hover { transform:translateX(3px); }
.mm-plus { display:none; } .mm-br.mm-closed .mm-plus { display:block; } .mm-br.mm-closed .mm-minus { display:none; }
.mm-br.mm-closed .mm-leaf,.mm-br.mm-closed .mm-lp { display:none; }

/* ===== à propos ===== */
.about-card { border:1px solid var(--ak-line);border-radius:18px;background:var(--ak-card);padding:20px 22px;margin:10px 0;animation:akrise .5s ease both; }
.about-card p { color:var(--ak-text);line-height:1.6;margin:10px 0 0; } .about-note { color:var(--ak-muted)!important;font-size:13px; }
.about-head { display:flex;align-items:center;gap:14px; } .about-head img { width:56px;height:56px;border-radius:16px;box-shadow:0 8px 20px rgba(79,109,245,.25); }
.about-head b,.about-editor b { display:block;font-size:19px;color:var(--ak-text); } .about-head span,.about-editor span { color:var(--ak-muted);font-size:12.5px; }
.about-chips { display:flex;flex-wrap:wrap;gap:8px;margin-top:12px; } .about-chips span { padding:4px 12px;border-radius:99px;background:var(--ak-soft);color:var(--ak-blue-dark);font-size:12.5px;font-weight:700; }
.about-editor { display:flex;gap:16px;align-items:flex-start; }
.about-av { flex:none;width:56px;height:56px;border-radius:50%;background:linear-gradient(135deg,#4f6df5,#8e6cf4);color:#fff;font-size:24px;font-weight:800;display:flex;align-items:center;justify-content:center; }

/* ===== page de présentation (avant connexion) ===== */
html:root:root:root:root body #login-view:has(#landing) { width:100%!important;max-width:none!important;padding:0!important;margin:0!important;gap:0!important; }
#landing { width:100%!important;max-width:none!important;padding:0!important;gap:0!important;background:transparent!important;border:0!important; }
#landing .block,#landing .html-container,#landing .prose { padding:0!important;max-width:none!important;background:transparent!important;border:0!important;box-shadow:none!important; }
#login-wrap { width:min(560px,100%)!important;margin:0 auto!important;gap:14px!important; }
#back-land { align-self:flex-start!important; }
#landing .ld { --lb:#4f6df5;--lp:#8e6cf4;color:#d9dbe6;overflow-x:clip;text-align:left; }
#landing .ld h1,#landing .ld h2,#landing .ld h3,#landing .ld p { margin:0; }
#landing .ld a { text-decoration:none; }
.lr { opacity:0;transform:translateY(28px);transition:opacity .8s ease,transform .8s cubic-bezier(.2,.8,.2,1);animation:ldshow .01s 3.2s forwards; }
.lr.in { opacity:1;transform:none; }
@keyframes ldshow { to { opacity:1;transform:none; } }
.ld-nav { position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:20px;padding:14px max(20px,calc((100% - 1120px)/2));background:rgba(10,10,12,.72);backdrop-filter:blur(14px);border-bottom:1px solid rgba(255,255,255,.07); }
.ld-brand { display:flex;align-items:center;gap:10px;margin-right:auto; } .ld-brand img { width:34px;height:34px;border-radius:10px; } .ld-brand b { font-size:19px;letter-spacing:.12em;color:#fff; }
.ld-links { display:flex;gap:22px; } .ld-links a { color:#9a9fb5;font-size:14px;font-weight:600;transition:color .2s; } .ld-links a:hover { color:#fff; }
.ld-btn { display:inline-flex;align-items:center;gap:8px;padding:11px 20px;border-radius:12px;font:inherit;font-weight:700;font-size:14.5px;cursor:pointer;border:1px solid transparent;transition:transform .2s ease,box-shadow .2s ease,background .2s ease; }
.ld-btn:hover { transform:translateY(-2px); }
.ld-btn-primary { background:linear-gradient(135deg,#4f6df5,#8e6cf4);color:#fff!important;box-shadow:0 10px 30px rgba(99,102,241,.4); } .ld-btn-primary:hover { box-shadow:0 14px 38px rgba(99,102,241,.55); } .ld-btn-primary span { transition:transform .2s; } .ld-btn-primary:hover span { transform:translateX(4px); }
.ld-btn-ghost { background:rgba(255,255,255,.06);border-color:rgba(255,255,255,.14);color:#e8e9f1!important; } .ld-btn-ghost:hover { background:rgba(255,255,255,.12); }
.ld-hero { position:relative;text-align:center;padding:84px 20px 40px;max-width:1120px;margin:0 auto; }
.ld-orb { position:absolute;border-radius:50%;filter:blur(70px);pointer-events:none;z-index:0; }
.ld-orb.o1 { width:420px;height:420px;left:-80px;top:-40px;background:rgba(79,109,245,.35);animation:akblob 11s ease-in-out infinite; }
.ld-orb.o2 { width:380px;height:380px;right:-60px;top:60px;background:rgba(232,111,176,.24);animation:akblob 13s ease-in-out infinite reverse; }
.ld-grid { position:absolute;inset:0;z-index:0;background-image:linear-gradient(rgba(255,255,255,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.04) 1px,transparent 1px);background-size:44px 44px;-webkit-mask:radial-gradient(ellipse at 50% 30%,#000 20%,transparent 70%);mask:radial-gradient(ellipse at 50% 30%,#000 20%,transparent 70%); }
.ld-hero > * { position:relative;z-index:1; } .ld-hero > .ld-orb,.ld-hero > .ld-grid { position:absolute; }
.ld-pill { display:inline-flex;align-items:center;gap:9px;padding:7px 16px;border-radius:99px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);font-size:13px;font-weight:600;color:#cfd3ea; }
.ld-dot { width:8px;height:8px;border-radius:50%;background:#4ade80;box-shadow:0 0 0 0 rgba(74,222,128,.7);animation:ldping 1.8s infinite; } @keyframes ldping { 70% { box-shadow:0 0 0 9px rgba(74,222,128,0); } 100% { box-shadow:0 0 0 0 rgba(74,222,128,0); } }
.ld-hero h1 { margin:22px auto 0!important;font-size:clamp(34px,6vw,68px);line-height:1.05;font-weight:850;letter-spacing:-.02em;color:#fff!important;max-width:900px; }
.ld-grad { background:linear-gradient(90deg,#6c8bff,#b58bff,#ff7ac3,#6c8bff);background-size:200% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent;animation:akflow 6s linear infinite; }
.ld-sub { max-width:680px;margin:22px auto 0!important;font-size:clamp(15px,2vw,18.5px);line-height:1.6;color:#a6abc2!important; }
.ld-cta { display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:30px; }
.ld-trust { display:flex;flex-wrap:wrap;justify-content:center;gap:10px 22px;margin-top:26px;color:#8a8fa8;font-size:13px; } .ld-trust span::before { content:"✓";color:#4ade80;margin-right:7px;font-weight:800; }
.ld-mock { margin:46px auto 0;max-width:900px;perspective:1200px; }
.ld-win { border-radius:18px;background:#14151c;border:1px solid rgba(255,255,255,.1);box-shadow:0 40px 90px rgba(0,0,0,.6),0 0 0 1px rgba(255,255,255,.03),0 0 80px rgba(99,102,241,.2);transform:rotateX(6deg);transform-origin:50% 100%;animation:ldfloat 7s ease-in-out infinite;overflow:hidden; }
@keyframes ldfloat { 0%,100% { transform:rotateX(6deg) translateY(0); } 50% { transform:rotateX(4deg) translateY(-10px); } }
.ld-bar { display:flex;align-items:center;gap:7px;padding:12px 16px;border-bottom:1px solid rgba(255,255,255,.07); } .ld-bar i { width:11px;height:11px;border-radius:50%;background:#3a3c4a; } .ld-bar i:first-child { background:#ff5f57; } .ld-bar i:nth-child(2) { background:#febc2e; } .ld-bar i:nth-child(3) { background:#28c840; } .ld-bar span { margin-left:10px;color:#7d8199;font-size:12.5px; }
.ld-body { display:grid;grid-template-columns:54px 1.5fr 1fr;gap:16px;padding:16px;text-align:left; }
.ld-side { display:flex;flex-direction:column;gap:12px;align-items:center;padding-top:6px; } .ld-side b { width:26px;height:26px;border-radius:8px;background:#23252f; } .ld-side b.on { background:linear-gradient(135deg,#4f6df5,#8e6cf4); }
.ld-chat { display:flex;flex-direction:column;gap:10px; } .ld-msg { padding:11px 14px;border-radius:14px;font-size:13.5px;line-height:1.5;max-width:92%; }
.ld-msg.me { align-self:flex-end;background:linear-gradient(135deg,#4f6df5,#7a62f0);color:#fff; } .ld-msg.ai { background:#1d1f29;border:1px solid rgba(255,255,255,.08);color:#d9dbe6; }
.ld-src { display:flex;gap:6px;margin-top:8px; } .ld-src span { font-size:11.5px;padding:3px 9px;border-radius:99px;background:rgba(99,102,241,.2);color:#aab6ff; }
.ld-chips { display:flex;gap:8px; } .ld-chips span { font-size:11.5px;padding:4px 11px;border-radius:99px;border:1px solid rgba(255,255,255,.12);color:#9a9fb5; }
.ld-cards { display:flex;flex-direction:column;gap:10px; } .ld-mini { padding:12px 14px;border-radius:14px;background:#1d1f29;border:1px solid rgba(255,255,255,.08); } .ld-mini em { display:block;font-style:normal;font-size:11.5px;color:#8a8fa8; } .ld-mini b { display:block;font-size:20px;color:#fff;margin:2px 0 7px; }
.ld-bar2 { height:6px;border-radius:99px;background:#2a2c38;overflow:hidden; } .ld-bar2 i { display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#4f6df5,#ff7ac3);animation:akgrow 1.4s ease both; }
.ld-roll { display:flex;gap:6px; } .ld-roll i { width:24px;height:30px;border-radius:6px;background:#2a2c38;animation:ldroll 1.6s ease-in-out infinite; } .ld-roll i:nth-child(2) { animation-delay:.2s;background:#4f6df5; } .ld-roll i:nth-child(3) { animation-delay:.4s; }
@keyframes ldroll { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-6px); } }
.ld-strip { margin:50px 0 0;border-block:1px solid rgba(255,255,255,.07);padding:16px 0;overflow:hidden;background:rgba(255,255,255,.02); }
.ld-marquee { display:flex;width:max-content;animation:ldmarq 38s linear infinite; } .ld-marquee > div { display:flex;gap:44px;padding-right:44px; } .ld-marquee span { color:#8a8fa8;font-weight:700;font-size:14px;white-space:nowrap; } .ld-marquee span::before { content:"✦";color:#6c8bff;margin-right:14px; }
@keyframes ldmarq { to { transform:translateX(-50%); } }
.ld-sec { max-width:1120px;margin:0 auto;padding:96px 20px 0; }
.ld-head { text-align:center;max-width:720px;margin:0 auto 44px; } .ld-kicker { display:inline-block;font-size:12px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#8ea2ff;margin-bottom:12px; }
.ld-head h2,.ld-text h2 { font-size:clamp(26px,4vw,42px);line-height:1.15;font-weight:850;letter-spacing:-.01em;color:#fff!important; } .ld-head p { margin-top:14px!important;color:#9a9fb5!important;font-size:16.5px;line-height:1.6; }
.ld-three { display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px; }
.ld-card,.ld-b { padding:26px;border-radius:20px;background:linear-gradient(160deg,rgba(255,255,255,.07),rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.09);transition:transform .25s ease,border-color .25s ease,box-shadow .25s ease; }
.ld-card:hover,.ld-b:hover { transform:translateY(-6px);border-color:rgba(142,162,255,.5);box-shadow:0 20px 50px rgba(79,109,245,.18); }
.ld-card h3,.ld-b h3 { font-size:18px;font-weight:800;color:#fff!important;margin:14px 0 8px!important; } .ld-card p,.ld-b p { color:#9a9fb5!important;line-height:1.6;font-size:14.5px; }
.ld-ico { width:46px;height:46px;border-radius:14px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,rgba(79,109,245,.3),rgba(142,108,244,.3));color:#b4c0ff;border:1px solid rgba(142,162,255,.3); } .ld-ico svg { width:23px;height:23px; }
.ld-b:hover .ld-ico { transform:rotate(-8deg) scale(1.1); } .ld-ico { transition:transform .25s ease; }
.ld-bento { display:grid;grid-template-columns:repeat(4,1fr);gap:16px; } .ld-b.big { grid-column:span 2; }
.ld-steps { display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px;counter-reset:s; }
.ld-step { position:relative;padding:28px 26px;border-radius:20px;border:1px solid rgba(255,255,255,.09);background:rgba(255,255,255,.03); } .ld-step span { width:42px;height:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:850;color:#fff;background:linear-gradient(135deg,#4f6df5,#ff7ac3);box-shadow:0 8px 24px rgba(99,102,241,.4); }
.ld-step h3 { margin:16px 0 8px!important;font-size:20px;font-weight:800;color:#fff!important; } .ld-step p { color:#9a9fb5!important;line-height:1.6;font-size:14.5px; }
.ld-split { display:grid;grid-template-columns:1fr 1fr;gap:48px;align-items:center; } .ld-split.rev .ld-demo { order:-1; }
.ld-text p { margin-top:16px!important;color:#9a9fb5!important;line-height:1.7;font-size:16px; } .ld-text ul { margin:18px 0 0;padding:0;list-style:none;display:grid;gap:10px; } .ld-text li { color:#cfd3ea;font-size:15px;padding-left:28px;position:relative; } .ld-text li::before { content:"✓";position:absolute;left:0;color:#4ade80;font-weight:800; }
.ld-demo { padding:20px;border-radius:22px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.09);display:grid;gap:10px; }
.ld-day { padding:13px 16px;border-radius:14px;background:#1a1c26;border:1px solid rgba(255,255,255,.07);display:grid;grid-template-columns:1fr auto;gap:2px 10px; } .ld-day b { color:#fff;font-size:14px; } .ld-day span { color:#8a8fa8;font-size:12.5px; } .ld-day p { grid-column:1/-1;color:#b9bdd2!important;font-size:13.5px; }
.ld-day.today { border-color:#6c8bff;box-shadow:0 0 0 3px rgba(108,139,255,.16); } .ld-day.rest { opacity:.6; } .ld-pg { grid-column:1/-1;display:flex;gap:6px;flex-wrap:wrap;margin-top:6px; } .ld-pg i { font-style:normal;font-size:12px;padding:2px 10px;border-radius:8px;background:rgba(245,158,11,.16);color:#fbbf24;font-weight:600; }
.ld-mm { width:100%;height:auto; } .ld-mm path { fill:none;stroke-width:2.5;stroke-dasharray:300;stroke-dashoffset:300;} .lr.in .ld-mm path,.ld-mm.in path { animation:mmdraw 1.4s ease forwards; } .ld-mm text { fill:#fff;font-size:12.5px;font-weight:700;font-family:inherit; } .ld-mm .leaf rect { fill:rgba(255,255,255,.08);stroke:rgba(255,255,255,.25); } .ld-mm .leaf text { fill:#d9dbe6;font-weight:600;font-size:11.5px; }
.ld-flow { display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:10px;margin:0 auto 28px; } .ld-flow span { padding:10px 18px;border-radius:12px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);color:#e8e9f1;font-weight:700;font-size:13.5px; } .ld-flow span.end { background:linear-gradient(135deg,#4f6df5,#8e6cf4);border-color:transparent; } .ld-flow i { width:26px;height:2px;background:linear-gradient(90deg,#4f6df5,#ff7ac3);position:relative; } .ld-flow i::after { content:"";position:absolute;right:-1px;top:-3px;border:4px solid transparent;border-left:7px solid #ff7ac3; }
.ld-faq { max-width:760px;margin:0 auto;display:grid;gap:10px; } .ld-faq details { border-radius:16px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);padding:0 20px;transition:border-color .2s; } .ld-faq details[open] { border-color:rgba(142,162,255,.5); }
.ld-faq summary { cursor:pointer;list-style:none;padding:18px 0;font-weight:700;color:#fff;display:flex;justify-content:space-between;gap:12px; } .ld-faq summary::-webkit-details-marker { display:none; } .ld-faq summary::after { content:"+";color:#8ea2ff;font-size:22px;line-height:1;transition:transform .25s; } .ld-faq details[open] summary::after { transform:rotate(45deg); }
.ld-faq p { padding:0 0 18px;color:#a6abc2!important;line-height:1.65;animation:akrise .35s both; }
.ld-final { text-align:center;margin:110px auto 0;max-width:760px;padding:56px 24px;border-radius:28px;background:radial-gradient(500px circle at 50% 0,rgba(99,102,241,.35),transparent 70%),rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.12); }
.ld-final img { width:64px;height:64px;border-radius:18px;margin-bottom:14px;animation:akpulse 4.5s ease-in-out infinite; } .ld-final h2 { font-size:clamp(26px,4vw,38px);font-weight:850;color:#fff!important; } .ld-final p { margin:12px 0 26px!important;color:#a6abc2!important; }
.ld-foot { text-align:center;padding:60px 20px 50px;color:#8a8fa8;font-size:13px; } .ld-slogan { max-width:640px;margin:0 auto 20px!important;font-size:18px;font-style:italic;font-weight:600;line-height:1.55;background:linear-gradient(90deg,#8ea2ff,#ff9ad0);-webkit-background-clip:text;background-clip:text;color:transparent!important;-webkit-text-fill-color:transparent; } .ld-foot b { color:#fff;letter-spacing:.1em; } .ld-foot small { display:block;margin-top:8px;color:#6b6f86; }

html:root:root:root:root body #landing .ld-hero h1,html:root:root:root:root body #landing .ld-head h2,html:root:root:root:root body #landing .ld-text h2,html:root:root:root:root body #landing .ld-card h3,html:root:root:root:root body #landing .ld-b h3,html:root:root:root:root body #landing .ld-step h3,html:root:root:root:root body #landing .ld-final h2 { color:#fff!important; }
html:root:root:root:root body #landing .ld-head p,html:root:root:root:root body #landing .ld-card p,html:root:root:root:root body #landing .ld-b p,html:root:root:root:root body #landing .ld-step p,html:root:root:root:root body #landing .ld-text p { color:#9a9fb5!important; }
html:root:root:root:root body #landing .ld-sub,html:root:root:root:root body #landing .ld-faq p,html:root:root:root:root body #landing .ld-final p { color:#a6abc2!important; }
html:root:root:root:root body #landing .ld-day p { color:#b9bdd2!important; }
html:root:root:root:root body #landing .ld-slogan { color:transparent!important; }
html:root:root:root:root body #landing .ld-trust span,html:root:root:root:root body #landing .ld-marquee span { color:#8a8fa8!important; }
html:root:root:root:root body #landing .ld-day span,html:root:root:root:root body #landing .ld-mini em,html:root:root:root:root body #landing .ld-bar span { color:#8a8fa8!important; }
html:root:root:root:root body #landing .ld-day b,html:root:root:root:root body #landing .ld-mini b { color:#fff!important; }
html:root:root:root:root body #landing .ld-pill { color:#cfd3ea!important; }
html:root:root:root:root body #landing .ld-foot,html:root:root:root:root body #landing .ld-foot div { color:#8a8fa8!important; }
@media (max-width:900px) { .ld-links { display:none; } .ld-bento { grid-template-columns:1fr 1fr; } .ld-split { grid-template-columns:1fr;gap:28px; } .ld-split.rev .ld-demo { order:0; } .ld-body { grid-template-columns:1fr; } .ld-side { display:none; } }
@media (max-width:560px) { .ld-bento { grid-template-columns:1fr; } .ld-b.big { grid-column:auto; } }
@media (prefers-reduced-motion: reduce) { .lr { transition:none;animation:none;opacity:1;transform:none; } .ld *,.ld-win { animation:none!important; } }


/* ===== téléphone : menu en tiroir, pages pleine largeur, gestes fluides (classe m-mode posée par le script) ===== */
#m-burger { display:none;position:fixed;top:max(10px,env(safe-area-inset-top));left:12px;z-index:250;width:44px;height:44px;border-radius:14px;
  border:1px solid var(--ak-line);background:var(--ak-card);color:var(--ak-text);box-shadow:0 6px 18px rgba(42,55,90,.16);align-items:center;justify-content:center;padding:0;cursor:pointer; }
#m-burger svg { width:22px;height:22px; }
#m-backdrop { position:fixed;inset:0;background:rgba(10,14,28,.5);opacity:0;pointer-events:none;transition:opacity .25s ease;z-index:280; }
html:root:root:root:root body.m-mode.in-app #m-burger { display:flex; }
html:root:root:root:root body.m-mode.m-nav-open #m-backdrop { opacity:1;pointer-events:auto; }
html:root:root:root:root body.m-mode { --sb-w:min(300px,84vw); --main-ml:0px; }
html:root:root:root:root body.m-mode #sidebar {
  left:0!important;top:0!important;height:100dvh!important;max-height:100dvh!important;border-radius:0 22px 22px 0!important;
  width:min(300px,84vw)!important;min-width:min(300px,84vw)!important;max-width:min(300px,84vw)!important;
  transform:translateX(-106%);transition:transform .3s var(--sb-ease)!important;z-index:300;padding-top:max(16px,env(safe-area-inset-top))!important;
  padding-bottom:max(12px,env(safe-area-inset-bottom))!important;will-change:transform;
}
html:root:root:root:root body.m-mode.m-nav-open #sidebar { transform:none;box-shadow:0 0 60px rgba(15,23,42,.45)!important; }
html:root:root:root:root body.m-mode #sidebar #sidebar-toggle { display:none!important; }
html:root:root:root:root body.m-mode #sidebar .navbtn { min-height:46px!important; }
html:root:root:root:root body.m-mode #main-column { margin-left:0!important;padding-top:54px; }
html:root:root:root:root body.m-mode #topbar { margin-bottom:4px!important; }
html:root:root:root:root body.m-mode .gradio-container .main { padding-left:0!important;padding-right:0!important; }
html:root:root:root:root body.m-mode #main-tabs .html-container { padding-left:0!important;padding-right:0!important; }
html:root:root:root:root body.m-mode #topbar > * { flex:1 1 100%!important;min-width:100%!important; }
html:root:root:root:root body.m-mode .gradio-container { padding-left:10px!important;padding-right:10px!important; }
html:root:root:root:root body.m-mode #main-tabs .tabitem { min-height:0!important; }
html:root:root:root:root body.m-mode .gradio-container button, html:root:root:root:root body.m-mode .gradio-container [role="tab"] { min-height:44px; }
html:root:root:root:root body.m-mode .gradio-container input, html:root:root:root:root body.m-mode .gradio-container textarea, html:root:root:root:root body.m-mode .gradio-container select { font-size:16px!important; }
html:root:root:root:root body.m-mode .stat-row { grid-template-columns:1fr 1fr!important; }
html:root:root:root:root body.m-mode .guide-grid, html:root:root:root:root body.m-mode .folder-grid, html:root:root:root:root body.m-mode .global-metrics { grid-template-columns:1fr!important; }
html:root:root:root:root body.m-mode .hero { padding:22px 18px!important; }
html:root:root:root:root body.m-mode .gc-detail { padding-left:18px!important; }
html:root:root:root:root body.m-mode #assistant-chatbot { height:calc(100dvh - 330px)!important;min-height:300px!important; }
html:root:root:root:root body.m-mode .quiz-card, html:root:root:root:root body.m-mode .flashcard { padding:18px!important; }
html:root:root:root:root body.m-mode .quiz-card h2 { font-size:18px!important; }
html:root:root:root:root body.m-mode button.option.opt-click { padding:14px 12px;min-height:52px; }
html:root:root:root:root body.m-mode .plan-task { flex-wrap:wrap; }
html:root:root:root:root body.m-mode .plan-head { flex-direction:column;align-items:flex-start; }
html:root:root:root:root body.m-mode .jr-gap { grid-template-columns:1fr 44px; } html:root:root:root:root body.m-mode .jr-gap .act-bar { grid-column:1/-1;order:3; }
html:root:root:root:root body.m-mode #sidebar, html:root:root:root:root body.m-mode #sidebar * { transition-duration:.2s; }
/* page de présentation sur téléphone : pleine largeur, effets lourds coupés */
html:root:root:root:root body.m-mode:has(#landing) .gradio-container { padding:0!important;max-width:none!important; }
html:root:root:root:root body.m-mode .ld-nav { padding:10px 14px; } html:root:root:root:root body.m-mode .ld-nav .ld-btn { padding:9px 14px;white-space:nowrap;font-size:13.5px; }
html:root:root:root:root body.m-mode .ld-hero { padding:48px 16px 20px; } html:root:root:root:root body.m-mode .ld-sec { padding:64px 16px 0; }
html:root:root:root:root body.m-mode .ld-cta .ld-btn { width:100%;justify-content:center; }
html:root:root:root:root body.m-mode .ld-orb { animation:none;filter:blur(48px);opacity:.7; }
html:root:root:root:root body.m-mode .ld-win { animation:none;transform:none; }
html:root:root:root:root body.m-mode .ld-nav { backdrop-filter:none;background:rgba(10,10,12,.94); }
html:root:root:root:root body.m-mode .hero-glow, html:root:root:root:root body.m-mode .art-blob { animation:none; }
html:root:root:root:root body.m-mode .ld-marquee { animation-duration:55s; }


/* ===== Écran unique : en-tête (page + cours actif) séparé du corps (page active), sans défilement de la page ===== */
html:root:has(body.in-app) { overflow:hidden!important; }
html:root:root:root:root body.in-app { overflow:hidden!important; }
html:root:root:root:root body.in-app #main-column { height:calc(100dvh - var(--mc-top, 14px) - 12px)!important;display:flex!important;flex-direction:column!important;
  flex-wrap:nowrap!important;gap:10px!important;min-height:0!important;overflow:hidden!important; }
html:root:root:root:root body.in-app #topbar { flex:none!important;margin:0!important;padding:8px 10px 8px 16px!important;min-height:62px;flex-wrap:nowrap!important;
  align-items:center!important;gap:12px!important;border-radius:16px!important; }
html:root:root:root:root body.in-app #topbar > #topbar-title { flex:1 1 auto!important;min-width:0!important; }
html:root:root:root:root body.in-app #topbar > #doc-select { flex:0 0 min(340px,46%)!important;min-width:0!important; }
.pt-wrap { display:flex;align-items:center;gap:12px;min-width:0; }
.pt-ico { flex:none;width:6px;height:34px;border-radius:6px;background:linear-gradient(180deg,#4f6df5,#8e6cf4); }
#ak-page-title { font-size:19px;font-weight:800;color:var(--ak-text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
.pt-sub { font-size:11.5px;color:var(--ak-muted);white-space:nowrap; }
html:root:root:root:root body.in-app #main-tabs { flex:1 1 auto!important;min-height:0!important;display:flex!important;flex-direction:column!important;
  background:var(--ak-card)!important;border:1px solid var(--ak-line)!important;border-radius:18px!important;overflow:hidden!important;margin:0!important; }
html:root:root:root:root body.in-app #main-tabs > .tabitem { flex:1 1 auto!important;min-height:0!important;overflow-y:auto!important;overflow-x:hidden!important;
  padding:14px 16px!important;scrollbar-width:thin; }
html:root:root:root:root body.in-app #main-tabs .tabitem { min-height:0!important; }
html:root:root:root:root body.in-app #main-tabs .tabitem > .column, html:root:root:root:root body.in-app #main-tabs .tabitem > div > .column { gap:10px!important; }
/* colonnes et volets */
.ak-cols { gap:14px!important;align-items:stretch!important; }
html:root:root:root:root body.in-app .ak-pane { background:var(--ak-bg)!important;border:1px solid var(--ak-line)!important;border-radius:16px!important;padding:12px 14px!important;gap:10px!important;min-width:0!important; }
.pane-title { font-size:13px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:var(--ak-blue); }
.ak-toolbar { gap:10px!important;align-items:center!important;flex-wrap:wrap!important; }
.ak-toolbar > * { flex:0 1 auto!important;min-width:0!important; }
.tb-hint { color:var(--ak-muted);font-size:13px; }
#summary-out { height:calc(var(--body-h, 600px) - 96px);overflow-y:auto;padding:14px 18px!important;border:1px solid var(--ak-line);border-radius:16px;background:var(--ak-bg); }
html:root:root:root:root body.in-app #assistant-chatbot { height:calc(var(--body-h, 600px) - 100px)!important;min-height:260px!important; }
html:root:root:root:root body.in-app #assistant-chatbot > .wrapper, html:root:root:root:root body.in-app #assistant-chatbot .bubble-wrap { height:100%!important;max-height:none!important; }
#assistant-bar { gap:10px!important;align-items:center!important;flex-wrap:nowrap!important; }
#assistant-bar #chat-scope { flex:none!important; }
#chat-scope .wrap { gap:4px!important; }
#chat-scope label { padding:6px 10px!important;font-size:12.5px!important; }
.mm-wrap { max-height:calc(var(--body-h, 600px) - 90px);display:flex;flex-direction:column; } .mm-scroll { overflow:auto;flex:1;min-height:0; }
/* accueil tenant dans l'écran */
.home-fit { display:flex;flex-direction:column;gap:12px; }
html:root:root:root:root body .hero.hero-compact { min-height:0!important;padding:16px 22px!important;margin:0!important; }
.hero-compact h1 { font-size:clamp(22px,2.4vw,30px)!important;margin:6px 0 4px!important; }
.hero-compact .home-slogan { margin:4px 0 0!important;font-size:13.5px; }
.hero-compact .hero-visual { flex:0 1 240px!important; } .hero-compact .hero-art { max-width:240px;max-height:150px; }
.home-fit .stat-row { margin:0!important; }
.home-fit .mini-stat { padding:10px 14px!important; } .home-fit .mini-stat b { font-size:20px; }
.home-fit .guide-box { margin:0!important;padding:14px 16px!important; }
.home-fit .guide-box h3 { margin:0 0 10px!important;font-size:15px!important; } .home-fit .guide-box h3 small { font-weight:500;color:var(--ak-muted);font-size:12.5px;margin-left:8px; }
html:root:root:root:root body .home-fit .guide-grid { grid-template-columns:repeat(5,minmax(0,1fr))!important;gap:10px!important; }
.home-fit details.guide-card > summary { padding:10px 12px;gap:10px;align-items:flex-start; }
.home-fit .gc-ico { width:34px;height:34px;border-radius:11px; } .home-fit .gc-body b { font-size:13.5px; }
.home-fit .gc-body > span { font-size:12px!important;line-height:1.35!important;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden; }
.home-fit .gc-chev { display:none; }
/* détail du guide : fenêtre au centre de l'écran (la page ne s'allonge pas) */
.home-foot { display:flex;align-items:center;justify-content:center;gap:10px;color:var(--ak-muted);font-size:12.5px;font-style:italic;text-align:center;padding:2px 0; }
.home-foot img { width:22px;height:22px;border-radius:7px; }
@media (max-width:1100px) { html:root:root:root:root body .home-fit .guide-grid { grid-template-columns:repeat(3,minmax(0,1fr))!important; } }
html:root:root:root:root body.m-mode .home-fit .guide-grid { grid-template-columns:repeat(2,minmax(0,1fr))!important; }
html:root:root:root:root body.m-mode .hero-compact .hero-visual { display:none!important; }
html:root:root:root:root body.m-mode.in-app #topbar { flex-wrap:wrap!important; } html:root:root:root:root body.m-mode.in-app #topbar > #doc-select { flex:1 1 100%!important; }
html:root:root:root:root body.m-mode .ak-pane { padding:10px!important; }

/* ===== Connexion façon Claude (sombre et chaleureux) + comptes récents façon Messenger ===== */
html:root:root:root:root body.on-login, html:root:root:root:root body.on-login .gradio-container { background:#1f1e1d!important; }
#login-wrap { width:min(440px,100%)!important;margin:0 auto!important;gap:16px!important; }
#auth-top { align-items:center!important;justify-content:space-between!important;flex-wrap:nowrap!important;background:transparent!important;border:0!important;margin-bottom:6px; }
#auth-top > * { flex:0 0 auto!important;width:auto!important; }
.auth-logo { display:flex;align-items:center;gap:10px; } .auth-logo img { width:34px;height:34px;border-radius:10px; }
.auth-logo span { font-size:20px;font-weight:800;letter-spacing:.12em;color:#faf9f5!important; }
.auth-hero { text-align:center;margin:18px 0 6px; }
.auth-hero h1 { font-family:Georgia,'Iowan Old Style','Times New Roman',serif;font-weight:400;font-size:clamp(34px,6vw,46px);line-height:1.12;letter-spacing:-.01em;color:#faf9f5!important;margin:0 0 12px!important; }
.auth-hero p { color:#a6a39b!important;font-size:15.5px;margin:0!important; }
html:root:root:root:root body .auth-card { background:#262624!important;border:1px solid #3d3d3a!important;border-radius:24px!important;padding:22px!important;gap:12px!important;box-shadow:0 20px 60px rgba(0,0,0,.35)!important; }
.auth-card-title { font-size:17px;font-weight:700;color:#faf9f5;text-align:center;margin-bottom:2px; }
html:root:root:root:root body .auth-card input[type="text"], html:root:root:root:root body .auth-card input[type="password"], html:root:root:root:root body .auth-card textarea { background:#30302e!important;border:1px solid #4a4945!important;border-radius:12px!important;color:#faf9f5!important;-webkit-text-fill-color:#faf9f5!important;min-height:48px!important;font-size:15px!important; }
html:root:root:root:root body .auth-card label > span, html:root:root:root:root body .auth-card [data-testid="block-info"] { color:#c2c0b6!important;font-size:13px!important;font-weight:600!important; }
html:root:root:root:root body #li-btn, html:root:root:root:root body #su-btn { min-height:48px!important;border-radius:12px!important;background:#faf9f5!important;color:#1f1e1d!important;-webkit-text-fill-color:#1f1e1d!important;font-size:15px!important; }
html:root:root:root:root body #li-btn:hover, html:root:root:root:root body #su-btn:hover { background:#e8e6dc!important; }
.auth-or { display:flex;align-items:center;gap:12px;color:#8a877f;font-size:12px;letter-spacing:.08em;margin:2px 0; }
.auth-or::before, .auth-or::after { content:"";flex:1;height:1px;background:#3d3d3a; }
html:root:root:root:root body .auth-alt, html:root:root:root:root body .auth-alt.gr-button { min-height:46px!important;border-radius:12px!important;background:transparent!important;border:1px solid #4a4945!important;color:#faf9f5!important;-webkit-text-fill-color:#faf9f5!important;font-weight:600!important; }
html:root:root:root:root body .auth-alt:hover { background:#30302e!important; }
.auth-legal { text-align:center;color:#8a877f;font-size:12px;line-height:1.5;margin-top:4px; }
#ak-accounts:empty { display:none; }
.acc-title { font-size:13px;color:#c2c0b6;font-weight:600;margin-bottom:8px; }
.acc-row { display:flex;gap:10px;overflow-x:auto;padding-bottom:4px; }
.acc { position:relative;flex:0 0 96px;display:flex;flex-direction:column;align-items:center;gap:6px;padding:10px 6px;border-radius:16px;border:1px solid #3d3d3a;background:#30302e;cursor:pointer;transition:border-color .15s,transform .15s; }
.acc:hover { border-color:#8a877f;transform:translateY(-2px); } .acc.on { border-color:#8ea2ff;box-shadow:0 0 0 3px rgba(142,162,255,.2); }
.acc-av { width:54px;height:54px;border-radius:50%;object-fit:cover;display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:800;color:#fff;background:linear-gradient(135deg,#4f6df5,#8e6cf4); }
.acc-name { font-size:12.5px;color:#faf9f5;font-weight:600;max-width:86px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
.acc-x { position:absolute;top:4px;right:6px;width:20px;height:20px;border-radius:50%;border:0;background:transparent;color:#8a877f;cursor:pointer;font-size:14px;line-height:1; }
.acc-x:hover { color:#faf9f5;background:#3d3d3a; }
.acc-add .acc-av { background:#3d3d3a;color:#c2c0b6;font-size:26px;font-weight:400; }
.acc-pick { display:none;text-align:center;color:#faf9f5;font-size:15px;margin:2px 0; } .acc-pick b { font-weight:700; }
.picked .acc-pick { display:block; } .picked #li-user { display:none!important; }
.ub-mail { display:none!important; }


/* ajustements : accueil sur une seule hauteur d'écran, quiz compact, connexion compacte */
html:root:root:root:root body .home-fit .guide-grid { grid-template-columns:repeat(6,minmax(0,1fr))!important; }
@media (max-width:1250px) { html:root:root:root:root body .home-fit .guide-grid { grid-template-columns:repeat(4,minmax(0,1fr))!important; } }
.hero-compact .hero-art { max-height:118px!important; }
.home-fit .hero-compact .eyebrow { font-size:9.5px; }
html:root:root:root:root body .quiz-card { padding:16px 20px!important; }
html:root:root:root:root body .quiz-card h2 { margin:8px 0 10px!important;font-size:18px!important; }
html:root:root:root:root body .quiz-card .option { padding:9px 12px!important;margin:6px 0!important; }
html:root:root:root:root body .quiz-card .quiz-explanation { margin-top:10px!important;padding:10px 12px!important; }
html:root:root:root:root body .quiz-card .quiz-swipe { margin-top:10px!important;padding:8px 12px!important; }
html:root:root:root:root body .quiz-card .quiz-verdict { margin-bottom:6px!important; }
html:root:root:root:root body.on-login.on-login, html:root:root:root:root body.on-login.on-login .gradio-container { background:#1f1e1d!important; }
html:root:root:root:root body.on-login #login-view { padding-top:14px!important;padding-bottom:20px!important;gap:10px!important; }
.auth-hero { margin:4px 0 2px!important; } .auth-hero h1 { font-size:clamp(30px,5vw,40px)!important;margin:0 0 8px!important; }
html:root:root:root:root body .auth-card { padding:18px 20px!important;gap:10px!important; }
html:root:root:root:root body .auth-card input[type="text"], html:root:root:root:root body .auth-card input[type="password"] { min-height:44px!important;padding:10px 14px!important; }
html:root:root:root:root body #li-btn, html:root:root:root:root body #su-btn { min-height:46px!important;margin-top:2px!important; }
html:root:root:root:root body .acc-x, html:root:root:root:root body .acc-x:hover { background:transparent!important;border:0!important;box-shadow:none!important;padding:0!important;min-height:0!important; }
html:root:root:root:root body .acc-x:hover { background:#3d3d3a!important; }


html:root:root:root:root body.in-app #main-tabs > .tab-wrapper, html:root:root:root:root body.in-app #main-tabs > .tab-nav, html:root:root:root:root body.in-app #main-tabs > [role="tablist"] { display:none!important; }
html:root:root:root:root body.in-app #topbar { min-height:0!important;padding:6px 10px 6px 14px!important; }
html:root:root:root:root body #doc-select { padding:0!important;background:transparent!important;border:0!important;box-shadow:none!important; }
html:root:root:root:root body #doc-select label, html:root:root:root:root body #doc-select .container { display:flex!important;flex-direction:row!important;align-items:center!important;gap:10px!important;margin:0!important; }
html:root:root:root:root body #doc-select [data-testid="block-info"] { margin:0!important;white-space:nowrap!important;flex:none!important; }
html:root:root:root:root body #doc-select .wrap { flex:1 1 auto!important;min-width:0!important; }
.home-fit details.guide-card > summary { display:grid!important;grid-template-columns:30px 1fr;gap:4px 8px!important;align-items:center!important;padding:9px 10px!important; }
.home-fit .gc-ico { width:30px!important;height:30px!important;border-radius:9px!important; } .home-fit .gc-ico .ak-ico { width:17px;height:17px; }
.home-fit .gc-body { display:contents!important; }
.home-fit .gc-body b { font-size:13px!important;line-height:1.2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
.home-fit .gc-body > span { grid-column:1/-1;-webkit-line-clamp:1!important;font-size:11.5px!important; }
html:root:root:root:root body.in-app #summary-out { height:calc(var(--body-h, 600px) - 140px)!important; }
html:root:root:root:root body.in-app #assistant-chatbot { height:calc(var(--body-h, 600px) - 120px)!important; }
html:root:root:root:root body.in-app .mm-wrap { max-height:calc(var(--body-h, 600px) - 110px)!important; }
html:root:root:root:root body.in-app #main-tabs iframe { height:min(450px, calc(var(--body-h, 600px) - 170px))!important; }


html:root:root:root:root body.in-app #assistant-chatbot { height:calc(var(--body-h, 600px) - 175px)!important; }


html:root:root:root:root body .ak-pane > * { flex-shrink:0!important; }
.home-fit .gc-body > span { display:none!important; }
.home-fit .gc-body b { white-space:normal!important;overflow:visible!important; }

#scope-row { flex-wrap:nowrap!important;gap:8px!important; } #scope-row > * { flex:0 0 auto!important;width:auto!important; }
html:root:root:root:root body .ak-pane .global-metrics { grid-template-columns:1fr 1fr!important; }

html:root:root:root:root body.m-mode .ak-cols { flex-direction:column!important;flex-wrap:nowrap!important; }
html:root:root:root:root body.m-mode .ak-cols > .ak-pane { min-width:0!important;width:100%!important;flex:1 1 auto!important; }
html:root:root:root:root body.m-mode #scope-row { flex-wrap:wrap!important; } html:root:root:root:root body.m-mode #chat-scope .wrap { flex-wrap:wrap!important; }
html:root:root:root:root body.m-mode.in-app #assistant-chatbot { height:calc(var(--body-h, 600px) - 215px)!important; }

html:root:root:root:root body.m-mode #chat-scope { flex:1 1 100%!important;max-width:100%!important; }

/* ===== Assistant : bulles, logo AKORI, heure, barre de saisie ===== */
html:root:root:root:root body #assistant-chatbot { background:transparent!important;border:0!important;box-shadow:none!important; }
html:root:root:root:root body #assistant-chatbot .message-row { gap:10px!important; }
html:root:root:root:root body #assistant-chatbot .avatar-container { width:34px!important;height:34px!important;flex:none!important;border-radius:50%!important;overflow:hidden!important;
  background:#0f1424!important;border:1px solid var(--ak-line)!important;padding:3px!important;box-shadow:0 4px 14px rgba(79,109,245,.25)!important; }
html:root:root:root:root body #assistant-chatbot .avatar-container img { width:100%!important;height:100%!important;object-fit:contain!important;border-radius:50%!important; }
html:root:root:root:root body #assistant-chatbot .message-row .user.message {
  background:linear-gradient(135deg,#2b4ad8,#3f5ee8)!important;border:0!important;border-radius:18px 18px 4px 18px!important;max-width:min(560px,78%)!important;padding:10px 14px!important;box-shadow:0 6px 18px rgba(43,74,216,.25)!important; }
html:root:root:root:root body #assistant-chatbot .message-row .bot.message {
  background:var(--ak-card)!important;border:1px solid var(--ak-line)!important;border-radius:18px 18px 18px 4px!important;max-width:min(720px,86%)!important;padding:10px 14px!important; }
html:root:root:root:root body #assistant-chatbot .message-row .message .message, html:root:root:root:root body #assistant-chatbot .message-row [data-testid="user"], html:root:root:root:root body #assistant-chatbot .message-row [data-testid="bot"],
html:root:root:root:root body #assistant-chatbot .message-row .message-content, html:root:root:root:root body #assistant-chatbot .message-row .md { background:transparent!important;border:0!important;box-shadow:none!important;padding:0!important;margin:0!important;max-width:none!important; }
html:root:root:root:root body #assistant-chatbot .message-row .user.message, html:root:root:root:root body #assistant-chatbot .message-row .user.message * { color:#fff!important;-webkit-text-fill-color:#fff!important; }
.msg-time { display:block;font-size:11px;opacity:.6;margin-top:2px;text-align:right; }
#assistant-chatbot .bot .msg-time { text-align:left; }
html:root:root:root:root body #assistant-bar { background:var(--ak-card)!important;border:1px solid var(--ak-line)!important;border-radius:999px!important;padding:5px 6px 5px 8px!important;
  gap:6px!important;align-items:center!important;flex-wrap:nowrap!important; }
html:root:root:root:root body #assistant-bar #assistant-input, html:root:root:root:root body #assistant-bar #assistant-input textarea { background:transparent!important;border:0!important;box-shadow:none!important;min-height:38px!important;padding:8px 6px!important;font-size:14.5px!important; }
html:root:root:root:root body #chat-mic, html:root:root:root:root body #assistant-send { width:40px!important;min-width:40px!important;max-width:40px!important;height:40px!important;min-height:40px!important;padding:0!important;
  border-radius:50%!important;flex:none!important;font-size:16px!important;display:flex!important;align-items:center!important;justify-content:center!important; }
html:root:root:root:root body #chat-mic { background:transparent!important;border:0!important;box-shadow:none!important;color:var(--ak-muted)!important; }
html:root:root:root:root body #chat-mic.rec-on { color:#ef4444!important;animation:akpulse 1.2s ease-in-out infinite; }
html:root:root:root:root body #assistant-send { border-radius:12px!important;background:linear-gradient(135deg,#4f6df5,#3f5ee8)!important;color:#fff!important; }
/* ===== Historique : chronologie simple ===== */
.hx { max-width:860px;margin:0 auto;display:flex;flex-direction:column;gap:18px; }
.hx-date { display:flex;align-items:baseline;gap:10px;font-weight:800;font-size:15px;color:var(--ak-text);margin-bottom:8px; }
.hx-date span { font-weight:500;font-size:12px;color:var(--ak-muted); }
.hx-day ul { list-style:none;margin:0;padding:0;border:1px solid var(--ak-line);border-radius:16px;background:var(--ak-bg);overflow:hidden; }
.hx-item { display:grid;grid-template-columns:48px 34px minmax(0,1fr) auto;align-items:center;gap:10px;padding:10px 14px;border-top:1px solid var(--ak-line);animation:akrise .35s ease both; }
.hx-item:first-child { border-top:0; } .hx-item:hover { background:var(--ak-soft); }
.hx-time { font-size:12.5px;color:var(--ak-muted);font-variant-numeric:tabular-nums; }
.hx-ico { width:32px;height:32px;border-radius:10px;display:flex;align-items:center;justify-content:center;background:var(--ak-soft);color:var(--ak-blue); }
.hx-ico .ak-ico svg { width:17px;height:17px; }
.hx-main { min-width:0;display:flex;flex-direction:column; } .hx-main b { font-size:14px;color:var(--ak-text); }
.hx-detail { font-size:13px;color:var(--ak-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
.hx-course { font-size:12px;padding:3px 10px;border-radius:99px;background:var(--ak-soft);color:var(--ak-muted);max-width:180px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
html:root:root:root:root body.m-mode .hx-item { grid-template-columns:40px 30px minmax(0,1fr); } html:root:root:root:root body.m-mode .hx-course { display:none; }
/* ===== Mes dossiers : supprimer un PDF ===== */
.folder-card { position:relative; }
.doc-del { position:absolute;top:8px;right:8px;width:32px;height:32px;border-radius:10px;border:1px solid var(--ak-line);background:var(--ak-card);color:#ef4444;
  cursor:pointer;opacity:0;transform:scale(.85);transition:opacity .15s,transform .15s,background .15s;z-index:2;display:flex;align-items:center;justify-content:center;padding:0; }
.folder-card:hover .doc-del, .doc-del:focus { opacity:1;transform:none; } .doc-del:hover { background:#ef4444;color:#fff; }
html:root:root:root:root body.m-mode .doc-del { opacity:1;transform:none; }
.folder-card:hover .folder-status { visibility:hidden; }
#ak-confirm { position:fixed;inset:0;z-index:600;display:flex;align-items:center;justify-content:center;background:rgba(8,10,20,.6);backdrop-filter:blur(2px);animation:akfade .15s ease both; }
#ak-confirm .box { width:min(420px,92vw);background:var(--ak-card);border:1px solid var(--ak-line);border-radius:20px;padding:22px;box-shadow:0 30px 80px rgba(0,0,0,.4); }
#ak-confirm h3 { margin:0 0 8px;font-size:18px;color:var(--ak-text); } #ak-confirm p { margin:0 0 18px;color:var(--ak-muted);font-size:14px;line-height:1.5; }
#ak-confirm .row { display:flex;justify-content:flex-end;gap:10px; }
#ak-confirm button { min-height:42px;padding:0 18px;border-radius:12px;font:inherit;font-weight:700;cursor:pointer;border:1px solid var(--ak-line);background:var(--ak-soft);color:var(--ak-text); }
#ak-confirm button.danger { background:#ef4444;border-color:#ef4444;color:#fff; }


html:root:root:root:root body #assistant-chatbot .avatar-container { width:40px!important;height:40px!important;padding:0!important;background:transparent!important;border:0!important;box-shadow:0 6px 16px rgba(79,109,245,.35)!important;border-radius:12px!important; }
html:root:root:root:root body #assistant-chatbot .avatar-container img { border-radius:12px!important;object-fit:cover!important; }
html:root:root:root:root body.akori-dark #assistant-bar #assistant-input textarea, html:root:root:root:root body #assistant-bar #assistant-input textarea, html:root:root:root:root body #assistant-bar #assistant-input .wrap, html:root:root:root:root body #assistant-bar #assistant-input label { background:transparent!important;border:0!important;box-shadow:none!important; }

html:root:root:root:root body #assistant-bar #assistant-input, html:root:root:root:root body #assistant-bar #assistant-input * { background:transparent!important;background-color:transparent!important;box-shadow:none!important;border-color:transparent!important; }
.oral-card { border:1px solid var(--ak-line);border-radius:18px;background:var(--ak-card);padding:22px 26px; }
.oral-tag { display:inline-block;font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--ak-blue)!important;margin-bottom:8px; }
.oral-card h3 { margin:0;font-size:21px;line-height:1.4;color:var(--ak-text)!important; }
#oral-mic.rec-on { background:#dc2626!important;color:#fff!important;border-color:#dc2626!important;animation:akpulse 1.2s ease-in-out infinite; }
@keyframes akpulse { 50% { box-shadow:0 0 0 6px rgba(220,38,38,.18); } }
.oral-result { border:1px solid var(--ak-line);border-radius:18px;background:var(--ak-card);padding:20px 24px;margin-top:6px; }
.oral-top { display:flex;align-items:center;gap:20px; }
.oral-score { width:84px;height:84px;border-radius:50%;display:flex;align-items:baseline;justify-content:center;flex:none;padding-top:22px;border:5px solid; }
.oral-score b { font-size:30px; } .oral-score span { font-size:14px;margin-left:1px; }
.oral-good { border-color:#16a34a;color:#16a34a!important; } .oral-mid { border-color:#d97706;color:#d97706!important; } .oral-low { border-color:#dc2626;color:#dc2626!important; }
.oral-verdict { font-size:18px;font-weight:700;color:var(--ak-text); } .oral-feedback { margin:4px 0 0;color:var(--ak-muted);line-height:1.5; }
.oral-offline { color:var(--ak-muted);font-size:12.5px;margin:10px 0 0; }
.oral-cols { display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-top:18px; }
.oral-cols h4 { margin:0 0 8px;font-size:14px;color:var(--ak-text)!important; } .oral-cols ul { margin:0;padding-left:20px; }
.oral-cols li { margin:5px 0;color:var(--ak-text);line-height:1.45; } .oral-none { color:var(--ak-muted)!important;list-style:none;margin-left:-20px; }
.oral-reread { margin:10px 0 0;color:var(--ak-muted);font-size:13px; }
.oral-model { margin-top:16px;border-top:1px solid var(--ak-line);padding-top:12px; }
.oral-model summary { cursor:pointer;font-weight:600;color:var(--ak-blue); } .oral-model p { color:var(--ak-text);line-height:1.55; } .oral-src { color:var(--ak-muted)!important;font-size:12.5px; }
.oral-history { margin-top:18px;border:1px solid var(--ak-line);border-radius:16px;background:var(--ak-card);padding:14px 18px; }
.oral-history-head { display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;color:var(--ak-text); } .oral-history-head span { color:var(--ak-muted);font-size:13px; }
.oral-history ul { list-style:none;margin:10px 0 0;padding:0; } .oral-history li { display:flex;align-items:center;gap:12px;padding:6px 0;color:var(--ak-text);font-size:13.5px; }
.oral-mini { font-weight:700;font-size:12.5px;min-width:46px;text-align:center;padding:2px 8px;border-radius:999px;background:var(--ak-soft); }
/* ===== Console admin : liste des utilisateurs ===== */
.ad-list-title { font-size:17px;font-weight:700;color:var(--ak-text);padding:6px 4px; }
#admin-list-head { align-items:center!important;justify-content:space-between!important;flex-wrap:nowrap!important; }
#admin-search textarea, #admin-search input { border-radius:999px!important;background:var(--ak-soft)!important;border:1px solid var(--ak-line)!important;padding:10px 18px!important;min-height:42px!important; }
.ad-table-wrap { overflow-x:auto;border:1px solid var(--ak-line);border-radius:16px;background:var(--ak-card); }
.ad-table { width:100%;border-collapse:collapse;font-size:13.5px; }
.ad-table, .ad-table th, .ad-table td { border-left:0!important;border-right:0!important; }
.ad-table th { border-top:0!important;border-bottom:0!important; }
.ad-table th { background:var(--ak-soft);color:var(--ak-muted);font-weight:700;font-size:12px;text-align:left;padding:12px 14px;white-space:nowrap; }
.ad-table td { padding:11px 14px;border-top:1px solid var(--ak-line);color:var(--ak-text);vertical-align:middle; }
.ad-table tbody tr:hover { background:var(--ak-soft); }
.ad-no { color:var(--ak-muted)!important;width:36px; }
.ad-profile { display:flex;align-items:center;gap:11px; }
.ad-profile b { display:block;color:var(--ak-text);font-size:13.5px; }
.ad-profile small, .ad-sub { display:block;color:var(--ak-muted);font-size:11.5px;margin-top:1px; }
.ad-avatar { width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#4f6df5,#8e6cf4);color:#fff!important;font-weight:800;font-size:14px;display:inline-flex;align-items:center;justify-content:center;flex:none; }
.ad-avatar-lg { width:46px;height:46px;font-size:18px; }
.ad-pill { display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600; }
.ad-admin { background:rgba(124,58,237,.14);color:#7c3aed!important; } .ad-user { background:rgba(79,109,245,.12);color:#4f6df5!important; }
.ad-on { background:rgba(22,163,74,.13);color:#16a34a!important; } .ad-off { background:rgba(220,38,38,.12);color:#dc2626!important; }
.ad-pw { font-family:ui-monospace,monospace;letter-spacing:.12em;color:var(--ak-text); }
.ad-pwnote { margin-left:10px;font-family:inherit;letter-spacing:0;font-size:11.5px;color:var(--ak-muted); }
.ad-actions { white-space:nowrap;text-align:right; } .ad-actions .ak-act { margin-left:6px; }
.ad-card { border:1px solid var(--ak-line);border-radius:16px;background:var(--ak-card);padding:16px 18px;margin-top:12px; }
.ad-card-empty { color:var(--ak-muted);text-align:center; }
.ad-card-head { display:flex;align-items:center;gap:12px;flex-wrap:wrap; }
.ad-card-head b { display:block;font-size:16px;color:var(--ak-text); } .ad-card-head small { color:var(--ak-muted); }
.ad-card-head .ad-pill:first-of-type { margin-left:auto; }
.ad-card-grid { display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0; }
.ad-card-grid > div { background:var(--ak-soft);border-radius:12px;padding:10px 12px; }
.ad-card-grid span { display:block;font-size:11.5px;color:var(--ak-muted); } .ad-card-grid b { font-size:14px;color:var(--ak-text); }
.ad-card-pw { display:flex;align-items:center;gap:14px;flex-wrap:wrap;color:var(--ak-muted);font-size:12.5px; }
.ak-ico { display:inline-flex;align-items:center;justify-content:center;width:1.15em;height:1.15em;vertical-align:-0.2em;flex:none; }
.ak-ico svg { width:100%;height:100%;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round; }
.ak-ico-l { margin-right:.5em; } .ak-ico-r { margin-left:.5em; }
.ak-spin svg { animation:akspin 1s linear infinite; } @keyframes akspin { to { transform:rotate(360deg); } }
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
#ak-toast { position:fixed;left:50%;bottom:7.5rem;transform:translateX(-50%) translateY(20px);z-index:99999;background:#182033;color:#fff;
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

html:root:root:root:root body #login-wrap #auth-tabs-bar, html:root:root:root:root body #login-wrap #auth-tabs-bar .html-container, html:root:root:root:root body #login-wrap #auth-tabs-bar .prose { padding:0!important;margin-left:0!important;margin-right:0!important;max-width:none!important; }
html:root:root:root:root body #login-wrap .auth-tabs { padding:0!important; }
html:root:root:root:root body #login-wrap .auth-card .pw-eye, html:root:root:root:root body.akori-dark #login-wrap .auth-card .pw-eye { background:transparent!important;border:0!important;box-shadow:none!important;color:#a6a39b!important;right:.6rem!important; }
html:root:root:root:root body #login-wrap .auth-card .pw-eye:hover { background:rgba(250,249,245,.08)!important;color:#faf9f5!important; }

html:root:root:root:root body #login-wrap { gap:.9rem!important; }
html:root:root:root:root body #login-wrap #auth-tabs-bar { margin-bottom:calc(-.9rem - 1px)!important; }
html:root:root:root:root body .gradio-container .auth-tabs button.auth-tab.auth-tab, html:root:root:root:root body.akori-dark .gradio-container .auth-tabs button.auth-tab.auth-tab { background:transparent!important;border:1px solid transparent!important;border-bottom:0!important;
  color:#a6a39b!important;-webkit-text-fill-color:#a6a39b!important;box-shadow:none!important;min-height:0!important;border-radius:1rem 1rem 0 0!important;padding:.7rem 1.2rem .8rem!important;width:auto!important; }
html:root:root:root:root body .gradio-container .auth-tabs button.auth-tab.auth-tab:hover, html:root:root:root:root body.akori-dark .gradio-container .auth-tabs button.auth-tab.auth-tab:hover { color:#faf9f5!important;-webkit-text-fill-color:#faf9f5!important;background:rgba(250,249,245,.04)!important; }
html:root:root:root:root body .gradio-container .auth-tabs button.auth-tab.auth-tab.on, html:root:root:root:root body.akori-dark .gradio-container .auth-tabs button.auth-tab.auth-tab.on { background:#262624!important;border-color:#3d3d3a!important;color:#faf9f5!important;-webkit-text-fill-color:#faf9f5!important; }
html:root:root:root:root body #login-wrap .auth-card .pw-eye { bottom:calc(10px + (2.85rem - 30px) / 2)!important;top:auto!important; }
html:root:root:root:root body.auth-register #register-panel.auth-card { border-top-left-radius:24px!important; }

/* ===== Connexion : onglets posés sur la carte, champs homogènes, sans artefacts ===== */
html:root:root:root:root body #auth-tabs-bar { margin:0 0 -1px!important;padding:0!important;position:relative;z-index:3; }
.auth-tabs { display:flex;gap:.35rem;padding:0 .2rem; }
.auth-tab { position:relative;flex:0 1 auto;padding:.7rem 1.2rem .75rem;border:1px solid transparent;border-bottom:0;border-radius:1rem 1rem 0 0;
  background:transparent;color:#a6a39b;font:inherit;font-size:.95rem;font-weight:700;cursor:pointer;transition:color .15s,background .15s; }
.auth-tab:hover { color:#faf9f5; }
.auth-tab.on { background:#262624;border-color:#3d3d3a;color:#faf9f5;animation:aktabin .3s cubic-bezier(.2,.8,.2,1) both; }
.auth-tab.on::after { content:"";position:absolute;bottom:-1px;right:-1rem;width:1rem;height:1rem;pointer-events:none;
  background:radial-gradient(circle at 100% 0, transparent calc(1rem - .5px), #3d3d3a calc(1rem - .5px), #3d3d3a 1rem, #262624 calc(1rem + .5px)); }
.auth-tab.on:not(:first-child)::before { content:"";position:absolute;bottom:-1px;left:-1rem;width:1rem;height:1rem;pointer-events:none;
  background:radial-gradient(circle at 0 0, transparent calc(1rem - .5px), #3d3d3a calc(1rem - .5px), #3d3d3a 1rem, #262624 calc(1rem + .5px)); }
html:root:root:root:root body .auth-card { position:relative;z-index:2; }
html:root:root:root:root body.auth-login #login-panel.auth-card { border-top-left-radius:0!important; }
/* les onglets remplacent « OU / Créer un compte » et le titre de la carte d'inscription */
html:root:root:root:root body #login-wrap .auth-or, html:root:root:root:root body #login-wrap .auth-alt, html:root:root:root:root body #login-wrap .auth-card-title { display:none!important; }
/* champs : même fond partout, plus de coins colorés autour des blocs */
html:root:root:root:root body .auth-card .block, html:root:root:root:root body .auth-card .form, html:root:root:root:root body .auth-card .wrap, html:root:root:root:root body .auth-card label, html:root:root:root:root body .auth-card .input-container,
html:root:root:root:root body .auth-card .container, html:root:root:root:root body .auth-card fieldset { background:transparent!important;border-color:transparent!important;box-shadow:none!important; }
html:root:root:root:root body.akori-dark .auth-card input[type="text"], html:root:root:root:root body.akori-dark .auth-card input[type="password"], html:root:root:root:root body.akori-dark .auth-card input[type="email"], html:root:root:root:root body.akori-dark .auth-card textarea,
html:root:root:root:root body .auth-card input[type="text"], html:root:root:root:root body .auth-card input[type="password"], html:root:root:root:root body .auth-card textarea {
  background:#30302e!important;border:1px solid #4a4945!important;border-radius:.8rem!important;color:#faf9f5!important;-webkit-text-fill-color:#faf9f5!important;
  min-height:2.85rem!important;padding:.7rem 2.8rem .7rem .9rem!important;transition:border-color .15s,box-shadow .15s!important; }
html:root:root:root:root body .auth-card input:focus, html:root:root:root:root body .auth-card textarea:focus { border-color:#8a877f!important;box-shadow:0 0 0 3px rgba(250,249,245,.08)!important;outline:none!important; }
/* remplissage automatique du navigateur : mêmes couleurs que le champ */
html:root:root:root:root body .auth-card input:-webkit-autofill, html:root:root:root:root body .auth-card input:-webkit-autofill:hover, html:root:root:root:root body .auth-card input:-webkit-autofill:focus,
html:root:root:root:root body .auth-card textarea:-webkit-autofill, html:root:root:root:root body .auth-card input:autofill {
  -webkit-box-shadow:0 0 0 1000px #30302e inset!important;box-shadow:0 0 0 1000px #30302e inset!important;-webkit-text-fill-color:#faf9f5!important;
  caret-color:#faf9f5!important;border:1px solid #4a4945!important;transition:background-color 99999s ease-out 0s!important; }
html:root:root:root:root body .auth-card .pw-eye { background:transparent!important;color:#a6a39b!important;border:0!important;box-shadow:none!important;right:.7rem!important;bottom:.5rem!important; }
html:root:root:root:root body .auth-card .pw-eye:hover { color:#faf9f5!important;background:rgba(250,249,245,.08)!important; }
html:root:root:root:root body #login-wrap .link-btn, html:root:root:root:root body #login-wrap .link-btn.gr-button { background:transparent!important;border:0!important;box-shadow:none!important;color:#a6a39b!important;-webkit-text-fill-color:#a6a39b!important;min-height:0!important;padding:.25rem .5rem!important; }
html:root:root:root:root body #login-wrap .link-btn:hover { color:#faf9f5!important;-webkit-text-fill-color:#faf9f5!important;text-decoration:underline; }
/* plus d'indicateur de chargement Gradio sous le bouton : le bouton affiche lui-même l'attente */
html:root:root:root:root body #login-wrap [data-testid="status-tracker"], html:root:root:root:root body #login-wrap .wrap.default, html:root:root:root:root body #login-wrap .wrap.translucent, html:root:root:root:root body #login-wrap .eta-bar, html:root:root:root:root body #login-wrap .progress-text { display:none!important; }
html:root:root:root:root body #login-wrap .auth-status, html:root:root:root:root body #login-wrap .auth-status p { min-height:0!important; }
html:root:root:root:root body #login-wrap .auth-status:empty, html:root:root:root:root body #login-wrap .auth-status:has(.prose:empty) { display:none!important; }
html:root:root:root:root body #li-btn, html:root:root:root:root body #su-btn { position:relative;overflow:hidden;transition:background .15s,transform .1s!important; }
html:root:root:root:root body #li-btn:active, html:root:root:root:root body #su-btn:active { transform:scale(.985); }
html:root:root:root:root body #li-btn.busy, html:root:root:root:root body #su-btn.busy { color:transparent!important;-webkit-text-fill-color:transparent!important;pointer-events:none; }
html:root:root:root:root body #li-btn.busy::after, html:root:root:root:root body #su-btn.busy::after { content:"";position:absolute;left:50%;top:50%;width:1.2rem;height:1.2rem;margin:-.6rem 0 0 -.6rem;border-radius:50%;
  border:2px solid rgba(31,30,29,.25);border-top-color:#1f1e1d;animation:akspinb .7s linear infinite; }
@keyframes akspinb { to { transform:rotate(360deg); } }
html:root:root:root:root body.m-mode #login-wrap { padding:0 .25rem!important; }
html:root:root:root:root body.m-mode .auth-tab { padding:.65rem .9rem .7rem;font-size:.9rem; }

/* =====================================================================================
   STABILITÉ AU ZOOM ET AU REDIMENSIONNEMENT
   - box-sizing partout, médias bornés à leur conteneur, enfants flexibles autorisés à rétrécir ;
   - tailles de mise en page en rem / % / vh avec clamp() ;
   - points de rupture posés par le script sur <body> (ak-narrow, ak-short, m-mode) : plus fiables
     que les @media, que Gradio réécrit en préfixant les sélecteurs.
   ===================================================================================== */
*, *::before, *::after { box-sizing:border-box; }
#app-view img, #app-view svg, #app-view video, #app-view iframe, #app-view canvas { max-width:100%; }
#app-view .row, #app-view .column, #app-view .block, #app-view .form { min-width:0!important; }
#app-view .prose, #app-view .html-container, .folder-card, .mini-stat, .plan-task, .hx-item, .ak-pane { overflow-wrap:anywhere; }
:root { --sb-open:clamp(13.5rem, 19vw, 16.5rem); --sb-rail:4.5rem; }

/* --- en-tête : l'onglet de la page « posé » sur la carte du corps --- */
html:root:root:root:root body.in-app #main-column { gap:0!important; }
html:root:root:root:root body.in-app #topbar { background:transparent!important;border:0!important;box-shadow:none!important;padding:0!important;margin:0 0 -1px 0!important;
  min-height:0!important;align-items:flex-end!important;flex-wrap:wrap!important;gap:.5rem .75rem!important;position:relative;z-index:3;overflow:visible!important; }
html:root:root:root:root body.in-app #topbar > #topbar-title { flex:0 1 auto!important;max-width:100%!important;min-width:0!important;position:relative;
  background:var(--ak-card)!important;border:1px solid var(--ak-line)!important;border-bottom-color:var(--ak-card)!important;
  border-radius:1rem 1rem 0 0!important;padding:.6rem 1.2rem .7rem 1rem!important;box-shadow:0 -0.5rem 1.25rem rgba(0,0,0,.08); }
html:root:root:root:root body.in-app #topbar > #topbar-title::after { content:"";position:absolute;right:-1rem;bottom:-1px;width:1rem;height:1rem;pointer-events:none;
  background:radial-gradient(circle at 100% 0, transparent calc(1rem - .5px), var(--ak-line) calc(1rem - .5px), var(--ak-line) 1rem, var(--ak-card) calc(1rem + .5px)); }
html:root:root:root:root body.in-app #topbar > #topbar-title.tab-anim { animation:aktabin .35s cubic-bezier(.2,.8,.2,1) both; }
@keyframes aktabin { from { opacity:.35;transform:translateY(.5rem); } to { opacity:1;transform:none; } }
html:root:root:root:root body.in-app #topbar > #topbar-title { width:auto!important; }
html:root:root:root:root body.in-app #topbar > .form:has(#doc-select), html:root:root:root:root body.in-app #topbar > #doc-select { flex:0 1 clamp(12rem, 28vw, 21rem)!important;width:auto!important;margin:0 0 .45rem auto!important;min-width:0!important;
  background:transparent!important;border:0!important;box-shadow:none!important; }
html:root:root:root:root body.in-app #topbar .form:has(#doc-select) > #doc-select { flex:1 1 auto!important;width:100%!important;min-width:0!important; }
html:root:root:root:root body.in-app #main-tabs { border-top-left-radius:0!important;position:relative;z-index:2; }
#ak-page-title { font-size:clamp(1rem, .9rem + .45vw, 1.2rem); }
.pt-sub { font-size:clamp(.66rem, .62rem + .15vw, .74rem); }
.pt-ico { height:clamp(1.6rem, 4vh, 2.1rem); }

/* --- grilles fluides : le nombre de colonnes suit la place disponible --- */
html:root:root:root:root body .home-fit .guide-grid { grid-template-columns:repeat(auto-fill, minmax(min(100%, 8.75rem), 1fr))!important; }
html:root:root:root:root body .stat-row { grid-template-columns:repeat(auto-fit, minmax(min(100%, 9.5rem), 1fr))!important; }
html:root:root:root:root body .folder-grid { grid-template-columns:repeat(auto-fill, minmax(min(100%, 12rem), 1fr))!important; }
html:root:root:root:root body .mod-grid, html:root:root:root:root body .ld-three, html:root:root:root:root body .plan-kpis, html:root:root:root:root body .jr-kpis { grid-template-columns:repeat(auto-fit, minmax(min(100%, 11rem), 1fr))!important; }
.hero-compact h1 { font-size:clamp(1.25rem, 1rem + 1.2vw, 1.9rem)!important; }
.hero-compact .hero-visual { flex:0 1 clamp(8rem, 18vw, 15rem)!important; }
html:root:root:root:root body .ak-toolbar, html:root:root:root:root body #scope-row, html:root:root:root:root body #chat-scope .wrap, html:root:root:root:root body .hero-chips, html:root:root:root:root body .act-bar-row { flex-wrap:wrap!important; }
html:root:root:root:root body #assistant-bar { flex-wrap:nowrap!important;max-width:100%; }
html:root:root:root:root body #assistant-bar #assistant-input { min-width:0!important;flex:1 1 auto!important; }

/* --- écran étroit (zoom avant, petite fenêtre) : les deux colonnes passent l'une sous l'autre --- */
html:root:root:root:root body.ak-narrow .ak-cols { flex-direction:column!important;flex-wrap:nowrap!important; }
html:root:root:root:root body.ak-narrow .ak-cols > .ak-pane { width:100%!important;min-width:0!important;flex:1 1 auto!important; }
html:root:root:root:root body.ak-narrow .hero-compact .hero-visual { display:none!important; }
html:root:root:root:root body.ak-narrow #documents-row { flex-wrap:wrap!important; } html:root:root:root:root body.ak-narrow #documents-row > * { flex:1 1 100%!important; }

/* --- fenêtre basse (fort zoom) : on abandonne « tout dans l'écran » et la page défile normalement --- */
html:root:has(body.in-app.ak-short) { overflow-y:auto!important; }
html:root:root:root:root body.in-app.ak-short { overflow:visible!important; }
html:root:root:root:root body.in-app.ak-short #main-column { height:auto!important;overflow:visible!important; }
html:root:root:root:root body.in-app.ak-short #main-tabs { flex:none!important;overflow:visible!important; }
html:root:root:root:root body.in-app.ak-short #main-tabs > .tabitem { overflow:visible!important; }
html:root:root:root:root body.in-app.ak-short #assistant-chatbot { height:max(18rem, 70vh)!important; }
html:root:root:root:root body.in-app.ak-short #summary-out { height:auto!important;max-height:none!important; }
html:root:root:root:root body.in-app.ak-short .mm-wrap { max-height:none!important; }
html:root:root:root:root body.in-app.ak-short #main-tabs iframe { height:min(28rem, 85vh)!important; }
/* barre latérale compacte quand la hauteur manque */
html:root:root:root:root body.ak-short #sidebar .brand-sub, html:root:root:root:root body.ak-short #sidebar .nav-title { display:none!important; }
html:root:root:root:root body.ak-short #sidebar .navbtn { min-height:clamp(1.55rem, 5.2vh, 2.2rem)!important; }
html:root:root:root:root body.ak-short #sidebar .brand { margin-bottom:.35rem!important; }
html:root:root:root:root body #sidebar { overflow-y:auto!important;overscroll-behavior:contain; }
html:root:root:root:root body.ak-sbtight #sidebar .brand-sub, html:root:root:root:root body.ak-sbtight #sidebar .nav-title { display:none!important; }
html:root:root:root:root body.ak-sbtight #sidebar .brand { margin:0 0 .3rem!important; }
html:root:root:root:root body.ak-sbtight #sidebar #sidebar-toggle { height:2rem!important;margin-bottom:.15rem!important; }
html:root:root:root:root body.ak-sbtight #sidebar .navbtn { min-height:clamp(1.15rem, calc((100vh - 17rem) / 17), 2.4rem)!important;height:auto!important; }
html:root:root:root:root body.ak-sbtight #sidebar { padding-top:.6rem!important;padding-bottom:.4rem!important; }

/* --- téléphone : le sélecteur de cours passe au-dessus de l'onglet --- */
html:root:root:root:root body.ak-narrow.in-app #topbar > .form:has(#doc-select), html:root:root:root:root body.ak-narrow.in-app #topbar > #doc-select { order:-1;flex:1 1 100%!important;margin:0 0 .5rem 0!important; }
html:root:root:root:root body.m-mode.in-app #topbar > .form:has(#doc-select), html:root:root:root:root body.m-mode.in-app #topbar > #doc-select { order:-1;flex:1 1 100%!important;margin:0 0 .5rem 0!important; }
html:root:root:root:root body.m-mode.in-app #topbar > #topbar-title { flex:1 1 auto!important; }
html:root:root:root:root body.m-mode.in-app #topbar > #topbar-title::after { display:none; }
/* ===== Progression : vue d'ensemble cliquable + détail par cours en carrousel ===== */
.gp-open { cursor:pointer;transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease; }
.gp-open:hover, .gp-open:focus-visible { transform:translateY(-3px);box-shadow:0 18px 40px rgba(0,0,0,.25);outline:none; }
.gp-hint { display:inline-flex;margin-top:.5rem;font-size:.8rem;font-weight:700;color:var(--ak-text);opacity:.75; }
.gp-open:hover .gp-hint { opacity:1; }
#app-view .gp-modal { display:none; }
body > .gp-modal { position:fixed;inset:0;z-index:700;display:flex;align-items:center;justify-content:center;overflow:hidden;
  background:radial-gradient(40rem circle at 30% 20%, rgba(46,163,131,.22), transparent 60%), radial-gradient(36rem circle at 75% 30%, rgba(224,164,58,.18), transparent 60%), rgba(12,14,20,.82);
  backdrop-filter:blur(10px);animation:akfade .2s ease both; }
body > .gp-modal .gp-stage { position:relative;width:min(22rem,78vw);height:min(36rem,82vh); }
body > .gp-modal .gp-card { position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;padding:1.4rem 1.4rem 0;border-radius:2rem;
  background:#1b1c22;border:1px solid rgba(255,255,255,.08);box-shadow:0 30px 70px rgba(0,0,0,.45);color:#f2f3f7;overflow:hidden;
  transition:transform .45s cubic-bezier(.2,.8,.2,1),opacity .45s ease,filter .45s ease;opacity:0;pointer-events:none;transform:scale(.7); }
body > .gp-modal .gp-card.pos0 { opacity:1;transform:none;pointer-events:auto;z-index:3; }
body > .gp-modal .gp-card.posL { opacity:.55;transform:translateX(-78%) scale(.84);filter:blur(3px);z-index:2; }
body > .gp-modal .gp-card.posR { opacity:.55;transform:translateX(78%) scale(.84);filter:blur(3px);z-index:2; }
.gp-glow { position:absolute;inset:-30% -20% auto;height:70%;background:radial-gradient(closest-side, var(--c), transparent);opacity:.35;pointer-events:none; }
.gp-date { position:relative;align-self:flex-start;font-size:.9rem;color:#c9cbd4; }
body > .gp-modal .gp-card h3 { position:relative;margin:1rem 0 .2rem;font-size:clamp(1.2rem,4vw,1.55rem);font-weight:800;text-align:center;color:#fff; }
.gp-sub { position:relative;font-size:.9rem;font-weight:600;color:#b9bcc8; }
.gp-ringwrap { position:relative;width:min(11rem,52vw);aspect-ratio:1;margin:1rem 0 .6rem;display:flex;align-items:center;justify-content:center; }
.gp-ring { position:absolute;inset:0;width:100%;height:100%; }
.gp-ring circle { fill:none;stroke-width:14; } .gp-track-c { stroke:#26304a; }
.gp-val { stroke-linecap:round;transition:stroke-dasharray .8s ease; }
.gp-pct { position:relative;font-size:clamp(1.8rem,7vw,2.4rem);font-weight:800;color:#fff; }
.gp-bars { position:relative;width:100%;display:grid;gap:.45rem;margin-bottom:auto; }
.gp-bar { display:grid;grid-template-columns:5.5rem 1fr 2.6rem;align-items:center;gap:.5rem;font-size:.78rem;color:#c9cbd4; }
.gp-bar > div { height:.4rem;border-radius:99px;background:#2a2c36;overflow:hidden; } .gp-bar i { display:block;height:100%;border-radius:99px; }
.gp-bar em { font-style:normal;text-align:right;color:#fff;font-weight:700; }
body > .gp-modal .gp-card .gp-foot { position:relative;width:calc(100% + 2.8rem);margin-top:1rem;padding:.9rem 1.4rem;border-top:1px solid rgba(255,255,255,.08);
  display:flex;align-items:center;justify-content:space-between;gap:.5rem; }
.gp-dots { display:flex; } .gp-dot { width:1.9rem;height:1.9rem;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-left:-.35rem;
  background:#2a2c36;border:2px solid #1b1c22;color:#7d8090;font-size:.8rem; } .gp-dot:first-child { margin-left:0; }
.gp-dot.on { background:var(--c);color:#fff; } .gp-dot .ak-ico svg { width:.9rem;height:.9rem; }
.gp-chip { padding:.35rem .9rem;border-radius:99px;background:#2a2c36;color:#e6e7ec;font-size:.8rem;font-weight:600;white-space:nowrap; }
body > .gp-modal .gp-nav { position:absolute;top:50%;transform:translateY(-50%);z-index:5;width:3.6rem;height:3.6rem;border-radius:50%;border:1px solid rgba(255,255,255,.25);
  background:rgba(255,255,255,.16);backdrop-filter:blur(8px);color:#fff;font-size:2rem;line-height:1;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background .15s,transform .15s; }
body > .gp-modal .gp-nav:hover { background:rgba(255,255,255,.28); }
body > .gp-modal .gp-prev { left:max(1rem, calc(50% - min(22rem,78vw)/2 - 4.6rem)); }
body > .gp-modal .gp-next { right:max(1rem, calc(50% - min(22rem,78vw)/2 - 4.6rem)); }
body > .gp-modal .gp-close { position:absolute;top:1rem;right:1rem;z-index:6;width:2.6rem;height:2.6rem;border-radius:50%;border:1px solid rgba(255,255,255,.2);background:rgba(255,255,255,.1);color:#fff;font-size:1rem;cursor:pointer; }
.gp-pager { position:absolute;bottom:max(1rem,3vh);left:50%;transform:translateX(-50%);display:flex;gap:.4rem;z-index:5; }
.gp-pager i { width:.5rem;height:.5rem;border-radius:99px;background:rgba(255,255,255,.3);transition:width .2s,background .2s; } .gp-pager i.on { width:1.4rem;background:#fff; }

/* ===== Champs de texte : bords neutres (plus de bleu) ===== */
html:root:root:root:root body input[type="text"], html:root:root:root:root body input[type="password"], html:root:root:root:root body input[type="email"], html:root:root:root:root body input[type="number"], html:root:root:root:root body input[type="date"], html:root:root:root:root body textarea, html:root:root:root:root body select,
html:root:root:root:root body.akori-dark input[type="text"], html:root:root:root:root body.akori-dark input[type="password"], html:root:root:root:root body.akori-dark input[type="number"], html:root:root:root:root body.akori-dark textarea { border-color:var(--ak-field-border)!important; }
html:root:root:root:root body input:focus, html:root:root:root:root body textarea:focus, html:root:root:root:root body select:focus, html:root:root:root:root body.akori-dark input:focus, html:root:root:root:root body.akori-dark textarea:focus {
  border-color:color-mix(in srgb, var(--ak-text) 35%, transparent)!important;box-shadow:0 0 0 3px color-mix(in srgb, var(--ak-text) 8%, transparent)!important;outline:none!important; }
html:root:root:root:root body .block:focus-within, html:root:root:root:root body .wrap:focus-within, html:root:root:root:root body .input-container:focus-within, html:root:root:root:root body label:focus-within { box-shadow:none!important;border-color:transparent!important; }
html:root:root:root:root body #login-wrap .auth-card input, html:root:root:root:root body #login-wrap .auth-card textarea { border-color:#4a4945!important; }
html:root:root:root:root body #login-wrap .auth-card input:focus, html:root:root:root:root body #login-wrap .auth-card textarea:focus { border-color:#8a877f!important;box-shadow:0 0 0 3px rgba(250,249,245,.08)!important; }
html:root:root:root:root body .ak-datepick { border-color:var(--ak-line)!important; }

/* ===== Bleu réservé aux labels (ex. « Cours actif ») : titres, sur-titres et chiffres en couleurs neutres ===== */
html:root:root:root:root body .pane-title, html:root:root:root:root body .eyebrow, html:root:root:root:root body.akori-dark .eyebrow, html:root:root:root:root body .card-label, html:root:root:root:root body.akori-dark .card-label, html:root:root:root:root body .oral-tag, html:root:root:root:root body .sem-title,
html:root:root:root:root body .quiz-result-kicker, html:root:root:root:root body.akori-dark .quiz-result-kicker, html:root:root:root:root body .global-list-title { color:var(--ak-muted)!important;-webkit-text-fill-color:var(--ak-muted)!important; }
html:root:root:root:root body .plan-kpis b, html:root:root:root:root body .jr-kpis b, html:root:root:root:root body .mini-stat b, html:root:root:root:root body .stat-row .mini-stat:hover b { color:var(--ak-text)!important; }
html:root:root:root:root body .hx-ico, html:root:root:root:root body .jr-ico { color:var(--ak-text)!important; }

:root { --ak-field-border:#d9dde5; }
html:root:root:root:root body.akori-dark { --ak-field-border:#363a44; }
html:root:root:root:root body .ak-datepick { border-color:var(--ak-field-border)!important; }
html:root:root:root:root body.akori-dark #login-wrap .auth-card .input-container, html:root:root:root:root body.akori-dark #login-wrap .auth-card .wrap, html:root:root:root:root body #login-wrap .auth-card .input-container { background:transparent!important;border-color:transparent!important;box-shadow:none!important; }

/* ===== Progression : couleurs alignées sur l'application (bleu → violet, cartes sombres de l'app) ===== */
body > .gp-modal { background:radial-gradient(40rem circle at 30% 20%, rgba(79,109,245,.22), transparent 60%), radial-gradient(36rem circle at 75% 30%, rgba(142,108,244,.18), transparent 60%), rgba(10,14,24,.84)!important; }
body > .gp-modal .gp-card { background:#161e30!important;border-color:#27324a!important; }
.gp-track-c { stroke:#27324a!important; } .gp-bar > div, .gp-dot, .gp-chip { background:#1f2a42!important; } .gp-dot { border-color:#161e30!important; }
.gp-bar i { background:linear-gradient(90deg,#4f6df5,#8e6cf4)!important; }
.gp-date, .gp-bar, .gp-sub { color:#9aa6bf!important; }
html:root:root:root:root body .global-progress-ring { background:conic-gradient(#4f6df5 var(--progress,0%), var(--ak-soft) 0)!important; }
html:root:root:root:root body .progress-ring { background:conic-gradient(#4f6df5 var(--progress,0%), var(--ak-soft) 0)!important;flex:none!important;aspect-ratio:1; }
html:root:root:root:root body .global-metrics { grid-template-columns:repeat(2,minmax(0,1fr))!important; }
html:root:root:root:root body .global-metrics b { color:var(--ak-text)!important; }

/* ===== Plan de révision : éléments communs (anneau, statuts, boutons) ===== */
html:root:root:root:root body .pl-ringw { position:relative;width:2.9rem;height:2.9rem;flex:none;display:flex;align-items:center;justify-content:center; }
html:root:root:root:root body .pl-ring { position:absolute;inset:0;width:100%;height:100%; }
html:root:root:root:root body .pl-ring circle { fill:none;stroke-width:5; } html:root:root:root:root body .pl-ring-t { stroke:var(--ak-soft); } html:root:root:root:root body .pl-ring-v { stroke:#4f6df5;stroke-linecap:round; }
html:root:root:root:root body .pl-ringw b { position:relative;font-size:.7rem;color:var(--ak-text); }
html:root:root:root:root body .pl-st { font-size:.7rem;font-weight:700;padding:.16rem .6rem;border-radius:99px;white-space:nowrap; }
html:root:root:root:root body .st-todo, html:root:root:root:root body .st-ok { background:rgba(46,163,131,.16);color:#2ea383; }
html:root:root:root:root body .st-run { background:rgba(224,164,58,.18);color:#d4952a; }
html:root:root:root:root body .st-opt { background:var(--ak-soft);color:var(--ak-muted); }
html:root:root:root:root body .st-plan { background:rgba(79,109,245,.14);color:#6b84f7; }
html:root:root:root:root body .pl-det, html:root:root:root:root body .pl-go { flex:1;min-width:0;border-radius:.6rem;padding:.45rem .5rem;font-size:.78rem;font-weight:700;cursor:pointer;white-space:nowrap;transition:filter .15s ease,background .15s ease; }
html:root:root:root:root body .pl-go, html:root:root:root:root body.akori-dark .pl-go { background:#4f6df5!important;border:1px solid #4f6df5!important;color:#fff!important;-webkit-text-fill-color:#fff!important; } html:root:root:root:root body .pl-go:hover { filter:brightness(1.1); }
html:root:root:root:root body .pl-btn { flex:1 1 9rem;border-radius:.75rem;padding:.62rem 1rem;font-weight:700;font-size:.84rem;cursor:pointer;border:1px solid var(--ak-line);background:var(--ak-card);color:var(--ak-text);display:inline-flex;justify-content:center;align-items:center;gap:.35rem;transition:background .15s ease,filter .15s ease; }
html:root:root:root:root body .pl-btn:hover { background:var(--ak-soft); }
html:root:root:root:root body .pl-btn.pl-primary, html:root:root:root:root body.akori-dark .pl-btn.pl-primary { background:#4f6df5!important;border-color:#4f6df5!important;color:#fff!important;-webkit-text-fill-color:#fff!important; } html:root:root:root:root body .pl-btn.pl-primary:hover { filter:brightness(1.1);background:#4f6df5; }
html:root:root:root:root body .pl-btn .ak-ico svg { width:.95rem;height:.95rem; }
html:root:root:root:root body.m-mode .pl-det, html:root:root:root:root body.m-mode .pl-go { white-space:normal;font-size:.74rem;padding:.45rem .35rem; }
html:root:root:root:root body #upload-opts { gap:.6rem;flex-wrap:wrap; }
html:root:root:root:root body.m-mode #upload-opts, html:root:root:root:root body.m-mode #upload-opts > .form { flex-direction:column!important;flex-wrap:nowrap!important; } html:root:root:root:root body.m-mode #upload-opts .block { width:100%!important;min-width:0!important;flex:1 1 auto!important; }
html:root:root:root:root body .pl-det, html:root:root:root:root body .pl-go { white-space:normal!important;line-height:1.2; }
html:root:root:root:root body .gradio-container button.pl-go.pl-go, html:root:root:root:root body.akori-dark .gradio-container button.pl-go.pl-go, html:root:root:root:root body .gradio-container button.pl-btn.pl-primary, html:root:root:root:root body.akori-dark .gradio-container button.pl-btn.pl-primary, html:root:root:root:root body .gradio-container button.pl-mins.pl-mins.on, html:root:root:root:root body.akori-dark .gradio-container button.pl-mins.pl-mins.on { background:#4f6df5!important;border:1px solid #4f6df5!important;color:#fff!important;-webkit-text-fill-color:#fff!important; }
html:root:root:root:root body .gradio-container button.pl-btn.pl-btn:not(.pl-primary), html:root:root:root:root body.akori-dark .gradio-container button.pl-btn.pl-btn:not(.pl-primary) { background:var(--ak-card)!important;color:var(--ak-text)!important;-webkit-text-fill-color:var(--ak-text)!important;border:1px solid var(--ak-line)!important; }
html:root:root:root:root body .gradio-container button.pl-det.pl-det:hover, html:root:root:root:root body.akori-dark .gradio-container button.pl-det.pl-det:hover, html:root:root:root:root body .gradio-container button.pl-btn.pl-btn:not(.pl-primary):hover, html:root:root:root:root body.akori-dark .gradio-container button.pl-btn.pl-btn:not(.pl-primary):hover, html:root:root:root:root body .gradio-container button.pl-mins.pl-mins:not(.on):hover, html:root:root:root:root body.akori-dark .gradio-container button.pl-mins.pl-mins:not(.on):hover { background:var(--ak-soft)!important; }
html:root:root:root:root body .gradio-container button.pl-go.pl-go:hover, html:root:root:root:root body.akori-dark .gradio-container button.pl-go.pl-go:hover, html:root:root:root:root body .gradio-container button.pl-btn.pl-primary:hover, html:root:root:root:root body.akori-dark .gradio-container button.pl-btn.pl-primary:hover { filter:brightness(1.1); }
/* ===== Plan de révision : assistant en 5 étapes (carte bleue, mascotte, boutons arrondis) ===== */
html:root:root:root:root body #plan-wizard { position:relative;display:flex!important;flex-direction:column!important;flex-wrap:nowrap!important;gap:.7rem!important;
  min-height:calc(var(--body-h, 600px) - 2rem);padding:1rem 1.4rem 1.1rem!important;border-radius:1.4rem!important;overflow:hidden;color:#eef4ff;
  background:radial-gradient(50rem 26rem at 85% -12%, rgba(120,175,255,.38), transparent 60%), linear-gradient(160deg,#21479b 0%,#16357a 48%,#0f2558 100%)!important;
  border:1px solid rgba(140,185,255,.28)!important;box-shadow:0 18px 50px rgba(15,36,87,.32)!important; }
html:root:root:root:root body #plan-wizard .block, html:root:root:root:root body #plan-wizard .form, html:root:root:root:root body #plan-wizard fieldset,
html:root:root:root:root body #plan-wizard .wrap, html:root:root:root:root body #plan-wizard .html-container, html:root:root:root:root body #plan-wizard .prose,
html:root:root:root:root body #plan-wizard .container, html:root:root:root:root body #plan-wizard .input-container, html:root:root:root:root body #plan-wizard .column, html:root:root:root:root body #plan-wizard .row {
  background:transparent!important;border-color:transparent!important;box-shadow:none!important; }
html:root:root:root:root body #plan-wizard .html-container, html:root:root:root:root body #plan-wizard .prose { padding:0!important; }
.pw-head { position:relative;text-align:center;padding-top:.2rem; }
.pw-bar { height:.38rem;border-radius:99px;background:rgba(255,255,255,.14);overflow:hidden;margin-right:8.5rem; }
.pw-bar i { display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#7cc4ff,#4f7bff);box-shadow:0 0 12px rgba(124,196,255,.7);transition:width .45s cubic-bezier(.2,.8,.2,1); }
.pw-brand { position:absolute;top:-.35rem;right:0;font-size:.95rem;font-weight:600;color:#e3ecff; }
html:root:root:root:root body .pw-head h2 { margin:.9rem 0 .2rem!important;font-size:clamp(1.25rem, 1rem + 1vw, 1.7rem);font-weight:800;letter-spacing:.02em;color:#fff!important;animation:akfade .35s ease both; }
html:root:root:root:root body .pw-head p { margin:0!important;color:#c9d8ff!important;font-size:.98rem; }
html:root:root:root:root body #pw-stage { flex:1 1 auto!important;display:flex!important;flex-wrap:nowrap!important;align-items:center!important;justify-content:center!important;gap:clamp(.8rem, 3vw, 2.5rem)!important; }
html:root:root:root:root body #pw-mascot { flex:0 0 clamp(6.5rem, 14vw, 10rem)!important;width:clamp(6.5rem, 14vw, 10rem)!important;min-width:0!important; }
html:root:root:root:root body #pw-fields { flex:0 1 34rem!important;max-width:34rem;gap:.6rem!important; }
html:root:root:root:root body .pw-step { gap:.7rem!important;animation:akfade .35s ease both; }
/* libellés et champs dans la carte bleue */
html:root:root:root:root body #plan-wizard label > span, html:root:root:root:root body #plan-wizard [data-testid="block-info"], html:root:root:root:root body #plan-wizard legend,
html:root:root:root:root body #plan-wizard .block-title, html:root:root:root:root body #plan-wizard .label-wrap span { color:#e3ecff!important;-webkit-text-fill-color:#e3ecff!important;font-weight:700!important;background:transparent!important; }
html:root:root:root:root body #plan-wizard .info, html:root:root:root:root body #plan-wizard [data-testid="block-info"] + div, html:root:root:root:root body #plan-wizard .gr-info { color:#b6c7f0!important;-webkit-text-fill-color:#b6c7f0!important; }
html:root:root:root:root body #plan-wizard input[type="text"], html:root:root:root:root body #plan-wizard input[type="number"], html:root:root:root:root body #plan-wizard textarea,
html:root:root:root:root body #plan-wizard .ak-datepick, html:root:root:root:root body #plan-wizard .wrap-inner, html:root:root:root:root body #plan-wizard .secondary-wrap {
  background:rgba(8,22,64,.55)!important;border:1px solid rgba(150,190,255,.35)!important;border-radius:.75rem!important;color:#fff!important;-webkit-text-fill-color:#fff!important;color-scheme:dark; }
html:root:root:root:root body #plan-wizard .secondary-wrap input, html:root:root:root:root body #plan-wizard .wrap-inner input { background:transparent!important;border:0!important; }
html:root:root:root:root body #plan-wizard input::placeholder, html:root:root:root:root body #plan-wizard textarea::placeholder { color:#8fa6d8!important;-webkit-text-fill-color:#8fa6d8!important; }
html:root:root:root:root body #plan-wizard input:focus, html:root:root:root:root body #plan-wizard textarea:focus, html:root:root:root:root body #plan-wizard .ak-datepick:focus { border-color:#9cc6ff!important;box-shadow:0 0 0 3px rgba(124,196,255,.22)!important; }
html:root:root:root:root body #plan-wizard input[type="range"] { accent-color:#7cc4ff; }
/* Heure de début : sélecteur natif, libellé sur une ligne, colonne assez large pour « 18:00 » et l'icône */
html:root:root:root:root body #pw-start { flex:0 0 11rem!important;min-width:11rem!important; }
html:root:root:root:root body #pw-start label > span, html:root:root:root:root body #pw-start .label-wrap span { white-space:nowrap!important; }
html:root:root:root:root body.m-mode #plan-wizard .row:has(> #plan-minutes), html:root:root:root:root body.ak-narrow #plan-wizard .row:has(> #plan-minutes) { flex-wrap:wrap!important; }
html:root:root:root:root body.m-mode #plan-wizard .row:has(> #plan-minutes) > *, html:root:root:root:root body.ak-narrow #plan-wizard .row:has(> #plan-minutes) > * { flex:1 1 100%!important;min-width:0!important;max-width:100%!important; }
html:root:root:root:root body.m-mode #pw-start, html:root:root:root:root body.ak-narrow #pw-start { flex:1 1 100%!important; }
/* téléphone : Précédent, Annuler et Suivant sur une seule ligne */
html:root:root:root:root body.m-mode #pw-nav { flex-wrap:nowrap!important;gap:.5rem!important; }
html:root:root:root:root body.m-mode #pw-nav button { min-width:0!important;flex:1 1 auto!important;padding:.65rem .7rem!important;font-size:.95rem!important; }
html:root:root:root:root body.m-mode #pw-nav #pw-prev { flex:0 1 auto!important;padding-left:.2rem!important;padding-right:.2rem!important; }
html:root:root:root:root body #plan-wizard .ak-pick-time { font-size:1.05rem;font-variant-numeric:tabular-nums;letter-spacing:.02em;cursor:pointer; }
html:root:root:root:root body #plan-wizard .ak-datepick::-webkit-calendar-picker-indicator { cursor:pointer;opacity:.85; }
html:root:root:root:root body #plan-wizard .token { background:rgba(79,123,255,.35)!important;color:#fff!important;border:1px solid rgba(160,195,255,.4)!important;border-radius:99px!important; }
html:root:root:root:root body #plan-wizard .token * { color:#fff!important;-webkit-text-fill-color:#fff!important; }
html:root:root:root:root body #plan-wizard .token-remove, html:root:root:root:root body #plan-wizard .remove-all { background:rgba(255,255,255,.16)!important;border:0!important;color:#fff!important;box-shadow:none!important; }
html:root:root:root:root body #plan-wizard ul.options { background:#13306f!important;border:1px solid rgba(150,190,255,.35)!important;border-radius:.75rem!important; }
html:root:root:root:root body #plan-wizard ul.options li { color:#eef4ff!important;background:transparent!important; } html:root:root:root:root body #plan-wizard ul.options li:hover, html:root:root:root:root body #plan-wizard ul.options li.active { background:rgba(124,196,255,.2)!important; }
/* cases à cocher et boutons radio en tuiles */
html:root:root:root:root body #plan-wizard .pw-step .wrap { display:flex!important;flex-wrap:wrap!important;gap:.5rem!important; }
html:root:root:root:root body #plan-wizard .pw-step .wrap > label { margin:0!important;display:flex!important;align-items:center!important;gap:.55rem!important;cursor:pointer;
  background:rgba(8,22,64,.45)!important;border:1px solid rgba(150,190,255,.3)!important;border-radius:.75rem!important;padding:.6rem .85rem!important;color:#eef4ff!important;
  transition:background .15s,border-color .15s,transform .15s; }
html:root:root:root:root body #plan-wizard .pw-step .wrap > label span { color:#eef4ff!important;-webkit-text-fill-color:#eef4ff!important; }
html:root:root:root:root body #plan-wizard .pw-step .wrap > label:hover { border-color:#9cc6ff!important;transform:translateY(-1px); }
html:root:root:root:root body #plan-wizard .pw-step .wrap > label:has(input:checked) { background:rgba(79,123,255,.45)!important;border-color:#9cc6ff!important; }
html:root:root:root:root body #plan-wizard .pw-step input[type="checkbox"], html:root:root:root:root body #plan-wizard .pw-step input[type="radio"] { accent-color:#4f7bff;width:1.05rem;height:1.05rem;flex:none;
  background-color:rgba(8,22,64,.6)!important;border:1.5px solid rgba(170,200,255,.6)!important; }
html:root:root:root:root body #plan-wizard .pw-step input[type="checkbox"]:checked, html:root:root:root:root body #plan-wizard .pw-step input[type="radio"]:checked { background-color:#7cc4ff!important;border-color:#7cc4ff!important; }
html:root:root:root:root body #plan-wizard #pw-intensity .wrap > label input:checked { background:#fff!important; }
html:root:root:root:root body #pw-acts .wrap, html:root:root:root:root body #pw-ai .wrap { flex-direction:column!important;flex-wrap:nowrap!important; }
/* étape 1 : contexte en pastilles */
html:root:root:root:root body #plan-wizard #pw-context .wrap { flex-wrap:nowrap!important;justify-content:center!important;background:rgba(8,22,64,.45)!important;border:1px solid rgba(150,190,255,.3)!important;border-radius:99px!important;padding:.3rem!important;gap:.25rem!important; }
html:root:root:root:root body #plan-wizard #pw-context .wrap > label { border-radius:99px!important;border-color:transparent!important;background:transparent!important;padding:.55rem .9rem!important;white-space:nowrap; }
html:root:root:root:root body.m-mode #plan-wizard #pw-context .wrap, html:root:root:root:root body.ak-narrow #plan-wizard #pw-context .wrap { flex-wrap:wrap!important;border-radius:1.2rem!important; }
html:root:root:root:root body #plan-wizard #pw-context .wrap > label:has(input:checked) { background:linear-gradient(135deg,#3f7bff,#2f5fe0)!important;box-shadow:0 6px 18px rgba(47,95,224,.45); }
/* étape 4 : intensité comme un curseur à trois crans, jours de repos en pastilles */
html:root:root:root:root body #plan-wizard #pw-intensity .wrap { position:relative;display:grid!important;grid-template-columns:repeat(3,1fr);gap:0!important;padding-top:1.4rem!important; }
html:root:root:root:root body #pw-intensity .wrap::before { content:"";position:absolute;left:16.6%;right:16.6%;top:.55rem;height:.3rem;border-radius:99px;background:rgba(255,255,255,.2); }
html:root:root:root:root body #plan-wizard #pw-intensity .wrap > label { flex-direction:column!important;justify-content:center;background:transparent!important;border:0!important;padding:.2rem!important;font-weight:600; }
html:root:root:root:root body #pw-intensity .wrap > label input { position:absolute;top:.25rem;width:1rem!important;height:1rem!important;appearance:none;-webkit-appearance:none;border-radius:50%;
  background:#9db8ef;border:2px solid #16357a;margin:0!important; }
html:root:root:root:root body #pw-intensity .wrap > label { position:relative; }
html:root:root:root:root body #pw-intensity .wrap > label input { top:-1.2rem; }
html:root:root:root:root body #pw-intensity .wrap > label:has(input:checked) input { background:#fff;width:1.35rem!important;height:1.35rem!important;top:-1.38rem;box-shadow:0 0 0 4px rgba(124,196,255,.45); }
html:root:root:root:root body #pw-intensity .wrap > label:has(input:checked) span { color:#fff!important;font-weight:800; }
html:root:root:root:root body #plan-wizard #pw-rest .wrap > label { padding:.45rem .7rem!important;min-width:3.3rem;justify-content:center; }
.pw-ai-help { font-size:.82rem;line-height:1.45;color:#c9d8ff;border-left:3px solid rgba(124,196,255,.5);padding-left:.75rem; }
html:root:root:root:root body .pw-ai-help p { margin:.15rem 0!important;color:#c9d8ff!important; } html:root:root:root:root body .pw-ai-help b { color:#fff!important; }
html:root:root:root:root body #plan-wizard #plan-status p { color:#ffd0d0!important;-webkit-text-fill-color:#ffd0d0!important;margin:0!important;font-weight:600; }
/* boutons de navigation */
html:root:root:root:root body #pw-nav { display:flex!important;flex-wrap:wrap!important;justify-content:flex-end!important;align-items:center!important;gap:.7rem!important;flex:none!important; }
html:root:root:root:root body #pw-nav > button { flex:0 0 auto!important;width:auto!important;min-height:2.85rem!important;border-radius:99px!important;padding:0 1.6rem!important;font-weight:700!important;font-size:.98rem!important; }
html:root:root:root:root body #plan-wizard button#pw-prev { margin-right:auto!important;background:transparent!important;border:0!important;color:#fff!important;-webkit-text-fill-color:#fff!important;box-shadow:none!important;padding:0 .6rem!important; }
html:root:root:root:root body #plan-wizard button#pw-cancel { background:#dbe7ff!important;border:0!important;color:#163a86!important;-webkit-text-fill-color:#163a86!important;box-shadow:0 4px 14px rgba(0,0,0,.18)!important; }
html:root:root:root:root body #plan-wizard button#pw-next { background:linear-gradient(135deg,#3f7bff,#2f5fe0)!important;border:0!important;color:#fff!important;-webkit-text-fill-color:#fff!important;box-shadow:0 6px 18px rgba(47,95,224,.45)!important; }
html:root:root:root:root body #plan-wizard button#pw-gen { background:#0b1a44!important;border:1.5px solid #6fb2ff!important;color:#fff!important;-webkit-text-fill-color:#fff!important;letter-spacing:.04em;
  text-transform:uppercase;box-shadow:0 0 0 3px rgba(111,178,255,.18), 0 0 22px rgba(111,178,255,.55)!important;animation:pwglow 2.2s ease-in-out infinite; }
@keyframes pwglow { 50% { box-shadow:0 0 0 5px rgba(111,178,255,.12), 0 0 32px rgba(111,178,255,.75); } }
html:root:root:root:root body #plan-wizard button:hover { filter:brightness(1.08); }
html:root:root:root:root body #pw-mascot:has(.mood-celebrate) { order:2; }
.pw-mascot { filter:drop-shadow(0 12px 20px rgba(5,15,50,.45)); }
html:root:root:root:root body.m-mode #pw-stage, html:root:root:root:root body.ak-narrow #pw-stage { flex-direction:column!important; }
html:root:root:root:root body.m-mode #pw-mascot, html:root:root:root:root body.ak-narrow #pw-mascot { flex-basis:5.5rem!important;width:5.5rem!important; }
html:root:root:root:root body.m-mode #pw-fields, html:root:root:root:root body.ak-narrow #pw-fields { flex-basis:auto!important;width:100%!important; }
html:root:root:root:root body.m-mode #plan-wizard { padding:.9rem .9rem 1rem!important; } html:root:root:root:root body.m-mode .pw-bar { margin-right:0; } html:root:root:root:root body.m-mode .pw-brand { display:none; }
html:root:root:root:root body.m-mode #pw-nav > button { padding:0 1rem!important;flex:1 1 auto!important; } html:root:root:root:root body.m-mode #plan-wizard button#pw-prev { flex:0 0 auto!important; }

/* ===== Plan de révision : emploi du temps intelligent ===== */
html:root:root:root:root body #plan-edt, html:root:root:root:root body #plan-edt > .block, html:root:root:root:root body #plan-edt .html-container { padding:0!important;background:transparent!important;border:0!important;box-shadow:none!important; }
.edt { display:flex;flex-direction:column;gap:.6rem;height:calc(var(--body-h, 600px) - 2rem);min-height:26rem; }
.edt-head { display:grid;grid-template-columns:minmax(0,1.5fr) repeat(3,minmax(0,1fr)) auto;gap:.6rem;align-items:center;border:1px solid var(--ak-line);border-radius:1rem;background:var(--ak-card);padding:.65rem .9rem; }
.edt-id { display:flex;flex-direction:column;min-width:0; } .edt-id b { font-size:1rem;color:var(--ak-text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis; } .edt-id small { color:var(--ak-muted);font-size:.76rem; }
.edt-ctx { align-self:flex-start;font-size:.66rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;padding:.12rem .55rem;border-radius:99px;background:rgba(79,109,245,.14);color:#6b84f7;margin-bottom:.15rem; }
.edt-kpi { display:flex;flex-direction:column;min-width:0;padding-left:.7rem;border-left:1px solid var(--ak-line); } .edt-kpi b { font-size:.92rem;color:var(--ak-text); } .edt-kpi span { font-size:.72rem;color:var(--ak-muted); }
.edt-pct { flex-direction:row;align-items:center;gap:.5rem; }
.edt-acts { display:flex;gap:.4rem; } html:root:root:root:root body #plan-edt .edt-acts .pl-btn { flex:0 0 auto;padding:.5rem .85rem;font-size:.78rem; }
.edt-alerts { display:grid;gap:.35rem; } .edt-alert { border-radius:.75rem;padding:.5rem .8rem;font-size:.8rem;background:rgba(224,164,58,.15);border:1px solid rgba(224,164,58,.4);color:var(--ak-text); }
.edt-bar { display:flex;align-items:center;gap:.5rem;flex-wrap:wrap; } .edt-wl { font-size:.92rem;color:var(--ak-text);min-width:14rem;text-align:center; }
html:root:root:root:root body #plan-edt button.edt-nav { width:2.1rem!important;height:2.1rem!important;min-height:0!important;border-radius:50%!important;padding:0!important;font-size:1.3rem!important;line-height:1!important;
  border:1px solid var(--ak-line)!important;background:var(--ak-card)!important;color:var(--ak-text)!important;-webkit-text-fill-color:var(--ak-text)!important;cursor:pointer; }
html:root:root:root:root body #plan-edt button.edt-nav:disabled { opacity:.35;cursor:default; }
.edt-legend { margin-left:auto;display:flex;flex-wrap:wrap;gap:.35rem; }
.edt-legend > span { font-size:.68rem;color:var(--ak-muted);padding:.12rem .5rem;border-radius:99px;border:1px solid var(--ak-line);display:inline-flex;align-items:center;gap:.25rem; }
.edt-legend > span { border-color:color-mix(in srgb, var(--k) 45%, var(--ak-line)); } .edt-legend .ak-ico { color:var(--k);margin-right:.15rem; }
.eb-t .ak-ico { margin-right:.2rem;width:.85rem;height:.85rem; }
.edt-weeks { position:relative;flex:1 1 auto;min-height:0;border:1px solid var(--ak-line);border-radius:1rem;background:var(--ak-card);overflow:hidden;touch-action:pan-y; }
.edt-week { display:none;height:100%;grid-template-columns:3rem repeat(7,minmax(0,1fr)); }
.edt-week.on { display:grid;animation:akfade .3s ease both; }
.edt-hours { position:relative;margin:2.8rem 0 .4rem; }
.edt-hours span { position:absolute;right:.35rem;transform:translateY(-50%);font-size:.62rem;color:var(--ak-muted);font-variant-numeric:tabular-nums; }
.edt-hours span:first-child { transform:none; }
.edt-col { display:flex;flex-direction:column;min-width:0;border-left:1px solid var(--ak-line); }
html:root:root:root:root body #plan-edt .edt-dh { height:2.8rem!important;min-height:0!important;flex:none;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0;padding:0!important;
  border:0!important;border-bottom:1px solid var(--ak-line)!important;border-radius:0!important;background:transparent!important;color:var(--ak-muted)!important;-webkit-text-fill-color:var(--ak-muted)!important;
  font-size:.7rem!important;font-weight:600!important;cursor:pointer;box-shadow:none!important; }
html:root:root:root:root body #plan-edt div.edt-dh { cursor:default; }
html:root:root:root:root body #plan-edt .edt-dh b { font-size:.95rem;color:var(--ak-text);-webkit-text-fill-color:var(--ak-text);line-height:1.2; }
html:root:root:root:root body #plan-edt .edt-dh small { font-size:.58rem; }
html:root:root:root:root body #plan-edt button.edt-dh:hover { background:var(--ak-soft)!important; }
html:root:root:root:root body #plan-edt .edt-col.today .edt-dh b { background:#4f6df5;color:#fff;-webkit-text-fill-color:#fff;border-radius:99px;padding:0 .4rem; }
html:root:root:root:root body #plan-edt .edt-col.over .edt-dh small { color:#e5484d;-webkit-text-fill-color:#e5484d;font-weight:800; }
.edt-body { position:relative;flex:1 1 auto;margin:0 .2rem .4rem;
  background:repeating-linear-gradient(to bottom, transparent 0, transparent calc(var(--tick, 25%) - 1px), color-mix(in srgb, var(--ak-line) 70%, transparent) calc(var(--tick, 25%) - 1px), color-mix(in srgb, var(--ak-line) 70%, transparent) var(--tick, 25%)); }
.edt-col.today .edt-body { background-color:rgba(79,109,245,.05); }
.edt-col.off { opacity:.4; } .edt-col.off .edt-body { background:none; }
.edt-col.rest .edt-body { background:repeating-linear-gradient(45deg, transparent 0 7px, var(--ak-soft) 7px 14px);border-radius:.6rem;display:flex;align-items:center;justify-content:center; }
.edt-rest { font-size:.7rem;font-weight:700;color:var(--ak-muted);writing-mode:vertical-rl;letter-spacing:.2em;text-transform:uppercase; }
html:root:root:root:root body #plan-edt .edt-blk { position:absolute;left:0;right:0;min-height:1.3rem!important;margin:0!important;padding:.18rem .35rem!important;border:0!important;border-left:3px solid var(--k)!important;
  border-radius:.5rem!important;background:color-mix(in srgb, var(--k) 20%, var(--ak-card))!important;color:var(--ak-text)!important;-webkit-text-fill-color:var(--ak-text)!important;
  display:flex;flex-direction:column;align-items:stretch;justify-content:flex-start;gap:.02rem;overflow:hidden;text-align:left;cursor:pointer;font-size:.66rem!important;font-weight:500!important;line-height:1.2;
  box-shadow:0 1px 0 rgba(0,0,0,.04)!important;transition:transform .15s ease, box-shadow .15s ease;animation:akfade .35s ease both; }
html:root:root:root:root body #plan-edt .edt-blk:hover { transform:scale(1.04);z-index:3;box-shadow:0 8px 20px rgba(0,0,0,.18)!important; }
.edt-blk > span { display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
.eb-t { font-weight:800;font-size:.68rem; } .eb-s { opacity:.85; } .eb-h { opacity:.65;font-variant-numeric:tabular-nums; }
.edt-blk.sm .eb-s, .edt-blk.sm .eb-h { display:none; }
html:root:root:root:root body #plan-edt .edt-blk.done { opacity:.55; } .edt-blk.done .eb-t::after { content:" ✓"; }
html:root:root:root:root body #plan-edt .edt-exam { position:absolute;inset:0;margin:0!important;border:1.5px dashed #e0a43a!important;border-radius:.6rem!important;background:rgba(224,164,58,.14)!important;
  color:var(--ak-text)!important;-webkit-text-fill-color:var(--ak-text)!important;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.3rem;font-size:1.3rem!important;cursor:pointer;box-shadow:none!important; }
.edt-exam span { font-size:.72rem;font-weight:800; }
.edt-note { margin:0!important;font-size:.72rem;color:var(--ak-muted)!important; }
html:root:root:root:root body.ak-short .edt, html:root:root:root:root body.m-mode .edt { height:auto; } html:root:root:root:root body.ak-short .edt-weeks, html:root:root:root:root body.m-mode .edt-weeks { height:30rem;flex:none; }
html:root:root:root:root body.ak-narrow .edt-head, html:root:root:root:root body.m-mode .edt-head { grid-template-columns:repeat(2,minmax(0,1fr)); }
html:root:root:root:root body.ak-narrow .edt-id, html:root:root:root:root body.m-mode .edt-id, html:root:root:root:root body.ak-narrow .edt-acts, html:root:root:root:root body.m-mode .edt-acts { grid-column:1 / -1; }
html:root:root:root:root body.ak-narrow .edt-kpi:nth-child(2), html:root:root:root:root body.m-mode .edt-kpi:nth-child(2) { padding-left:0;border-left:0; }
html:root:root:root:root body.ak-narrow .edt-legend, html:root:root:root:root body.m-mode .edt-legend { display:none; }
html:root:root:root:root body.m-mode .edt-week { grid-template-columns:2.3rem repeat(7,minmax(0,1fr)); }
html:root:root:root:root body.m-mode #plan-edt .edt-blk { padding:.12rem .15rem!important; } html:root:root:root:root body.m-mode .edt-blk .eb-s, html:root:root:root:root body.m-mode .edt-blk .eb-h { display:none; }
html:root:root:root:root body.m-mode .edt-wl { min-width:0;flex:1;font-size:.8rem; }
html:root:root:root:root body.m-mode .edt-blk .eb-t { font-size:0!important;text-align:center; } html:root:root:root:root body.m-mode .edt-blk .eb-t .ak-ico { margin:0;width:.95rem;height:.95rem; }
/* fenêtre de détail (séance, jour, examen) */
body > .edt-modal { position:fixed;inset:0;z-index:700;display:flex;align-items:center;justify-content:center;padding:1rem;background:rgba(8,12,28,.55);backdrop-filter:blur(4px);animation:akfade .18s ease both; }
.edt-mbox { position:relative;width:min(30rem,100%);max-height:86vh;overflow:auto;border-radius:1.2rem;background:var(--ak-card, #fff);border:1px solid var(--ak-line, #e4e9f2);
  box-shadow:0 30px 80px rgba(0,0,0,.35);padding:1.2rem 1.3rem;color:var(--ak-text, #182033);animation:akrise .25s ease both; }
.edt-mclose { position:absolute;top:.7rem;right:.7rem;width:2rem;height:2rem;border-radius:50%;border:1px solid var(--ak-line, #e4e9f2);background:var(--ak-soft, #eef2ff);color:var(--ak-text, #182033);cursor:pointer; }
.ed-card { display:flex;flex-direction:column;gap:.8rem; }
.ed-top { display:flex;gap:.8rem;align-items:center;padding-right:2rem; }
.ed-ico { width:2.8rem;height:2.8rem;border-radius:.8rem;flex:none;display:flex;align-items:center;justify-content:center;font-size:1.3rem;background:color-mix(in srgb, var(--k) 22%, transparent);color:var(--k); }
.ed-ico .ak-ico { width:1.3rem;height:1.3rem;margin:0; }
.ed-top small { font-size:.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--k); }
.ed-top h3 { margin:.1rem 0 0;font-size:1.05rem;line-height:1.3;color:var(--ak-text, #182033); }
.ed-grid { display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.5rem; }
.ed-grid > div { background:var(--ak-soft, #eef2ff);border-radius:.7rem;padding:.5rem .65rem;min-width:0; }
.ed-grid span { display:block;font-size:.68rem;color:var(--ak-muted, #69758b); } .ed-grid b { font-size:.85rem;color:var(--ak-text, #182033); }
.ed-grid i { font-style:normal; }
.ed-why { border-left:3px solid var(--k);padding:.1rem 0 .1rem .75rem;font-size:.85rem; } .ed-why b { color:var(--ak-text, #182033); }
.ed-why p { margin:.2rem 0 0;color:var(--ak-muted, #69758b);line-height:1.45; } .ed-why.warn { border-color:#e5484d; }
.ed-pages b { font-size:.82rem; } .ed-pages .plan-pages { margin-top:.35rem; }
.ed-day { list-style:none;margin:0;padding:0;display:grid;gap:.4rem; }
.ed-day li { display:grid;grid-template-columns:auto 1fr;gap:.1rem .6rem;padding:.5rem .65rem;border-radius:.7rem;background:var(--ak-soft, #eef2ff);border-left:3px solid var(--k); }
.ed-day em { font-style:normal;font-size:.75rem;font-weight:700;color:var(--ak-muted, #69758b);font-variant-numeric:tabular-nums;grid-row:span 2; }
.ed-day span { font-size:.84rem;color:var(--ak-text, #182033); } .ed-day small { font-size:.72rem;color:var(--ak-muted, #69758b); }
.ed-acts:empty { display:none; } .ed-acts .pl-go { width:100%;padding:.7rem!important;font-size:.88rem!important;border-radius:.8rem!important; }

/* ===== Accueil : fiche du guide dans une fenêtre centrée + animations d'entrée qui laissent agir le survol ===== */
html:root:root:root:root body .rv, html:root:root:root:root body .mini-stat, html:root:root:root:root body .hero { animation-fill-mode:backwards!important; }
html:root:root:root:root body details.guide-card { transition:transform .22s cubic-bezier(.2,.8,.2,1), box-shadow .22s ease, border-color .22s ease!important; will-change:transform; }
html:root:root:root:root body details.guide-card:hover { transform:translateY(-3px); }
html:root:root:root:root body details.guide-card:active { transform:translateY(-1px) scale(.99); }
html:root:root:root:root body .mini-stat { transition:transform .22s cubic-bezier(.2,.8,.2,1), box-shadow .22s ease; }
body > .gc-modal { position:fixed;inset:0;z-index:650;display:flex;align-items:center;justify-content:center;padding:1rem;
  background:rgba(8,12,28,.5);backdrop-filter:blur(3px);animation:gcfade .2s ease both; }
body > .gc-modal.out { animation:gcfadeout .18s ease both; }
body > .gc-modal .gc-mbox { position:relative;width:min(34rem,100%);max-height:86vh;overflow:auto;border-radius:1.25rem;padding:1.3rem 1.4rem 1.2rem;
  background:var(--ak-card, #fff);border:1px solid var(--ak-line, #e4e9f2);color:var(--ak-text, #182033);box-shadow:0 30px 80px rgba(0,0,0,.35);
  animation:gcpop .28s cubic-bezier(.2,.9,.3,1.15) both; }
body > .gc-modal.out .gc-mbox { animation:gcpopout .18s ease both; }
@keyframes gcfade { from { opacity:0; } to { opacity:1; } }
@keyframes gcfadeout { to { opacity:0; } }
@keyframes gcpop { from { opacity:0;transform:translateY(14px) scale(.96); } to { opacity:1;transform:none; } }
@keyframes gcpopout { to { opacity:0;transform:translateY(8px) scale(.97); } }
.gc-mhead { display:flex;align-items:center;gap:.75rem;margin:0 2.4rem .9rem 0; }
.gc-mhead .gc-ico { width:2.6rem;height:2.6rem;border-radius:.8rem;flex:none; }
.gc-mhead .gc-detail-title { font-size:1.15rem;font-weight:800;color:var(--ak-text, #182033); }
body > .gc-modal .gc-close { position:absolute;top:.8rem;right:.8rem;width:2rem;height:2rem;border-radius:50%;border:1px solid var(--ak-line, #e4e9f2);
  background:var(--ak-soft, #eef2ff);color:var(--ak-text, #182033);cursor:pointer;font-size:.95rem;line-height:1;transition:transform .15s ease; }
body > .gc-modal .gc-close:hover { transform:rotate(90deg); }
.gc-mbody ol { margin:0 0 .8rem;padding-left:1.25rem;color:var(--ak-text, #182033);line-height:1.55;font-size:.88rem; }
.gc-mbody li { margin:.3rem 0;animation:gcpop .35s ease both; }
.gc-mbody li:nth-child(2) { animation-delay:.06s; } .gc-mbody li:nth-child(3) { animation-delay:.12s; }
.gc-mbody .gc-tip { margin:.6rem 0 .9rem; } .gc-mbody .ak-go { animation:gcpop .35s ease .18s both; }

/* ===== Progression détaillée (carrousel) : couleurs du thème choisi ===== */
html:root:root:root:root body > .gp-modal { background:radial-gradient(40rem circle at 30% 20%, rgba(79,109,245,.16), transparent 60%),
  radial-gradient(36rem circle at 75% 30%, rgba(142,108,244,.13), transparent 60%), color-mix(in srgb, var(--ak-bg) 82%, transparent)!important; }
html:root:root:root:root body > .gp-modal .gp-card { background:var(--ak-card)!important;border-color:var(--ak-line)!important;color:var(--ak-text)!important;
  box-shadow:0 30px 70px color-mix(in srgb, var(--ak-text) 18%, transparent)!important; }
html:root:root:root:root body > .gp-modal .gp-card h3, html:root:root:root:root body > .gp-modal .gp-pct, html:root:root:root:root body > .gp-modal .gp-bar em { color:var(--ak-text)!important; }
html:root:root:root:root body > .gp-modal .gp-date, html:root:root:root:root body > .gp-modal .gp-sub, html:root:root:root:root body > .gp-modal .gp-bar { color:var(--ak-muted)!important; }
html:root:root:root:root body > .gp-modal .gp-glow { opacity:.22; }
html:root:root:root:root body > .gp-modal .gp-track-c { stroke:var(--ak-soft)!important; }
html:root:root:root:root body > .gp-modal .gp-bar > div, html:root:root:root:root body > .gp-modal .gp-chip { background:var(--ak-soft)!important; }
html:root:root:root:root body > .gp-modal .gp-chip { color:var(--ak-text)!important; }
html:root:root:root:root body > .gp-modal .gp-dot { background:var(--ak-soft)!important;border-color:var(--ak-card)!important;color:var(--ak-muted)!important; }
html:root:root:root:root body > .gp-modal .gp-dot.on { background:var(--c)!important;color:#fff!important; }
html:root:root:root:root body > .gp-modal .gp-card .gp-foot { border-top-color:var(--ak-line)!important; }
html:root:root:root:root body > .gp-modal .gp-nav, html:root:root:root:root body > .gp-modal .gp-close { background:var(--ak-card)!important;border-color:var(--ak-line)!important;color:var(--ak-text)!important;
  box-shadow:0 8px 22px color-mix(in srgb, var(--ak-text) 14%, transparent); }
html:root:root:root:root body > .gp-modal .gp-nav:hover, html:root:root:root:root body > .gp-modal .gp-close:hover { background:var(--ak-soft)!important; }
html:root:root:root:root body > .gp-modal .gp-pager i { background:color-mix(in srgb, var(--ak-text) 25%, transparent)!important; }
html:root:root:root:root body > .gp-modal .gp-pager i.on { background:#4f6df5!important; }

/* ===== Assistant : conversation encadrée, centrée et sans boutons parasites ===== */
html:root:root:root:root body #assistant-chatbot .top-panel, html:root:root:root:root body #assistant-chatbot .message-buttons,
html:root:root:root:root body #assistant-chatbot .message-buttons-left, html:root:root:root:root body #assistant-chatbot .message-buttons-right { display:none!important; }
html:root:root:root:root body.in-app #assistant-chatbot, html:root:root:root:root body.in-app.akori-dark #assistant-chatbot {
  border:1px solid var(--ak-line)!important;border-radius:1rem!important;background:var(--ak-bg)!important; }
html:root:root:root:root body.in-app #assistant-chatbot > .wrapper, html:root:root:root:root body.in-app #assistant-chatbot .bubble-wrap { background:transparent!important; }
html:root:root:root:root body #assistant-chatbot .message-wrap { max-width:58rem;margin:0 auto!important;padding:1.1rem 1.2rem 1.4rem!important; }
html:root:root:root:root body #scope-row, html:root:root:root:root body #assistant-bar { width:100%!important;max-width:58rem;margin-left:auto!important;margin-right:auto!important; }
html:root:root:root:root body.m-mode #assistant-chatbot .message-wrap { padding:.7rem .6rem 1rem!important; }
/* ===== Mascotte AKORI (expressions animées) ===== */
.mascot { display:block;width:100%;height:auto;overflow:visible; }
.mascot .ms-body { transform-box:view-box;transform-origin:100px 190px;animation:msfloat 3.6s ease-in-out infinite; }
.mascot .ms-eyes { transform-box:view-box;transform-origin:100px 75px;animation:msblink 5s ease-in-out infinite; }
.mascot .ms-arm { fill:none;stroke:#4a86ff;stroke-width:10;stroke-linecap:round; }
.mascot .ms-hand { fill:#5f98ff;stroke:#3a6fe8;stroke-width:1.5; }
.mascot .ms-sym { font:800 27px Inter,system-ui,sans-serif;fill:#fff; }
.mascot .ms-bubble { transform-box:view-box;transform-origin:150px 56px;animation:mspop .55s cubic-bezier(.2,.9,.3,1.4) .15s both, msbob 2.8s ease-in-out .8s infinite; }
.mascot .ms-wave { transform-box:view-box;transform-origin:134px 128px;animation:mswave 1.3s ease-in-out infinite; }
.mascot .ms-spark { transform-box:fill-box;transform-origin:center;animation:mstwinkle 1.6s ease-in-out infinite; }
.mascot .ms-drop { animation:msdrop 2.4s ease-in infinite; }
.mascot.mood-celebrate .ms-body { animation:msjump 1.1s cubic-bezier(.3,.7,.4,1) infinite; }
.mascot.mood-sad .ms-body { animation-duration:5.5s; }
.mascot.mood-sleep .ms-body { animation:msbreath 4s ease-in-out infinite; }
.mascot.mood-determined .ms-arms { transform-box:view-box;transform-origin:100px 126px;animation:msflex 1.6s ease-in-out infinite; }
@keyframes msfloat { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-5px); } }
@keyframes msjump { 0%,100% { transform:translateY(0) scale(1,1); } 15% { transform:translateY(0) scale(1.05,.94); } 45% { transform:translateY(-14px) scale(.97,1.04); } 70% { transform:translateY(0) scale(1.03,.97); } }
@keyframes msbreath { 0%,100% { transform:scale(1); } 50% { transform:scale(1.025,.98); } }
@keyframes msblink { 0%,90%,100% { transform:scaleY(1); } 94% { transform:scaleY(.08); } }
@keyframes mspop { from { opacity:0;transform:scale(.3); } to { opacity:1;transform:scale(1); } }
@keyframes msbob { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-3px); } }
@keyframes mswave { 0%,100% { transform:rotate(0); } 50% { transform:rotate(-16deg); } }
@keyframes msflex { 0%,100% { transform:scale(1); } 50% { transform:scale(1.04); } }
@keyframes mstwinkle { 0%,100% { opacity:.25;transform:scale(.6) rotate(0); } 50% { opacity:1;transform:scale(1.15) rotate(25deg); } }
@keyframes msdrop { 0% { opacity:0;transform:translateY(-4px); } 30% { opacity:1; } 100% { opacity:0;transform:translateY(10px); } }
@media (prefers-reduced-motion: reduce) { .mascot * { animation:none!important; } }
/* bloc mascotte + message */
.ms-note { display:flex!important;flex-direction:column;align-items:center;gap:.4rem;text-align:center; }
.ms-note .ms-box { width:clamp(5.5rem, 12vw, 8rem); }
.ms-note .ms-msg { display:flex;flex-direction:column;gap:.3rem;max-width:30rem; }
.ms-note .ms-msg b { font-size:1.02rem;color:var(--ak-text); } .ms-note .ms-msg span { color:var(--ak-muted);line-height:1.5; }
.ms-row { display:flex;align-items:center;gap:1rem; } .ms-row .ms-box { flex:none;width:5.5rem; }
/* ===== Paramètres et Profil : pas de sélecteur « Cours actif » ===== */
html:root:root:root:root body.no-docsel #topbar > .form:has(#doc-select), html:root:root:root:root body.no-docsel #topbar #doc-select { display:none!important; }

/* ===== Mascotte dans les résultats (flashcards, quiz, colle orale, progression, accueil) ===== */
.ms-result { display:flex;align-items:center;justify-content:center;gap:1rem;margin:0 auto .6rem;max-width:34rem;text-align:left; }
.ms-result .ms-box { flex:none;width:6.5rem; }
.ms-result b { display:block;font-size:1.15rem;color:var(--ak-text); } .ms-result span { display:block;color:var(--ak-muted);font-size:.9rem;line-height:1.45;margin-top:.2rem; }
.oral-ms { flex:none;width:5rem; }
.gp-ms { flex:none;width:clamp(3.6rem, 7vw, 5rem); }
/* vue d'ensemble : même carte que la vue détaillée (anneau, titre, sous-titre), le texte garde toute la place */
html:root:root:root:root body .progress-hero.gp-hero { display:grid!important;grid-template-columns:auto minmax(0,1fr) auto;align-items:center;gap:1rem!important; }
html:root:root:root:root body .progress-hero .progress-ring { flex:none;width:4.6rem;height:4.6rem; }
html:root:root:root:root body .gp-copy, html:root:root:root:root body .progress-hero > div { min-width:0;overflow-wrap:normal!important;word-break:normal!important; }
html:root:root:root:root body .gp-copy .progress-title { font-size:1.05rem;line-height:1.3; }
html:root:root:root:root body .gp-copy .progress-sub { line-height:1.45; }
html:root:root:root:root body .gp-copy .gp-hint { margin-top:.35rem;font-size:.78rem; }
html:root:root:root:root body.m-mode .progress-hero.gp-hero { gap:.7rem!important;padding:1rem!important; }
html:root:root:root:root body.m-mode .progress-hero.gp-hero .gp-ms { width:3rem; } html:root:root:root:root body.m-mode .progress-hero .progress-ring { width:4rem;height:4rem; }
html:root:root:root:root body .global-metrics > div { overflow-wrap:normal!important;word-break:normal!important; }
html:root:root:root:root body .hero-compact .hero-visual .hero-mascot { width:auto;height:clamp(7rem, 17vh, 9.5rem);max-width:none; }
html:root:root:root:root body .hero-compact .hero-visual { justify-content:center!important; }
.pw-mascot.mascot { width:100%; }
html:root:root:root:root body #plan-wizard .ms-sym { fill:#fff; }

/* ===== Assistant : conversation sobre et réaliste (colonne centrée, réponses sans bulle) ===== */
html:root:root:root:root body.in-app #assistant-chatbot, html:root:root:root:root body.in-app.akori-dark #assistant-chatbot {
  border:0!important;background:transparent!important;box-shadow:none!important;border-radius:0!important; }
html:root:root:root:root body #assistant-chatbot .message-wrap { max-width:47rem!important;padding:.6rem .4rem 1.6rem!important; }
html:root:root:root:root body #assistant-chatbot .prose, html:root:root:root:root body #assistant-chatbot .message,
html:root:root:root:root body #assistant-chatbot .message * { overflow-wrap:break-word!important;word-break:normal!important; }
html:root:root:root:root body #assistant-chatbot .message-row { margin:.9rem 0!important;gap:.75rem!important; }
/* message de l'étudiant : bulle discrète à droite */
html:root:root:root:root body.in-app #assistant-chatbot .message-row .user.message {
  background:var(--ak-soft)!important;border:1px solid var(--ak-line)!important;border-radius:1.15rem 1.15rem .35rem 1.15rem!important;
  box-shadow:none!important;padding:.65rem 1rem!important; }
html:root:root:root:root body.in-app #assistant-chatbot .message-row .user.message,
html:root:root:root:root body.in-app #assistant-chatbot .message-row .user.message * { color:var(--ak-text)!important;-webkit-text-fill-color:var(--ak-text)!important; }
html:root:root:root:root body.in-app.akori-dark #assistant-chatbot .message-row .user.message { background:#262d3d!important;border-color:#323b50!important; }
/* la ligne de l'étudiant occupe toute la largeur : la bulle prend la largeur de son texte (sans couper les mots) */
html:root:root:root:root body #assistant-chatbot .message-row.user-row { width:100%!important;max-width:100%!important;justify-content:flex-end!important; }
html:root:root:root:root body #assistant-chatbot .user-row .flex-wrap { max-width:min(36rem, 82%)!important;width:auto!important; }
html:root:root:root:root body.in-app #assistant-chatbot .message-row .user.message { width:fit-content!important;max-width:100%!important; }
html:root:root:root:root body #assistant-chatbot .user.message .message.panel-full-width { width:auto!important; }
/* réponse d'AKORI : texte posé sur la page, sans bulle */
html:root:root:root:root body.in-app #assistant-chatbot .message-row .bot.message {
  background:transparent!important;border:0!important;box-shadow:none!important;padding:.2rem 0 0!important;max-width:100%!important;width:100%!important; }
html:root:root:root:root body.in-app #assistant-chatbot .bot-row .flex-wrap { flex:1 1 auto!important;min-width:0!important;max-width:100%!important; }
html:root:root:root:root body.in-app #assistant-chatbot .bot.message .prose { font-size:1rem!important;line-height:1.72!important; }
html:root:root:root:root body.in-app #assistant-chatbot .bot.message .prose p { margin:0 0 .7rem!important; }
html:root:root:root:root body.in-app #assistant-chatbot .bot.message .prose li { margin:.25rem 0!important; }
html:root:root:root:root body #assistant-chatbot .avatar-container { width:2.3rem!important;height:2.3rem!important;border-radius:.6rem!important;box-shadow:none!important;margin-top:.15rem!important; }
html:root:root:root:root body #assistant-chatbot .avatar-container img { border-radius:.6rem!important; }
html:root:root:root:root body #assistant-chatbot .msg-time { font-size:.68rem;opacity:.55; }
html:root:root:root:root body #assistant-chatbot .user .msg-time { text-align:right; }
/* actions sous la réponse : petites, discrètes, plus visibles au survol */
html:root:root:root:root body #assistant-chatbot .ak-msg-actions { gap:.25rem!important;margin-top:.2rem!important;opacity:.75;transition:opacity .15s; }
html:root:root:root:root body #assistant-chatbot .bot.message:hover .ak-msg-actions { opacity:1; }
html:root:root:root:root body #assistant-chatbot .ak-act { border-color:transparent!important;padding:.3rem .55rem!important;border-radius:.55rem!important;font-size:.76rem!important;white-space:nowrap; }
html:root:root:root:root body #assistant-chatbot .ak-act:hover, html:root:root:root:root body #assistant-chatbot .ak-act.on { background:var(--ak-soft)!important;border-color:var(--ak-line)!important; }
html:root:root:root:root body #assistant-chatbot .ak-act .ak-ico { width:.95rem;height:.95rem; }
html:root:root:root:root body #assistant-chatbot .ak-src-list { background:var(--ak-card)!important;border-radius:.8rem!important; }
/* zone de saisie : grande carte arrondie */
html:root:root:root:root body #assistant-bar { max-width:47rem!important;border-radius:1.4rem!important;padding:.45rem .5rem .45rem .7rem!important;
  border:1px solid var(--ak-line)!important;background:var(--ak-card)!important;box-shadow:0 6px 24px color-mix(in srgb, var(--ak-text) 9%, transparent)!important; }
html:root:root:root:root body #assistant-bar:focus-within { border-color:color-mix(in srgb, var(--ak-text) 28%, transparent)!important; }
html:root:root:root:root body #assistant-bar #assistant-input textarea { font-size:1rem!important;min-height:2.6rem!important;line-height:1.5!important; }
html:root:root:root:root body #assistant-send { border-radius:.8rem!important;width:2.5rem!important;min-width:2.5rem!important;height:2.5rem!important;min-height:2.5rem!important; }
html:root:root:root:root body #scope-row { max-width:47rem!important;justify-content:center!important; }
html:root:root:root:root body #chat-scope label { border-radius:99px!important; }
/* conversation vide : accueil centré */
.chat-empty { display:flex;flex-direction:column;align-items:center;gap:.55rem;text-align:center;padding:2rem 1rem;max-width:30rem;margin:auto; }
.chat-empty img { width:4.5rem;height:4.5rem;border-radius:1.1rem;box-shadow:0 10px 30px rgba(79,109,245,.3);animation:akpulse 4.5s ease-in-out infinite; }
.chat-empty b { font-size:1.4rem;color:var(--ak-text); } .chat-empty span { color:var(--ak-muted);line-height:1.5; }
/* Message d'accueil vide : Gradio l'enveloppe dans une bulle .md.prose (fond blanc en thème clair) : on la rend transparente. */
html:root:root:root:root body #assistant-chatbot .placeholder, html:root:root:root:root body #assistant-chatbot .placeholder .md,
html:root:root:root:root body #assistant-chatbot .placeholder .prose { background:transparent!important;border:0!important;box-shadow:none!important;padding:0!important;max-width:none!important; }
html:root:root:root:root body #assistant-chatbot .placeholder-content { display:flex;align-items:center;justify-content:center;min-height:100%; }
html:root:root:root:root body #assistant-chatbot .placeholder-content, html:root:root:root:root body #assistant-chatbot .placeholder { height:100%;display:flex;align-items:center;justify-content:center; }
/* ===== Assistant : actions en icônes alignées sous la réponse, nom affiché au survol, heure à droite ===== */
html:root:root:root:root body #assistant-chatbot .ak-msg-actions { display:flex!important;flex-wrap:nowrap!important;align-items:center!important;gap:.15rem!important;margin-top:.35rem!important;opacity:1!important; }
html:root:root:root:root body #assistant-chatbot .ak-act { position:relative;width:2.1rem!important;height:2.1rem!important;min-height:0!important;padding:0!important;justify-content:center!important;
  border-radius:.6rem!important;color:var(--ak-muted)!important;-webkit-text-fill-color:var(--ak-muted)!important; }
html:root:root:root:root body #assistant-chatbot .ak-act .ak-ico { margin:0!important;width:1.05rem!important;height:1.05rem!important; }
html:root:root:root:root body #assistant-chatbot .ak-act > span:not(.ak-ico) { position:absolute;left:50%;bottom:calc(100% + .4rem);transform:translate(-50%, .25rem);opacity:0;pointer-events:none;
  white-space:nowrap;font-size:.72rem;font-weight:600;padding:.3rem .55rem;border-radius:.45rem;background:#111827;color:#fff!important;-webkit-text-fill-color:#fff!important;
  box-shadow:0 6px 18px rgba(0,0,0,.25);transition:opacity .15s ease, transform .15s ease;z-index:5; }
html:root:root:root:root body #assistant-chatbot .ak-act:hover > span:not(.ak-ico), html:root:root:root:root body #assistant-chatbot .ak-act:focus-visible > span:not(.ak-ico) { opacity:1;transform:translate(-50%, 0); }
html:root:root:root:root body #assistant-chatbot .ak-act:hover, html:root:root:root:root body #assistant-chatbot .ak-act.on { color:var(--ak-text)!important;-webkit-text-fill-color:var(--ak-text)!important; }
html:root:root:root:root body #assistant-chatbot .ak-act.on { color:#4f6df5!important;-webkit-text-fill-color:#4f6df5!important; }
html:root:root:root:root body #assistant-chatbot .ak-msg-actions .msg-time { margin-left:auto;padding-left:.6rem;font-size:.72rem;opacity:.6;white-space:nowrap;cursor:default; }
html:root:root:root:root body #assistant-chatbot .ak-timing { display:none!important; }
@media (hover: none) { html:root:root:root:root body #assistant-chatbot .ak-act > span:not(.ak-ico) { display:none; } }

/* ===== Mindmap : barre de zoom, carte ajustée à l'écran, déplacement en glissant ===== */
.mm-bar { display:flex;align-items:center;justify-content:space-between;gap:.6rem;flex-wrap:wrap;margin-bottom:.5rem; }
html:root:root:root:root body .mm-bar .mm-hint { margin:0!important;flex:1 1 16rem; }
.mm-zoom { display:inline-flex;align-items:center;gap:.2rem;border:1px solid var(--ak-line);border-radius:99px;padding:.2rem;background:var(--ak-bg); }
html:root:root:root:root body .gradio-container button.mm-z.mm-z { width:2.1rem;height:2.1rem;min-height:0;padding:0;border-radius:50%!important;border:0!important;background:transparent!important;
  color:var(--ak-text)!important;-webkit-text-fill-color:var(--ak-text)!important;font-size:1.15rem;font-weight:700;line-height:1;cursor:pointer;box-shadow:none!important; }
html:root:root:root:root body .gradio-container button.mm-z.mm-fit { width:auto;padding:0 .8rem;border-radius:99px!important;font-size:.78rem; }
html:root:root:root:root body .gradio-container button.mm-z.mm-z:hover { background:var(--ak-soft)!important; }
.mm-zl { min-width:3.2rem;text-align:center;font-size:.75rem;font-weight:700;color:var(--ak-muted);font-variant-numeric:tabular-nums; }
html:root:root:root:root body .mm-scroll { overflow:auto!important;touch-action:none;cursor:grab;border-radius:.8rem; }
html:root:root:root:root body .mm-scroll.drag { cursor:grabbing; } html:root:root:root:root body .mm-scroll.drag * { user-select:none; }
html:root:root:root:root body .mm-scroll .mm-svg { width:calc(100% * var(--mmz, 1))!important;max-width:none!important;min-width:0!important;height:auto!important; }
html:root:root:root:root body.m-mode .mm-wrap { padding:.7rem!important; } html:root:root:root:root body.m-mode .mm-bar .mm-hint { font-size:.74rem; }

/* ===== Dictée : indicateur « écoute en cours » et texte reconnu ===== */
.mic-live { display:none;margin:.35rem 0 .1rem;padding:.5rem .75rem;border-radius:.7rem;background:rgba(229,72,77,.1);border:1px solid rgba(229,72,77,.35);font-size:.82rem; }
.mic-live.on { display:flex;flex-direction:column;gap:.15rem; }
.mic-live b { color:#e5484d;display:flex;align-items:center;gap:.45rem; }
.mic-live b::before { content:"";width:.55rem;height:.55rem;border-radius:50%;background:#e5484d;animation:akpulse 1.1s ease-in-out infinite; }
.mic-live span { color:var(--ak-text);font-style:italic; }
html:root:root:root:root body #assistant-bar { position:relative; }
html:root:root:root:root body #assistant-bar .mic-live { position:absolute;left:1rem;right:1rem;bottom:calc(100% + .4rem);z-index:4;background:var(--ak-card); }
html:root:root:root:root body #oral-answer textarea { font-size:1rem!important;line-height:1.55!important; }

/* ===== Colle orale : résultat lisible sur téléphone ===== */
html:root:root:root:root body .oral-top { flex-wrap:wrap!important; }
html:root:root:root:root body .oral-top > div:last-child { flex:1 1 14rem;min-width:0;overflow-wrap:normal!important;word-break:normal!important; }
html:root:root:root:root body.m-mode .oral-top > div:last-child, html:root:root:root:root body.ak-narrow .oral-top > div:last-child { flex-basis:100%; }
html:root:root:root:root body .oral-result, html:root:root:root:root body .oral-result * { overflow-wrap:break-word!important;word-break:normal!important; }
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
  const I18N = [["NAVIGATION", "NAVIGATION"], ["Accueil", "Home"], ["Mes dossiers", "My folders"], ["Réviser", "Revise"], ["Flashcards", "Flashcards"], ["Quiz / QCM", "Quiz / MCQ"], ["Résumé", "Summary"], ["Assistant IA", "AI Assistant"], ["Progression", "Progress"], ["À revoir", "To review"], ["Historique", "History"], ["Paramètres", "Settings"], ["Administration", "Administration"], ["Déconnexion", "Log out"], ["Administrateur", "Administrator"], ["Utilisateur", "User"], ["AKORI · Espace de révision", "AKORI · Revision space"], ["Cours actif", "Active course"], ["Bonjour 👋", "Hello 👋"], ["Que révisez-vous aujourd'hui ?", "What are you revising today?"], ["AKORI transforme vos cours PDF en un espace de révision intelligent : résumé, flashcards roulette, quiz et assistant RAG.", "AKORI turns your PDF courses into a smart revision space: summary, roulette flashcards, quiz and RAG assistant."], ["💡 Guide de révision AKORI & mode roulette flashcards", "💡 AKORI revision guide & flashcards roulette mode"], ["1. Chargez vos dossiers", "1. Load your folders"], ["Glissez vos PDF dans Mes dossiers pour activer l'indexation FAISS.", "Drop your PDFs in My folders to enable FAISS indexing."], ["2. Lancement roulette", "2. Start the roulette"], ["Générez au moins 7 flashcards. Les questions défilent automatiquement en boucle.", "Generate at least 7 flashcards. Questions scroll automatically in a loop."], ["3. Clic & réponse effacée", "3. Click & answer hidden"], ["Cliquez sur la carte pour stopper/relancer. La réponse s'efface à chaque relance.", "Click the card to stop/restart. The answer is hidden at each restart."], ["Cours importés", "Imported courses"], ["Fragments indexés", "Indexed fragments"], ["Échanges du cours", "Course exchanges"], ["Recherche active", "Search active"], ["Réviser un cours en un clic", "Revise a course in one click"], ["Générez les outils principaux à partir du cours actif.", "Generate the main tools from the active course."], ["Résumé · Flashcards · Quiz / QCM · Assistant IA", "Summary · Flashcards · Quiz / MCQ · AI Assistant"], ["Mes cours", "My courses"], ["Vos supports de révision indexés localement.", "Your revision materials indexed locally."], ["Aucun cours sélectionné", "No course selected"], ["Commencer la révision →", "Start revising →"], ["Tous vos supports PDF sont centralisés ici. Dès qu'un document est sélectionné, AKORI extrait son contenu et construit automatiquement son index.", "All your PDF materials are gathered here. As soon as a document is selected, AKORI extracts its content and builds its index automatically."], ["＋\nAjouter un document", "＋\nAdd a document"], ["Sélectionnez un PDF : extraction et indexation automatiques.", "Select a PDF: automatic extraction and indexing."], ["Indexé", "Indexed"], ["Sélectionnez un cours pour afficher son espace de révision.", "Select a course to display its revision space."], ["Date inconnue", "Unknown date"], ["Général", "General"], ["📁 Aucun document pour le moment.", "📁 No document yet."], ["Ajoutez votre premier PDF pour commencer.", "Add your first PDF to get started."], ["📚 Aucun cours pour le moment.", "📚 No course yet."], ["Importez votre premier PDF dans 'Mes dossiers' pour commencer.", "Import your first PDF in 'My folders' to get started."], ["Aucun document chargé.", "No document loaded."], ["Réviser ce cours", "Revise this course"], ["Une vue centrale pour accéder rapidement au résumé, aux flashcards, au quiz et à l'assistant.", "A central view to quickly reach the summary, flashcards, quiz and assistant."], ["Flashcards — Mode Roulette", "Flashcards — Roulette mode"], ["✦ Générer les flashcards", "✦ Generate flashcards"], ["🔄 Régénérer (7 minimum)", "🔄 Regenerate (7 minimum)"], ["Afficher la réponse", "Show the answer"], ["✓ Je savais", "✓ I knew it"], ["↻ À revoir", "↻ To review"], ["Aucune flashcard générée.", "No flashcards generated."], ["Choisissez un cours puis cliquez sur « Générer les flashcards ».", "Choose a course then click “Generate flashcards”."], ["SESSION TERMINÉE", "SESSION COMPLETED"], ["Maîtrise", "Mastery"], ["Maîtrisées", "Mastered"], ["Cliquez sur « 🔄 Régénérer (7 minimum) » pour obtenir une nouvelle série.", "Click “🔄 Regenerate (7 minimum)” to get a new set."], ["Quiz d'évaluation", "Evaluation quiz"], ["Cliquez sur une réponse, validez, puis passez à la question suivante.", "Click an answer, validate, then go to the next question."], ["✦ Générer le quiz", "✦ Generate the quiz"], ["🔁 Rejouer", "🔁 Replay"], ["Valider la réponse", "Validate the answer"], ["Question suivante →", "Next question →"], ["Voir le résultat 🎯", "See the result 🎯"], ["Choisissez votre réponse", "Choose your answer"], ["Aucun quiz généré.", "No quiz generated."], ["Choisissez un cours puis cliquez sur « Générer le quiz ».", "Choose a course then click “Generate the quiz”."], ["QUIZ TERMINÉ", "QUIZ COMPLETED"], ["Votre résultat a été enregistré dans la progression de ce cours.", "Your result has been saved in this course's progress."], ["🏆 Sans faute, bravo !", "🏆 Perfect score, well done!"], ["✅ Bonne réponse !", "✅ Correct answer!"], ["❌ Réponse incorrecte", "❌ Incorrect answer"], ["Explication :", "Explanation:"], ["Résumé du cours", "Course summary"], ["Générer le résumé", "Generate the summary"], ["Sélectionnez un cours puis lancez la génération.", "Select a course then start the generation."], ["Assistant AKORI", "AKORI Assistant"], ["Posez une question sur le cours actif. Le moteur récupère d'abord les passages pertinents avec FAISS, puis Gemini génère la réponse à partir du contexte récupéré.", "Ask a question about the active course. The engine first retrieves the relevant passages with FAISS, then Gemini generates the answer from the retrieved context."], ["Envoyer", "Send"], ["Posez une question sur le cours…", "Ask a question about the course…"], ["Ma progression", "My progress"], ["Commencez par la vue globale, puis consultez le détail du cours sélectionné.", "Start with the global view, then check the details of the selected course."], ["Détail du cours sélectionné", "Selected course details"], ["Votre progression commencera ici.", "Your progress will start here."], ["Importez un cours pour créer votre premier suivi. Tant qu'aucun cours n'est chargé, la progression reste à 0 %.", "Import a course to create your first tracking. As long as no course is loaded, progress stays at 0%."], ["Aucune progression à afficher.", "No progress to display."], ["Importez un cours puis utilisez les flashcards ou terminez un quiz pour commencer à construire votre progression.", "Import a course then use the flashcards or finish a quiz to start building your progress."], ["VUE D'ENSEMBLE", "OVERVIEW"], ["Progression globale", "Global progress"], ["Synthèse de tous vos cours réellement importés et de vos interactions de révision.", "Summary of all your imported courses and your revision interactions."], ["Cours suivis", "Tracked courses"], ["Cours commencés", "Started courses"], ["Flashcards maîtrisées", "Mastered flashcards"], ["Moyenne des quiz", "Quiz average"], ["Progression par cours", "Progress by course"], ["Le détail pédagogique reste lié au cours sélectionné : points forts, points faibles, flashcards et résultats des quiz.", "The detailed breakdown stays linked to the selected course: strengths, weaknesses, flashcards and quiz results."], ["Calculée uniquement à partir des interactions enregistrées sur ce cours.", "Computed only from the interactions recorded on this course."], ["Quiz terminés", "Completed quizzes"], ["✦ Points forts", "✦ Strengths"], ["↗ Points faibles", "↗ Weaknesses"], ["Pas encore assez de réponses de quiz pour identifier un point fort.", "Not enough quiz answers yet to identify a strength."], ["Pas encore assez de réponses de quiz pour identifier un point faible.", "Not enough quiz answers yet to identify a weakness."], ["Aucun point fort clairement établi pour le moment.", "No clear strength established yet."], ["Aucun point faible clairement établi pour le moment.", "No clear weakness established yet."], ["Cette section regroupera les flashcards marquées « À revoir » et les erreurs de quiz.", "This section will gather the flashcards marked “To review” and the quiz mistakes."], ["📌 Votre file « À revoir » apparaîtra ici après les interactions.", "📌 Your “To review” queue will appear here after some interactions."], ["Historique du cours actif", "Active course history"], ["Aucun historique pour le moment.", "No history yet."], ["Aucune conversation enregistrée.", "No conversation recorded."], ["Vous", "You"], ["Personnalisez AKORI. Vos choix sont mémorisés dans ce navigateur.", "Customize AKORI. Your choices are saved in this browser."], ["Mode sombre", "Dark mode"], ["Basculez entre le thème clair et le thème sombre d'un seul clic.", "Switch between the light and dark theme with one click."], ["Langue", "Language"], ["Langue de l'interface et des réponses générées par l'IA.", "Language of the interface and of the generated summaries, flashcards and quizzes. In the chat, AKORI answers in the language you write in."], ["Connexion", "Sign in"], ["Créer un compte", "Create an account"], ["Nom d'utilisateur", "Username"], ["Mot de passe", "Password"], ["Confirmer le mot de passe", "Confirm the password"], ["Rester connecté", "Stay signed in"], ["Se connecter", "Sign in"], ["Créer mon compte", "Create my account"], ["Bienvenue sur AKORI", "Welcome to AKORI"], ["Connectez-vous pour retrouver vos cours, flashcards, quiz et historiques archivés.", "Sign in to find your archived courses, flashcards, quizzes and history."], ["Vos données sont enregistrées dans votre compte et restent disponibles à chaque connexion.", "Your data is saved in your account and available every time you sign in."], ["⚠️ Nom d'utilisateur invalide (3 à 32 caractères : lettres, chiffres, . _ -).", "⚠️ Invalid username (3 to 32 characters: letters, digits, . _ -)."], ["⚠️ Mot de passe trop court (6 caractères minimum).", "⚠️ Password too short (6 characters minimum)."], ["⚠️ Ce nom d'utilisateur existe déjà.", "⚠️ This username already exists."], ["✅ Compte créé.", "✅ Account created."], ["⚠️ Identifiants incorrects.", "⚠️ Incorrect username or password."], ["⛔ Ce compte est désactivé. Contactez l'administrateur.", "⛔ This account is disabled. Contact the administrator."], ["⏳ Trop de tentatives. Réessayez dans une minute.", "⏳ Too many attempts. Try again in a minute."], ["⚠️ Les mots de passe ne correspondent pas.", "⚠️ The passwords do not match."], ["✅ Vous êtes déconnecté.", "✅ You are signed out."], ["Connexion requise.", "Sign-in required."], ["⏳ Analyse du PDF… extraction, découpage et indexation en cours.", "⏳ Analyzing the PDF… extraction, splitting and indexing in progress."], ["⏳ Extraction du texte…", "⏳ Extracting the text…"], ["⏳ Enregistrement…", "⏳ Saving…"], ["⚠️ Aucun fichier sélectionné.", "⚠️ No file selected."], ["⚠️ Aucun texte exploitable dans ce PDF (document scanné ?).", "⚠️ No usable text in this PDF (scanned document?)."], ["⚠️ Veuillez d'abord sélectionner un cours.", "⚠️ Please select a course first."], ["⏳ Génération…", "⏳ Generating…"], ["⚠️ Le titre du document est indisponible.", "⚠️ The document title is unavailable."], ["⚠️ Aucune information pertinente n'a pu être extraite du document.", "⚠️ No relevant information could be extracted from the document."], ["Gemini est très sollicité en ce moment. Réessayez dans une dizaine de secondes.", "Gemini is very busy right now. Try again in about ten seconds."], ["Quota ou limite temporaire Gemini atteinte. Attendez un peu avant de relancer la génération.", "Gemini quota or temporary limit reached. Wait a little before generating again."], ["⚠️ Gemini est temporairement indisponible. Réessayez dans quelques secondes.", "⚠️ Gemini is temporarily unavailable. Try again in a few seconds."], ["⚠️ La limite temporaire de Gemini a été atteinte. Attendez un peu puis réessayez.", "⚠️ Gemini's temporary limit was reached. Wait a little then try again."], ["⚠️ Impossible de générer les flashcards à partir du contexte.", "⚠️ Unable to generate flashcards from the context."], ["⚠️ Impossible de générer le quiz.", "⚠️ Unable to generate the quiz."], ["⚠️ Sélectionnez d'abord un cours.", "⚠️ Select a course first."], ["⚠️ Aucun document sélectionné.", "⚠️ No document selected."], ["❌ Document vide.", "❌ Empty document."], ["❌ Le modèle n'a renvoyé aucune flashcard valide.", "❌ The model returned no valid flashcard."], ["⏳ Génération des flashcards…", "⏳ Generating flashcards…"], ["⏳ Génération du quiz…", "⏳ Generating the quiz…"], ["⏳ Génération du résumé…", "⏳ Generating the summary…"], ["🔁 Quiz relancé.", "🔁 Quiz restarted."], ["⚠️ Générez d'abord un quiz.", "⚠️ Generate a quiz first."], ["ℹ️ Réponse déjà validée : passez à la suite.", "ℹ️ Answer already validated: move on."], ["⚠️ Sélectionnez une réponse avant de valider.", "⚠️ Select an answer before validating."], ["✅ Bonne réponse.", "✅ Correct answer."], ["❌ Réponse incorrecte. Consultez l'explication.", "❌ Incorrect answer. Check the explanation."], ["⚠️ Validez d'abord cette réponse.", "⚠️ Validate this answer first."], ["Console d'administration", "Administration console"], ["Suivi, maintenance et gestion des comptes AKORI", "Monitoring, maintenance and account management for AKORI"], ["← Retour à l'application", "← Back to the app"], ["↻ Actualiser", "↻ Refresh"], ["Utilisateurs", "Users"], ["Activité", "Activity"], ["Maintenance", "Maintenance"], ["Nouveau mot de passe", "New password"], ["Activer / Désactiver", "Enable / Disable"], ["Réinitialiser le mot de passe", "Reset the password"], ["Promouvoir / Rétrograder", "Promote / Demote"], ["Supprimer le compte", "Delete the account"], ["Je confirme la suppression définitive du compte et de ses données", "I confirm the permanent deletion of the account and its data"], ["Sauvegarde complète (ZIP)", "Full backup (ZIP)"], ["Recharger les index", "Reload the indexes"], ["Purger les connexions expirées", "Purge expired sign-ins"], ["Alléger le journal", "Trim the log"], ["Sauvegarde", "Backup"], ["Comptes", "Accounts"], ["Actifs (24 h)", "Active (24 h)"], ["Stockage utilisé", "Storage used"], ["Requêtes IA", "AI requests"], ["Jetons estimés", "Estimated tokens"], ["Erreurs (24 h)", "Errors (24 h)"], ["Sessions en ligne", "Online sessions"], ["Modèle principal", "Main model"], ["Modèles de secours", "Fallback models"], ["Clé API Gemini", "Gemini API key"], ["Disponibilité", "Uptime"], ["Dossier de données", "Data folder"], ["Taille PDF maximale", "Maximum PDF size"], ["✅ configurée", "✅ configured"], ["⛔ Accès réservé à l'administrateur.", "⛔ Reserved for the administrator."], ["⚠️ Sélectionnez un utilisateur.", "⚠️ Select a user."], ["⚠️ Vous ne pouvez pas désactiver votre propre compte.", "⚠️ You cannot disable your own account."], ["⚠️ Vous ne pouvez pas supprimer votre propre compte.", "⚠️ You cannot delete your own account."], ["⚠️ Il doit rester au moins un administrateur.", "⚠️ At least one administrator must remain."], ["⚠️ Cochez la confirmation pour supprimer définitivement ce compte et ses données.", "⚠️ Tick the confirmation to permanently delete this account and its data."], ["Rôle", "Role"], ["Statut", "Status"], ["Créé le", "Created on"], ["Dernière connexion", "Last sign-in"], ["Connexions", "Sign-ins"], ["Cours", "Courses"], ["Échanges", "Exchanges"], ["Quiz", "Quizzes"], ["Stockage", "Storage"], ["Actif", "Active"], ["Désactivé", "Disabled"], ["Date", "Date"], ["Événement", "Event"], ["Détail", "Detail"], ["⚠️ Session expirée : rechargez la page pour vous reconnecter.", "⚠️ Session expired: reload the page to sign in again."], ["Gemini met trop de temps à répondre. Réessayez dans un instant.", "Gemini is taking too long to answer. Try again in a moment."], ["Content de te revoir", "Good to see you again"], ["Connecte-toi pour accéder à tes cours", "Sign in to access your courses"], ["Email", "Email"], ["Mot de passe oublié ?", "Forgot your password?"], ["Pas encore de compte ?", "No account yet?"], ["Crée ton compte", "Create your account"], ["Gratuit, sans carte bancaire", "Free, no credit card"], ["Déjà un compte ?", "Already have an account?"], ["ton@email.com", "you@email.com"], ["8 caractères minimum", "8 characters minimum"], ["Comment tu t'appelles ?", "What's your name?"], ["Et quel est ton niveau scolaire ?", "And what's your school level?"], ["Prénom", "First name"], ["Ton prénom", "Your first name"], ["Niveau", "Level"], ["Objectif", "Goal"], ["🏫 Collège\nDe la 6ème à la 3ème", "🏫 Middle school\nGrades 6 to 9"], ["🎓 Lycée\nSeconde, Première, Terminale", "🎓 High school\nYears 10 to 12"], ["🎯 Études supérieures\nUniversité, Prépa, BTS...", "🎯 Higher education\nUniversity, prep school, vocational..."], ["📚 Réviser mes cours\nRelire, comprendre et mémoriser régulièrement", "📚 Revise my courses\nReread, understand and memorize regularly"], ["📝 Préparer un examen\nFlashcards et quiz avant le jour J", "📝 Prepare for an exam\nFlashcards and quizzes before the big day"], ["🧠 Mieux comprendre un chapitre\nPoser mes questions directement au cours", "🧠 Understand a chapter better\nAsk my questions directly to the course"], ["Continuer", "Continue"], ["Retour", "Back"], ["Accéder à AKORI", "Open AKORI"], ["Que veux-tu faire avec AKORI ?", "What do you want to do with AKORI?"], ["Choisis ton objectif principal", "Choose your main goal"], ["Dans quelle langue veux-tu travailler ?", "Which language do you want to work in?"], ["Tu pourras la changer à tout moment dans les paramètres", "You can change it at any time in the settings"], ["Ton espace est prêt. Importe ton premier cours en PDF et laisse AKORI générer tes flashcards, quiz et résumés.", "Your space is ready. Import your first course as a PDF and let AKORI generate your flashcards, quizzes and summaries."], ["⚠️ Adresse e-mail invalide.", "⚠️ Invalid email address."], ["⚠️ Un compte existe déjà avec cet e-mail.", "⚠️ An account already exists with this email."], ["⚠️ Entre ton prénom et choisis ton niveau.", "⚠️ Enter your first name and choose your level."], ["⚠️ Choisis ton objectif principal.", "⚠️ Choose your main goal."], ["⚠️ Ton prénom est trop long (40 caractères maximum).", "⚠️ Your first name is too long (40 characters maximum)."], ["ℹ️ Pour réinitialiser ton mot de passe, contacte l'administrateur d'AKORI.", "ℹ️ To reset your password, contact the AKORI administrator."], ["Stockage", "Storage"], ["✅ Google Drive", "✅ Google Drive"], ["⚠️ Temporaire (non sauvegardé)", "⚠️ Temporary (not backed up)"], ["⏳ Mise à jour de l'index du cours (une seule fois)…", "⏳ Updating the course index (one time only)…"], ["OCR (PDF scannés)", "OCR (scanned PDFs)"], ["désactivé", "disabled"], ["⚠️ Aucun texte n'a pu être lu dans ce PDF scanné (OCR sans résultat).", "⚠️ No text could be read in this scanned PDF (OCR returned nothing)."], ["Tester Gemini", "Test Gemini"], ["Profil", "Profile"], ["Mon profil", "My profile"], ["Ajoutez votre photo et ajustez vos informations. Elles apparaissent dans le menu et sur l'accueil.", "Add your photo and adjust your information. They appear in the menu and on the home page."], ["Ajouter une photo", "Add a photo"], ["Retirer la photo", "Remove the photo"], ["Enregistrer", "Save"], ["Ajoutez une photo pour personnaliser votre espace.", "Add a photo to personalize your space."], ["Collège", "Middle school"], ["Lycée", "High school"], ["Études supérieures", "Higher education"], ["Réviser mes cours", "Revise my courses"], ["Préparer un examen", "Prepare for an exam"], ["Mieux comprendre un chapitre", "Understand a chapter better"], ["✅ Photo de profil enregistrée.", "✅ Profile photo saved."], ["✅ Photo retirée.", "✅ Photo removed."], ["✅ Profil enregistré.", "✅ Profile saved."], ["⚠️ Image trop volumineuse (8 Mo maximum).", "⚠️ Image too large (8 MB maximum)."], ["⚠️ Image illisible : choisissez un fichier JPG ou PNG.", "⚠️ Unreadable image: choose a JPG or PNG file."], ["Pas encore commencé : générez une série dans Flashcards.", "Not started: generate a set in Flashcards."], ["Pas encore commencé : terminez un quiz pour obtenir une moyenne.", "Not started: finish a quiz to get an average."], ["Pas encore commencé : lancez une colle orale.", "Not started: start an oral exam."], ["Générez vos flashcards pour commencer à mémoriser.", "Generate your flashcards to start memorizing."], ["Terminez un quiz pour mesurer ce que vous retenez.", "Finish a quiz to measure what you retain."], ["Passez une colle orale pour vous entraîner à expliquer le cours.", "Take an oral exam to practice explaining the course."], ["Bravo : les trois activités sont lancées. Continuez pour monter vers 100 %.", "Well done: all three activities are started. Keep going towards 100%."], ["question(s) posée(s) à l'assistant", "question(s) asked to the assistant"], ["Résumé :", "Summary:"], ["généré", "generated"], ["non généré", "not generated"], ["chapitre(s) épinglé(s) dans À revoir", "chapter(s) pinned in To review"], ["d'après vos réponses de quiz sur ce cours", "based on your quiz answers for this course"], ["Moyenne de la progression de tous vos cours. Chaque cours compte 3 activités (flashcards, quiz, colle orale) : une activité non commencée compte pour 0 %.", "Average progress of all your courses. Each course counts 3 activities (flashcards, quiz, oral exam): an activity not started counts as 0%."], ["Le détail pédagogique reste lié au cours sélectionné : activités, points forts, points faibles et prochaine étape.", "The detailed breakdown stays linked to the selected course: activities, strengths, weaknesses and next step."], ["Résumé · Flashcards · Quiz / QCM · Colle orale · Assistant IA", "Summary · Flashcards · Quiz / MCQ · Oral exam · AI Assistant"], ["AKORI transforme vos cours PDF en un espace de révision intelligent : résumé, flashcards roulette, quiz, colle orale et assistant RAG.", "AKORI turns your PDF courses into a smart revision space: summary, roulette flashcards, quiz, oral exam and RAG assistant."], ["Passages indexés", "Indexed passages"], ["dans votre espace", "in your space"], ["Questions posées", "Questions asked"], ["Progression du cours actif", "Active course progress"], ["flashcards, quiz et colle orale", "flashcards, quiz and oral exam"], ["💡 Guide de révision : que faire avec chaque fonction ?", "💡 Revision guide: what to do with each feature"], ["Importez vos PDF, même scannés : AKORI lit le texte, le découpe et l'indexe.", "Import your PDFs, even scanned ones: AKORI reads the text, splits it and indexes it."], ["Les points clés du cours en un coup d'œil, dans votre langue.", "The key points of the course at a glance, in your language."], ["Posez vos questions : la réponse s'appuie sur votre cours, avec la page citée. Épinglez ou écoutez-la.", "Ask your questions: the answer relies on your course, with the page cited. Pin it or listen to it."], ["Au moins 7 cartes en mode roulette : cliquez pour stopper, la réponse s'efface à chaque relance.", "At least 7 cards in roulette mode: click to stop, the answer is hidden at each restart."], ["Testez-vous : correction expliquée, points forts et faibles dans Progression.", "Test yourself: explained correction, strengths and weaknesses in Progress."], ["Répondez à l'oral ou à l'écrit : note sur 10 d'après votre cours et page à relire.", "Answer orally or in writing: score out of 10 based on your course and the page to reread."], ["Retrouvez les réponses que vous avez épinglées, cours par cours.", "Find the answers you pinned, course by course."], ["Importer un PDF →", "Import a PDF →"], ["Générer le résumé →", "Generate the summary →"], ["Poser une question →", "Ask a question →"], ["Lancer la roulette →", "Start the roulette →"], ["Passer un quiz →", "Take a quiz →"], ["Passer une colle →", "Take an oral →"], ["Ouvrir À revoir →", "Open To review →"], ["Continuer", "Continue"], ["Commencer", "Start"], ["PROCHAINE ÉTAPE", "NEXT STEP"], ["Moyenne des colles", "Oral exam average"], ["tous cours confondus", "all courses combined"], ["au moins une activité faite", "at least one activity done"], ["Colle orale", "Oral exam"], ["AKORI vous interroge sur votre cours, comme à l'oral. Répondez à l'écrit ou à la voix : la correction se base uniquement sur le contenu du cours et vous indique la page à relire.", "AKORI quizzes you on your course, like an oral exam. Answer in writing or by voice: the grading relies only on the course content and tells you which page to reread."], ["Prêt pour votre colle ?", "Ready for your oral?"], ["Cliquez sur « Nouvelle question » : AKORI vous interroge sur votre cours, comme à l'oral.", "Click “New question”: AKORI quizzes you on your course, like an oral exam."], ["Nouvelle question", "New question"], ["Dicter ma réponse", "Dictate my answer"], ["Votre réponse", "Your answer"], ["Écrivez votre réponse ici, ou cliquez sur « Dicter ma réponse »…", "Write your answer here, or click “Dictate my answer”…"], ["Corriger ma réponse", "Grade my answer"], ["Question orale", "Oral question"], ["Très bien", "Very good"], ["Correct, à compléter", "Correct, needs completing"], ["À reprendre", "To redo"], ["Points maîtrisés", "Mastered points"], ["À retravailler", "To work on"], ["Aucun point clé repéré.", "No key point found."], ["Rien d'oublié, bravo.", "Nothing forgotten, well done."], ["À relire", "To reread"], ["Réponse modèle", "Model answer"], ["Mes dernières colles", "My latest orals"], ["Répondez à l'écrit ou à la voix, puis cliquez sur « Corriger ma réponse ».", "Answer in writing or by voice, then click “Grade my answer”."], ["⚠️ Cliquez d'abord sur « Nouvelle question ».", "⚠️ Click “New question” first."], ["⚠️ Écrivez ou dictez une réponse avant de la faire corriger.", "⚠️ Write or dictate an answer before grading it."], ["Correction approximative (IA indisponible).", "Approximate grading (AI unavailable)."], ["Liste des utilisateurs", "User list"], ["Rechercher un utilisateur", "Search a user"], ["N°", "No."], ["Profil", "Profile"], ["E-mail", "Email"], ["Mot de passe", "Password"], ["Chiffré", "Encrypted"], ["Modifier", "Edit"], ["Supprimer", "Delete"], ["Générer un mot de passe temporaire", "Generate a temporary password"], ["Sélectionnez un utilisateur dans la liste ou cliquez sur « Modifier ».", "Select a user in the list or click “Edit”."], ["Mot de passe chiffré : personne ne peut le lire. Définissez-en un nouveau ou générez un mot de passe temporaire.", "Encrypted password: nobody can read it. Set a new one or generate a temporary password."], ["Créé le", "Created on"], ["Afficher le mot de passe", "Show password"], ["Épingler comme chapitre", "Pin as chapter"], ["Lire à voix haute", "Read aloud"], ["Arrêter la lecture", "Stop reading"], ["Épinglé ✓", "Pinned ✓"], ["Lecture vocale indisponible", "Voice reading unavailable"], ["Extrait", "Excerpt"], ["Chapitres", "Chapters"], ["Chapitres épinglés", "Pinned chapters"], ["Les réponses de l'assistant que vous épinglez depuis le chat sont rangées ici, cours par cours. Relisez-les, ou écoutez-les, avant l'examen.", "The assistant answers you pin from the chat are kept here, course by course. Reread them, or listen to them, before the exam."], ["Aucun chapitre épinglé pour ce cours.", "No pinned chapter for this course."], ["Dans l'assistant, cliquez sur « Épingler comme chapitre » sous une réponse utile.", "In the assistant, click “Pin as chapter” under a useful answer."], ["Retirer", "Remove"], ["📌 Chapitre épinglé", "📌 Chapter pinned"], ["Retrouvez-le dans À revoir.", "Find it in To review."], ["Recherche (médiane)", "Search (median)"], ["1er mot (médiane)", "First word (median)"], ["Réponse complète (médiane)", "Full answer (median)"], ["Réponse complète (95e centile)", "Full answer (95th percentile)"], ["Questions hors cours refusées", "Off-topic questions declined"], ["Modèle d'embeddings", "Embedding model"], ["Seuil de pertinence", "Relevance threshold"], ["Mesures sur", "Measured over"]];
  const PATTERNS = [["^(\\d+) maîtrisée\\(s\\) sur (\\d+) · (\\d+) à revoir$", "$1 mastered out of $2 · $3 to review"], ["^(\\d+) maîtrisée\\(s\\) sur (\\d+)$", "$1 mastered out of $2"], ["^(\\d+) quiz terminé\\(s\\) · moyenne (\\d+) % sur (\\d+) questions$", "$1 quiz(zes) completed · average $2% over $3 questions"], ["^(\\d+) colle\\(s\\) · moyenne (.+)\\/10$", "$1 oral(s) · average $2/10"], ["^Progression du cours · (.*)$", "Course progress · $1"], ["^Moyenne de 3 activités : flashcards, quiz et colle orale \\((\\d)\\/3 commencées\\)\\. Une activité non commencée compte pour 0 %\\.$", "Average of 3 activities: flashcards, quiz and oral exam ($1/3 started). An activity not started counts as 0%."], ["^dans le cours actif \\((\\d+) au total\\)$", "in the active course ($1 in total)"], ["^à l'assistant sur ce cours \\((\\d+) sur tous vos cours\\)$", "to the assistant on this course ($1 across all your courses)"], ["^(\\d+) quiz terminé\\(s\\), tous cours$", "$1 quiz(zes) completed, all courses"], ["^(\\d+) colle\\(s\\), tous cours$", "$1 oral(s), all courses"], ["^Flashcards (.+) · Quiz (.+) · Colle (.+)$", "Flashcards $1 · Quiz $2 · Oral $3"], ["^Revoyez vos (\\d+) flashcard\\(s\\) marquée\\(s\\) « à revoir »\\.$", "Review your $1 flashcard(s) marked “to review”."], ["^(\\d+) passages indexés · Progression du cours (\\d+)%$", "$1 indexed passages · Course progress $2%"], ["^⏳ Préparation de la question… \\((\\d+) s\\)$", "⏳ Preparing the question… ($1 s)"], ["^⏳ Correction en cours… \\((\\d+) s\\)$", "⏳ Grading in progress… ($1 s)"], ["^✅ Note : (\\d+)\\/10$", "✅ Score: $1/10"], ["^(\\d+) séance\\(s\\) · moyenne (.+)\\/10$", "$1 session(s) · average $2/10"], ["^Source : (.*)$", "Source: $1"], ["^(\\d+) cours · (\\d+) échanges · (.+)$", "$1 courses · $2 exchanges · $3"], ["^(\\d+) connexions$", "$1 sign-ins"], ["^✅ Mot de passe temporaire de « (.+) » : (.+) — à lui communiquer \\(affiché une seule fois\\)\\.$", "✅ Temporary password of “$1”: $2 — give it to the user (shown only once)."], ["^ℹ️ Pour supprimer « (.+) » : cochez la confirmation, puis cliquez sur « Supprimer le compte »\\.$", "ℹ️ To delete “$1”: tick the confirmation, then click “Delete the account”."], ["^ℹ️ Modification de « (.+) » : choisissez une action ci-dessous\\.$", "ℹ️ Editing “$1”: choose an action below."], ["^Clé API Gemini invalide : elle contient des caractères non autorisés \\(accents, symboles\\)\\. Collez la clé exacte, qui commence par AIza, dans le secret GEMINI_API_KEY\\.$", "Invalid Gemini API key: it contains forbidden characters (accents, symbols). Paste the exact key, starting with AIza, into the GEMINI_API_KEY secret."], ["^Gemini est très sollicité en ce moment\\. Réessayez dans une dizaine de secondes\\.(.*)$", "Gemini is very busy right now. Try again in about ten seconds.$1"], ["^Quota Gemini atteint : la limite de requêtes de votre clé est épuisée \\(par minute ou par jour\\)\\. Attendez une minute ou utilisez une autre clé\\.(.*)$", "Gemini quota reached: your key's request limit is used up (per minute or per day). Wait a minute or use another key.$1"], ["^Aucun modèle Gemini n'est accessible avec cette clé \\(404\\)\\. Vérifiez GEMINI_API_KEY et les noms de modèles\\.(.*)$", "No Gemini model is available with this key (404). Check GEMINI_API_KEY and the model names.$1"], ["^Connexion à Gemini impossible \\(réseau ou délai dépassé\\)\\. Réessayez dans un instant\\.(.*)$", "Cannot reach Gemini (network or timeout). Try again in a moment.$1"], ["^⏳ Lecture OCR des pages scannées (\\d+)\\/(\\d+)…$", "⏳ Reading scanned pages with OCR $1/$2…"], ["^✅ '(.+)' indexé avec succès \\((\\d+) fragments, OCR sur (\\d+) pages\\)\\.$", "✅ '$1' indexed successfully ($2 fragments, OCR on $3 pages)."], ["^Chapitre (\\d+)$", "Chapter $1"], ["^(\\d+) réponse\\(s\\)$", "$1 answer(s)"], ["^✅ '(.+)' indexé avec succès \\((\\d+) fragments\\)\\.$", "✅ '$1' indexed successfully ($2 fragments)."], ["^✅ '(.+)' est déjà indexé : réutilisé instantanément\\.$", "✅ '$1' is already indexed: reused instantly."], ["^❌ Erreur lors de l'indexation : (.*)$", "❌ Indexing error: $1"], ["^⏳ Indexation (\\d+)\\/(\\d+)…$", "⏳ Indexing $1/$2…"], ["^⚠️ Fichier trop volumineux \\(maximum (\\d+) Mo\\)\\.$", "⚠️ File too large (maximum $1 MB)."], ["^✅ (\\d+) flashcards prêtes !$", "✅ $1 flashcards ready!"], ["^⚠️ Seulement (\\d+) flashcard\\(s\\) obtenue\\(s\\) : il en faut au moins (\\d+)\\. Relancez la génération\\.$", "⚠️ Only $1 flashcard(s) obtained: at least $2 are needed. Generate again."], ["^❌ Erreur : (.*)$", "❌ Error: $1"], ["^⚠️ Génération impossible : (.*)$", "⚠️ Generation failed: $1"], ["^✅ Quiz de (\\d+) questions généré\\.$", "✅ Quiz of $1 questions generated."], ["^🎯 Quiz terminé : (\\d+)\\/(\\d+)\\.$", "🎯 Quiz completed: $1/$2."], ["^Score : (\\d+)$", "Score: $1"], ["^(\\d+)% de bonnes réponses$", "$1% correct answers"], ["^Bonne réponse : (.*)$", "Correct answer: $1"], ["^Le titre du document est : (.*)$", "The title of the document is: $1"], ["^QUIZ \\/ QCM · (.*)$", "QUIZ / MCQ · $1"], ["^(\\d+) fragments indexés · Progression (\\d+)% · RAG local$", "$1 indexed fragments · Progress $2% · Local RAG"], ["^Progression réelle · (.*)$", "Real progress · $1"], ["^Ajouté le (.*)$", "Added on $1"], ["^Indexé · (\\d+) fragments$", "Indexed · $1 fragments"], ["^✅ Compte « (.+) » (activé|désactivé)\\.$", "✅ Account “$1” updated."], ["^✅ Mot de passe de « (.+) » réinitialisé.*$", "✅ Password of “$1” reset (their remembered sign-ins are revoked)."], ["^✅ « (.+) » est maintenant (administrateur|utilisateur)\\.$", "✅ “$1” role updated."], ["^✅ Compte « (.+) » et toutes ses données supprimés\\.$", "✅ Account “$1” and all its data deleted."], ["^✅ Sauvegarde créée \\((.+)\\)\\.$", "✅ Backup created ($1)."], ["^✅ (\\d+) connexion\\(s\\) mémorisée\\(s\\) expirée\\(s\\) purgée\\(s\\)\\.$", "✅ $1 expired remembered sign-in(s) purged."], ["^✅ Journal allégé : (\\d+) ligne\\(s\\) ancienne\\(s\\) supprimée\\(s\\)\\.$", "✅ Log trimmed: $1 old line(s) removed."], ["^✅ Index et cours rechargés.*$", "✅ Indexes and courses will be reloaded from disk on each user's next action."], ["^⏳ Génération des flashcards… \\((\\d+) s\\)$", "⏳ Generating flashcards… ($1 s)"], ["^⏳ Génération du quiz… \\((\\d+) s\\)$", "⏳ Generating the quiz… ($1 s)"], ["^⏳ Génération du résumé… \\((\\d+) s\\)$", "⏳ Generating the summary… ($1 s)"], ["^C'est parti (.+) !$", "Let's go $1!"], ["^Bonjour (.+) 👋$", "Hello $1 👋"], ["^⚠️ Mot de passe trop court \\((\\d+) caractères minimum\\)\\.$", "⚠️ Password too short ($1 characters minimum)."]].map(([re, to]) => [new RegExp(re), to]);
  I18N.push(["Plan de révision", "Revision plan"],["Plan de révision intelligent", "Smart revision plan"],["Date de l'examen", "Exam date"],["Intitulé (facultatif)", "Title (optional)"],["Portée du plan", "Plan scope"],["Minutes de révision par jour", "Revision minutes per day"],["Générer / mettre à jour le plan", "Generate / update plan"],["Supprimer le plan", "Delete plan"],["Mes modules · espace semestriel", "My modules · semester space"],["Nouveau module", "New module"],["Semestre", "Semester"],["Créer le module", "Create module"],["Module du cours actif", "Active course module"],["Supprimer ce module", "Delete this module"],["Chercher dans", "Search in"],["Ce cours", "This course"],["Mon module", "My module"],["Tout mon espace", "My whole space"],["Mon historique de révision", "My revision history"],["Conversation du cours actif", "Active course conversation"],["Aujourd'hui", "Today"],["Demain", "Tomorrow"],["Hier", "Yesterday"],["Jour de repos : laissez la mémoire consolider.", "Rest day: let your memory consolidate."],["Jour de l'examen", "Exam day"],["Fait", "Done"],["Commencer", "Start"],["Jour J", "D-Day"],["de préparation estimée", "estimated readiness"],["page(s) faible(s) à relire", "weak page(s) to reread"],["min de révision cette semaine", "revision min this week"],["jour(s) d'affilée", "day streak"],["jour(s) étudié(s) sur 30", "day(s) studied out of 30"],["séance(s) au total", "session(s) in total"],["moyenne des quiz", "quiz average"],["14 derniers jours", "Last 14 days"],["Mes lacunes (notions les moins réussies aux quiz)", "My gaps (concepts least mastered in quizzes)"],["Votre journal de révision commence ici.", "Your revision journal starts here."],["Indiquez la date de votre examen.", "Enter your exam date."],["Sans module", "No module"],["Aucun module", "No module"],["Tous mes cours", "All my courses"],["Jour de repos", "Rest day"]);
  I18N.push(["Mes dossiers et modules", "My folders and modules"],["Importez vos PDF (même scannés) et rangez-les par module et par semestre.", "Import your PDFs (even scanned ones) and sort them by module and semester."],["Ajoutez un ou plusieurs PDF : AKORI lit le texte, lance l'OCR si la page est une image, puis indexe chaque page.", "Add one or more PDFs: AKORI reads the text, runs OCR if a page is an image, then indexes every page."],["Créez un module par matière (ex. Réseaux, Algorithmique) et choisissez son semestre.", "Create a module per subject (e.g. Networks, Algorithms) and pick its semester."],["Rangez directement vos nouveaux fichiers dans un module au moment de l'import.", "File your new documents straight into a module when you import them."],["Un semestre contient souvent plusieurs cours : un module les réunit pour les interroger ensemble.", "A semester often holds several courses: a module groups them so you can query them together."],["Importer un PDF", "Import a PDF"],["Les points clés du cours en un coup d'œil, dans votre langue.", "The key points of the course at a glance, in your language."],["Choisissez le cours actif en haut de la page.", "Pick the active course at the top of the page."],["Lancez « Générer le résumé » : le texte est conservé avec le cours.", "Click “Generate summary”: the text is saved with the course."],["Relisez-le la veille d'un examen, ou au début d'une séance.", "Reread it the day before an exam, or at the start of a session."],["Le résumé revient dans votre plan de révision comme première étape.", "The summary comes back in your revision plan as the first step."],["Générer le résumé", "Generate summary"],["Posez vos questions : la réponse s'appuie sur votre cours, avec la page citée.", "Ask your questions: the answer relies on your course, with the page cited."],["Choisissez où chercher : ce cours, tout un module, ou tout votre espace.", "Choose where to search: this course, a whole module, or your whole space."],["Lisez la réponse et ouvrez « Sources » pour voir les passages et les pages utilisés.", "Read the answer and open “Sources” to see the passages and pages used."],["Épinglez une réponse dans « À revoir », ou écoutez-la à voix haute.", "Pin an answer to “To review”, or listen to it aloud."],["Si l'information n'est pas dans vos cours, AKORI le dit au lieu d'inventer.", "If the information is not in your courses, AKORI says so instead of making it up."],["Poser une question", "Ask a question"],["Au moins 7 cartes en mode roulette pour mémoriser activement.", "At least 7 cards in roulette mode to memorize actively."],["Générez les cartes à partir du cours actif.", "Generate the cards from the active course."],["Cliquez pour stopper la roulette, essayez de répondre, puis affichez la réponse.", "Click to stop the roulette, try to answer, then reveal the answer."],["Marquez « Je savais » ou « À revoir » : les cartes à revoir reviennent dans votre plan.", "Mark “I knew it” or “To review”: cards to review come back in your plan."],["Mieux vaut 10 minutes chaque jour qu'une heure la veille : c'est la répétition espacée.", "10 minutes every day beat one hour the night before: that is spaced repetition."],["Lancer la roulette", "Spin the roulette"],["Testez-vous : correction expliquée, points forts et points faibles.", "Test yourself: explained corrections, strengths and weaknesses."],["Générez un quiz de 5 questions.", "Generate a 5-question quiz."],["Répondez, validez, lisez l'explication, puis passez à la suite.", "Answer, validate, read the explanation, then move on."],["Vos erreurs alimentent la carte de vos lacunes et les pages à relire.", "Your mistakes feed your gap map and the pages to reread."],["Un quiz raté n'est pas un échec : il indique exactement quelle page relire.", "A failed quiz is not a failure: it tells you exactly which page to reread."],["Passer un quiz", "Take a quiz"],["Répondez à l'oral ou à l'écrit, comme devant un examinateur.", "Answer aloud or in writing, as in front of an examiner."],["AKORI choisit un passage de votre cours et pose une question ouverte.", "AKORI picks a passage from your course and asks an open question."],["Dictez votre réponse au micro ou écrivez-la.", "Dictate your answer to the microphone or type it."],["Recevez une note sur 10, ce qui manquait, et la page à relire.", "Get a mark out of 10, what was missing, and the page to reread."],["Dire la réponse à voix haute révèle ce que vous maîtrisez vraiment.", "Saying the answer aloud reveals what you really master."],["Passer une colle", "Take an oral"],["Indiquez la date de l'examen : AKORI construit le plan jour par jour.", "Enter the exam date: AKORI builds a day-by-day plan."],["Saisissez la date, la portée (un cours, un module, tout) et le temps disponible par jour.", "Enter the date, the scope (a course, a module, everything) and your daily time."],["Le plan mélange vos points faibles et une répétition espacée (1, 3, 7 puis 14 jours).", "The plan mixes your weak points with spaced repetition (1, 3, 7 then 14 days)."],["Il se recalcule après chaque séance et vous indique les pages à relire.", "It recalculates after every session and shows the pages to reread."],["Un jour de repos est prévu : la mémoire consolide aussi quand vous ne travaillez pas.", "A rest day is planned: memory also consolidates when you are not working."],["Créer mon plan", "Create my plan"],["Retrouvez les réponses épinglées, cours par cours.", "Find your pinned answers, course by course."],["Épinglez une réponse de l'assistant depuis le chat.", "Pin an assistant answer from the chat."],["Retrouvez-la ici, rangée par cours.", "Find it here, sorted by course."],["Relisez-la ou écoutez-la avant l'examen.", "Reread it or listen to it before the exam."],["C'est votre fiche de révision personnelle, écrite avec vos propres questions.", "It is your personal revision sheet, written with your own questions."],["Ouvrir À revoir", "Open To review"],["Un état clair : flashcards, quiz et colle orale, cours par cours.", "A clear status: flashcards, quizzes and oral practice, course by course."],["Consultez la vue globale, puis le détail du cours choisi.", "Check the global view, then the detail of the chosen course."],["Chaque pourcentage précise sur quoi il porte (cartes maîtrisées, quiz terminés…).", "Each percentage states what it measures (mastered cards, completed quizzes…)."],["Repérez les notions à renforcer.", "Spot the concepts to reinforce."],["La progression d'un cours est la moyenne de ses trois activités.", "A course's progress is the average of its three activities."],["Voir ma progression", "See my progress"],["Votre journal de révision et la carte de vos lacunes.", "Your revision journal and the map of your gaps."],["Chaque séance est notée : quiz, flashcards, colle orale, questions.", "Every session is logged: quizzes, flashcards, oral practice, questions."],["Suivez votre série de jours et vos notions les moins réussies.", "Follow your day streak and your least mastered concepts."],["L'assistant adapte ses explications à votre niveau et à vos résultats.", "The assistant adapts its explanations to your level and your results."],["Plus vous révisez, plus les réponses s'ajustent à vous.", "The more you revise, the better the answers fit you."],["Ouvrir l'historique", "Open history"],["Les réponses viennent de vos cours, et chacune indique sa page.", "Answers come from your courses, and each one shows its page."],["💡 Guide de révision : cliquez sur une fonction pour la découvrir en détail", "💡 Revision guide: click a feature to explore it in detail"],["AKORI transforme vos cours PDF en un espace de révision intelligent : résumé, flashcards roulette, quiz, colle orale, plan de révision et assistant RAG.", "AKORI turns your PDF courses into a smart revision space: summary, roulette flashcards, quiz, oral practice, revision plan and RAG assistant."],["Plan", "Plan"],["Assistant IA", "AI Assistant"],["Ajouter les nouveaux fichiers au module", "Add new files to the module"],["Sélectionnez un ou plusieurs PDF : extraction et indexation automatiques.", "Select one or more PDFs: automatic extraction and indexing."],["Un semestre contient souvent plusieurs cours. Regroupez-les par module (matière) et par semestre pour interroger simultanément plusieurs unités d'enseignement (ex. Systèmes d'exploitation, Réseaux, Algorithmique) au sein d'un même espace de travail : dans l'Assistant IA, choisissez « Mon module » ou « Tout mon espace ».", "A semester often holds several courses. Group them by module (subject) and semester to query several teaching units at once (e.g. Operating Systems, Networks, Algorithms) within one workspace: in the AI Assistant, choose “My module” or “My whole space”."],["AKORI suit vos interactions (quiz, flashcards, colle orale, questions) pour cartographier vos lacunes conceptuelles et adapter dynamiquement le niveau d'explication des réponses générées par l'assistant.", "AKORI tracks your interactions (quizzes, flashcards, oral practice, questions) to map your conceptual gaps and dynamically adapt the explanation level of the assistant's answers."]);
  I18N.push(["Générez un quiz de 5 questions.", "Generate a 5-question quiz."],["Touchez une réponse : la correction et l'explication s'affichent aussitôt, sans valider.", "Tap an answer: the correction and explanation appear instantly, no need to validate."],["Glissez vers la gauche pour passer à la question suivante. Vos erreurs alimentent la carte de vos lacunes.", "Swipe left to go to the next question. Your mistakes feed your gap map."],["La carte mentale du cours, d'un seul regard.", "The course mind map at a glance."],["Choisissez le cours actif puis lancez « Générer la mindmap ».", "Pick the active course then click “Generate mind map”."],["Le thème est au centre, les grandes notions autour, leurs détails au bout des branches.", "The theme is at the center, the main concepts around it, their details at the branch tips."],["Cliquez sur une branche pour la replier ou la déplier et réviser bloc par bloc.", "Click a branch to fold or unfold it and revise block by block."],["La carte est conservée avec le cours : vous la retrouvez à chaque visite.", "The map is saved with the course: you find it at every visit."],["Ouvrir la mindmap", "Open the mind map"],["Mindmap", "Mind map"],["Mindmap du cours", "Course mind map"],["À propos", "About"],["✦ Générer la mindmap", "✦ Generate mind map"],["Générer la mindmap", "Generate mind map"],["La carte mentale du cours actif : le thème au centre, les grandes notions autour, leurs détails au bout. Cliquez sur une branche pour la replier.", "The mind map of the active course: the theme at the center, main concepts around it, details at the tips. Click a branch to fold it."],["Touchez une réponse : la correction s'affiche aussitôt.", "Tap an answer: the correction appears instantly."],["Cliquez sur une branche pour la replier ou la déplier.", "Click a branch to fold or unfold it."],["Aucune mindmap pour ce cours.", "No mind map for this course."],["← Présentation d'AKORI", "← About AKORI"]);
  I18N.push(["Touchez une réponse : la correction s'affiche aussitôt. Glissez vers la gauche pour passer à la question suivante.", "Tap an answer: the correction appears instantly. Swipe left to go to the next question."]);
  I18N.push(["Mes documents", "My documents"],["Modules · semestres", "Modules · semesters"],["Ajouter des PDF", "Add PDFs"],["Ranger les nouveaux PDF dans le module", "Put new PDFs in the module"],["Nouveau module", "New module"],["Supprimer", "Delete"],["Corriger", "Check"],["Dicter", "Dictate"],["Nouvelle question", "New question"],["Votre réponse", "Your answer"],["Écrivez ou dictez votre réponse…", "Type or dictate your answer…"],["La correction s'affichera ici.", "The correction will appear here."],["Note sur 10, points manquants et page à relire.", "Mark out of 10, missing points and page to reread."],["✦ Générer le résumé", "✦ Generate summary"],["Les points clés du cours actif, conservés avec le document.", "Key points of the active course, saved with the document."],["Vue d'ensemble", "Overview"],["Cours actif", "Active course"],["Mon examen", "My exam"],["Cours concernés", "Courses included"],["Minutes par jour", "Minutes per day"],["Mon journal de révision", "My revision journal"],["Thème clair ou sombre.", "Light or dark theme."],["Interface et réponses de l'IA.", "Interface and AI answers."],["Que voulez-vous faire ?", "What do you want to do?"],["Touchez une fonction pour voir comment elle marche.", "Tap a feature to see how it works."],["Cours importés", "Imported courses"],["Passages indexés (cours actif)", "Indexed passages (active course)"],["Questions posées (cours actif)", "Questions asked (active course)"],["Progression du cours actif", "Active course progress"],["AKORI · Espace de révision", "AKORI · Revision space"],["Vos cours,", "Your courses,"],["votre réussite", "your success"],["Un assistant de révision qui ne répond qu'avec vos propres documents.", "A revision assistant that answers only from your own documents."],["E-mail", "Email"],["Entrez votre adresse e-mail", "Enter your email address"],["Votre mot de passe", "Your password"],["Continuer", "Continue"],["OU", "OR"],["Créer un compte", "Create an account"],["J'ai déjà un compte", "I already have an account"],["Découvrir AKORI", "Discover AKORI"],["Comptes récents", "Recent accounts"],["Autre compte", "Other account"],["Une adresse réelle que vous consultez.", "A real address you check."],["Retapez le mot de passe", "Retype the password"],["Vos documents restent dans votre espace personnel. AKORI ne répond qu'à partir de vos cours.", "Your documents stay in your personal space. AKORI answers only from your courses."]);
  I18N.push(["Aucune activité pour le moment.", "No activity yet."],["Vos quiz, flashcards, colles orales, questions et résumés apparaîtront ici.", "Your quizzes, flashcards, orals, questions and summaries will appear here."],["Posez votre question…", "Ask your question…"],["Résumé généré", "Summary generated"],["Carte mentale générée", "Mind map generated"],["Question à l'assistant", "Question to the assistant"]);
  I18N.push(["Titre du cours (facultatif)", "Course title (optional)"],["Ranger dans le module", "File into module"],["MON EXAMEN", "MY EXAM"],["Priorités du plan", "Plan priorities"],["Modifier le plan", "Edit plan"],["Voir calendrier complet", "See full calendar"],["Ajuster le temps", "Adjust time"],["Détails", "Details"],["Masquer", "Hide"],["Lancer la séance", "Start session"],["À faire", "To do"],["Planifiée", "Planned"],["Facultative", "Optional"],["En cours", "In progress"],["Fait", "Done"],["Temps par jour :", "Time per day:"],["Touchez un jour pour voir ses séances.", "Tap a day to see its sessions."],["Cours commencés", "Courses started"],["Séance faite", "Session done"],["Fermer", "Close"],["Séances", "Sessions"],["Pages", "Pages"],["Plan de révision", "Revision plan"],["Haute priorité", "High priority"],["Moyenne priorité", "Medium priority"],["Basse priorité", "Low priority"]);
  I18N.push(["ÉTAPE 1 : CONTEXTE", "STEP 1: CONTEXT"],["ÉTAPE 2 : CALENDRIER", "STEP 2: CALENDAR"],["ÉTAPE 3 : MATIÈRES", "STEP 3: SUBJECTS"],["ÉTAPE 4 : RYTHME", "STEP 4: PACE"],["ÉTAPE 5 : INTELLIGENCE", "STEP 5: INTELLIGENCE"],
    ["Bienvenue ! Dis-moi, c'est pour quoi ?", "Welcome! Tell me, what's this for?"],["Quel est le grand jour, et combien de temps par jour ?", "When is the big day, and how long per day?"],["Qu'est-ce que tu révises ?", "What are you studying?"],["Quelle intensité veux-tu ?", "How intense do you want to be?"],["Presque fini ! Choisis les fonctions IA.", "Almost there! Choose AI features."],
    ["Examen", "Exam"],["Contrôle continu", "Continuous assessment"],["Apprentissage libre", "Free learning"],["Heure de début", "Start time"],["Facultative en apprentissage libre (plan sur 4 semaines).", "Optional in free learning (4-week plan)."],
    ["Matières (vos PDF importés)", "Subjects (your imported PDFs)"],["Tapez pour rechercher une matière.", "Type to search a subject."],["Activités", "Activities"],["Notes (résumé et pages à relire)", "Notes (summary and pages to reread)"],["Colle orale", "Oral exam"],
    ["Intensité", "Intensity"],["Faible", "Low"],["Moyenne", "Medium"],["Élevée", "High"],["Jours de repos", "Rest days"],["Lun", "Mon"],["Mar", "Tue"],["Mer", "Wed"],["Jeu", "Thu"],["Ven", "Fri"],["Sam", "Sat"],["Dim", "Sun"],
    ["Fonctions IA", "AI features"],["Répétition espacée", "Spaced repetition"],["Priorisation IA", "AI prioritization"],["Alertes de surcharge", "Overload alerts"],
    ["Répétition espacée", "Spaced repetition"],[": chaque activité revient à intervalles croissants.", ": each activity comes back at growing intervals."],[": les matières et les pages les plus fragiles passent en premier.", ": the weakest subjects and pages come first."],[": AKORI signale les journées qui dépassent ton temps.", ": AKORI flags the days that go over your time."],
    ["‹ Précédent", "‹ Previous"],["Annuler", "Cancel"],["Suivant", "Next"],["Générer le plan final", "Generate the final plan"],["Modifier le plan", "Edit the plan"],
    ["⚠️ Choisis la date de ton examen.", "⚠️ Pick your exam date."],["⚠️ Choisis la date de ton contrôle.", "⚠️ Pick your test date."],["⚠️ Cette date est déjà passée : choisis une date à venir.", "⚠️ This date has passed: pick an upcoming date."],
    ["⚠️ Choisis au moins une matière parmi tes PDF.", "⚠️ Pick at least one subject from your PDFs."],["⚠️ Importe d'abord un PDF dans « Mes dossiers ».", "⚠️ Import a PDF in “My folders” first."],["⚠️ Garde au moins une activité (notes, flashcards, quiz ou colle orale).", "⚠️ Keep at least one activity (notes, flashcards, quiz or oral exam)."],["⚠️ Garde au moins un jour de révision dans la semaine.", "⚠️ Keep at least one revision day in the week."],
    ["Résumé", "Summary"],["Relecture", "Rereading"],["Revue finale", "Final review"],["Lancer la séance", "Start the session"],["Pourquoi cette séance ?", "Why this session?"],["Pages à relire", "Pages to reread"],["Quand", "When"],["Durée", "Duration"],["Statut", "Status"],
    ["Jour de repos", "Rest day"],["Journée libre", "Free day"],["Repos", "Rest"],["Le jour J", "The big day"],["Contrôle", "Test"],["Fin du plan", "End of plan"],["du plan", "of the plan"],["terminé", "completed"],["Supprimer", "Delete"],
    ["Laissez la mémoire consolider : aucune séance n'est prévue.", "Let your memory consolidate: no session is planned."],["Pas de nouvelle notion : relisez seulement votre résumé et vos flashcards « à revoir ».", "No new concepts: just reread your summary and your “to review” flashcards."],
    ["⚠️ Journée chargée : elle dépasse votre temps par jour.", "⚠️ Busy day: it goes over your daily time."],["Votre examen", "Your exam"],["Votre contrôle", "Your test"],["Mes révisions", "My revision"],["Plan sur 4 semaines", "4-week plan"],
    ["Aucune matière dans ce plan.", "No subject in this plan."],["La date du plan est dépassée.", "The plan date has passed."],
    ["Lire le résumé du cours", "Read the course summary"],["Relire les pages faibles", "Reread the weak pages"],["Relire les pages faibles (rappel)", "Reread the weak pages (reminder)"],["Générer puis faire une série de flashcards", "Generate then do a set of flashcards"],["Entretenir tes flashcards", "Keep up your flashcards"],["Faire un quiz pour mesurer ta base", "Take a quiz to measure your base"],["Faire un quiz d'entretien", "Take a maintenance quiz"],["Passer une première colle orale", "Take a first oral exam"],["Passer une colle orale d'entretien", "Take a maintenance oral exam"],
    ["Le résumé n'a pas encore été généré : c'est la base avant de s'entraîner.", "The summary hasn't been generated yet: it's the base before practicing."],["Première séance : elle mesure votre niveau de départ.", "First session: it measures your starting level."],["Résultats encore fragiles : la séance revient plus tôt.", "Results still fragile: the session comes back sooner."],["Répétition espacée : on revient sur ce cours avant de l'oublier.", "Spaced repetition: we come back to this course before you forget it."],["Dernière séance avant le jour J : on repasse uniquement sur l'essentiel.", "Last session before the big day: only the essentials."],
    ["Un assistant en 5 étapes construit votre emploi du temps de révision.", "A 5-step assistant builds your revision timetable."],["Contexte et calendrier : examen, contrôle continu ou apprentissage libre, la date et le temps par jour.", "Context and calendar: exam, continuous assessment or free learning, the date and the time per day."],["Matières et rythme : choisissez vos PDF, l'intensité, les jours de repos et les fonctions IA.", "Subjects and pace: choose your PDFs, the intensity, the rest days and the AI features."],["L'emploi du temps s'affiche semaine par semaine : touchez une séance pour voir son détail et la lancer.", "The timetable shows week by week: tap a session to see its details and start it."]);
  PATTERNS.push(...[["^Semaine (\\d+) · (.+) · (\\d+) min$", "Week $1 · $2 · $3 min"], ["^Plan sur (\\d+) semaines$", "$1-week plan"], ["^(\\d+) matière\\(s\\)$", "$1 subject(s)"], ["^(\\d+) séance\\(s\\)$", "$1 session(s)"],
    ["^par jour · dès (.+)$", "per day · from $1"], ["^Rythme (.+) · repos : (.+)$", "Pace $1 · rest: $2"], ["^Répétition espacée : rappel n° (\\d+) pour ancrer la mémoire à long terme\\.$", "Spaced repetition: reminder no. $1 to anchor long-term memory."],
    ["^Pages repérées grâce à vos erreurs(.*)$", "Pages spotted from your mistakes$1"], ["^Revue finale : (.+)$", "Final review: $1"]].map(([re, to]) => [new RegExp(re), to]));
  I18N.push(["Tout est maîtrisé, bravo !", "Everything mastered, well done!"],["Reviens dans quelques jours pour ancrer ces notions sur la durée.", "Come back in a few days to lock these concepts in for good."],
    ["Très bon travail !", "Great work!"],["Encore quelques notions à consolider, et ce sera parfait.", "A few more concepts to consolidate and it will be perfect."],
    ["Tu progresses !", "You're making progress!"],["Revois les notions ratées puis recommence : ça va venir.", "Review the missed concepts and try again: you'll get there."],
    ["Pas encore ça… on continue ensemble ?", "Not quite yet… shall we keep going together?"],["Relis les pages indiquées, puis refais une série : chaque essai compte.", "Reread the pages shown, then do another round: every try counts."],
    ["Une question sur ce cours ?", "A question about this course?"],["AKORI répond à partir de ses pages et cite le passage utilisé.", "AKORI answers from its pages and cites the passage it used."],
    ["Aucun module pour le moment.", "No module yet."],["Importez vos PDF puis regroupez-les par module (matière) et par semestre.", "Import your PDFs, then group them by module (subject) and semester."],
    ["Aucun document pour le moment.", "No document yet."],["Ajoutez votre premier PDF pour commencer.", "Add your first PDF to get started."],["Aucun chapitre épinglé pour ce cours.", "No pinned chapter for this course."],
    ["La correction s'affichera ici.", "The correction will appear here."],["Note sur 10, points manquants et page à relire.", "Mark out of 10, missing points and page to reread."]);
  I18N.push(["Zoomez avec + et −, puis faites glisser la carte.", "Zoom with + and −, then drag the map."],["Ajuster", "Fit"],["Progression globale", "Global progress"],["Voir le détail par cours →", "See details by course →"]);
  PATTERNS.push(...[["^Moyenne de vos (\\d+) cours · 3 activités par cours : flashcards, quiz et colle orale\\.$", "Average of your $1 courses · 3 activities per course: flashcards, quiz and oral exam."],
    ["^Réponse en (.+)$", "Answered in $1"]].map(([re, to]) => [new RegExp(re), to]));
  const EXACT = new Map(I18N.map(([fr, en]) => [fr.trim(), en]));
  const codeOf = (label) => /English/.test(String(label)) ? 'en' : 'fr';
  const labelOf = (code) => code === 'en' ? 'English' : 'Français';
  window.__akoriLang = window.__akoriLang || 'fr';
  // Les emoji de début / fin de texte sont remplacés par de vraies icônes : la traduction doit donc aussi
  // fonctionner sur le texte « nu » (sans le symbole).
  const SYM = '\\s\\p{Extended_Pictographic}\\uFE0F✓✕✗✦↻↺';
  const SYM_LEAD = new RegExp('^[' + SYM + ']+', 'u');
  const SYM_TAIL = new RegExp('[' + SYM + ']+$', 'u');
  const stripSym = (t) => t.replace(SYM_LEAD, '').replace(SYM_TAIL, '');
  const EXACT2 = new Map();
  I18N.forEach(([fr, en]) => { const k = stripSym(fr); if (k) EXACT2.set(k, stripSym(en) || en); });
  const PATTERNS2 = PATTERNS.map(([re, to]) => {
    const src = re.source.replace(new RegExp('^\\^[' + SYM + ']+', 'u'), '^');
    return src !== re.source ? [new RegExp(src), to.replace(SYM_LEAD, '')] : null;
  }).filter(Boolean);
  const trCore = (t) => {
    if (EXACT.has(t)) return EXACT.get(t);
    for (const [re, to] of PATTERNS) if (re.test(t)) return t.replace(re, to);
    const bare = stripSym(t);
    if (bare) {
      if (EXACT2.has(bare)) return EXACT2.get(bare);
      for (const [re, to] of PATTERNS2) if (re.test(bare)) return bare.replace(re, to);
    }
    return null;
  };
  // ---- icônes (traits fins, héritent la couleur du texte)
  const SVG = {
    check: '<polyline points="20 6 9 17 4 12"/>',
    alert: '<path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    xcircle: '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
    x: '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    ban: '<circle cx="12" cy="12" r="10"/><line x1="4.9" y1="4.9" x2="19.1" y2="19.1"/>',
    info: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    spin: '<path d="M21 12a9 9 0 1 1-6.2-8.6"/>',
    refresh: '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.5 9a9 9 0 0 1 14.9-3.4L23 10M1 14l4.6 4.4A9 9 0 0 0 20.5 15"/>',
    sparkle: '<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/>',
    target: '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    trophy: '<path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0z"/><path d="M17 5h3v2a3 3 0 0 1-3 3M7 5H4v2a3 3 0 0 0 3 3"/>',
    pin: '<path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/>',
    clip: '<path d="M21.4 11.1l-9.2 9.2a6 6 0 0 1-8.5-8.5l9.2-9.2a4 4 0 0 1 5.7 5.7l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5"/>',
    sound: '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.5 8.5a5 5 0 0 1 0 7M19 5a10 10 0 0 1 0 14"/>',
    layers: '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
    bulb: '<path d="M9 18h6M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.3h6c0-1 .4-1.8 1-2.3A7 7 0 0 0 12 2z"/>',
    book: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    folder: '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>',
    shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    box: '<path d="M21 16V8l-9-5-9 5v8l9 5z"/><path d="M3.3 7L12 12l8.7-5M12 22V12"/>',
    key: '<path d="M21 2l-2 2M15.5 7.5L19 4M11.4 11.6a5.5 5.5 0 1 1-7.8 7.8 5.5 5.5 0 0 1 7.8-7.8zm0 0L15.5 7.5"/>',
    arrow: '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>',
    cap: '<path d="M22 10L12 5 2 10l10 5z"/><path d="M6 12v5c3 2 9 2 12 0v-5"/>',
    school: '<path d="M3 21h18M5 21V8l7-5 7 5v13M9 21v-6h6v6"/>',
    bolt: '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    chat: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    user: '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    dice: '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1"/><circle cx="15.5" cy="15.5" r="1"/><circle cx="12" cy="12" r="1"/>',
    mic: '<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8"/>',
    search: '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    trash: '<polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6m5 0V4a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v2"/>',
    lock: '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    play: '<polygon points="5 3 19 12 5 21 5 3"/>',
    trend: '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    clock: '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    send: '<line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>',
    mind: '<circle cx="5" cy="12" r="2.6"/><circle cx="19" cy="5" r="2.6"/><circle cx="19" cy="19" r="2.6"/><path d="M7.5 11l9-5M7.5 13l9 5"/>',
    cal: '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/>',
  };
  const ICON_MAP = {'✅': 'check', '✓': 'check', '⚠️': 'alert', '⚠': 'alert', '❌': 'xcircle', '✕': 'x', '✗': 'x', '⛔': 'ban',
    'ℹ️': 'info', '⏳': 'spin', '🔄': 'refresh', '🔁': 'refresh', '↻': 'refresh', '♻️': 'refresh', '✦': 'sparkle', '🎯': 'target',
    '🏆': 'trophy', '📌': 'pin', '📎': 'clip', '🔊': 'sound', '🧠': 'layers', '📝': 'edit', '💡': 'bulb', '📚': 'book', '📁': 'folder',
    '📂': 'folder', '🛡': 'shield', '🎰': 'dice', '📦': 'box', '🔑': 'key', '🚀': 'arrow', '🎉': 'sparkle', '🏫': 'school',
    '🎓': 'cap', '⚡': 'bolt', '💬': 'chat', '👤': 'user', '🤖': 'sparkle', '🔎': 'search', '🎤': 'mic', '🗑': 'trash',
    '🔒': 'lock', '▶': 'play', '👋': '', '📅': 'cal', '🗓': 'cal', '📖': 'book', '📈': 'trend', '🕑': 'clock', '🕸': 'mind', '🗑': 'trash', '➤': 'send'};
  const STATUS = {'✅': 'ok', '⚠️': 'err', '⚠': 'err', '❌': 'err', '⛔': 'err'};
  const EMO_ALT = Object.keys(ICON_MAP).sort((a, b) => b.length - a.length).map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
  const LEAD = new RegExp('^(\\s*)(' + EMO_ALT + ')\\uFE0F?\\s*([\\s\\S]*)$', 'u');
  const TRAIL = new RegExp('^([\\s\\S]*?)\\s*(' + EMO_ALT + ')\\uFE0F?\\s*$', 'u');
  const syncIcon = (node, side, key) => {
    const prop = side === 'L' ? '__akIcoL' : '__akIcoR';
    let el = node[prop];
    const kind = key ? ICON_MAP[key] : null;
    if (!kind) { if (el && el.parentNode) el.remove(); node[prop] = null; return; }
    if (!el || !el.isConnected) {
      if (!node.parentNode) return;
      el = document.createElement('span');
      node.parentNode.insertBefore(el, side === 'L' ? node : node.nextSibling);
      node[prop] = el;
    }
    el.className = 'ak-ico ' + (side === 'L' ? 'ak-ico-l' : 'ak-ico-r') + (kind === 'spin' ? ' ak-spin' : '');
    if (el.dataset.k !== kind) { el.dataset.k = kind; el.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true">' + SVG[kind] + '</svg>'; }
    if (STATUS[key]) el.dataset.s = STATUS[key]; else delete el.dataset.s;
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
    let fr = node.__akFr, lead = null, trail = null, pre = '';
    let m = fr.match(LEAD);
    if (m) { pre = m[1]; lead = m[2]; fr = m[3]; }
    m = fr.match(TRAIL);
    if (m && m[1].trim()) { fr = m[1]; trail = m[2]; }
    const out = tr(fr, window.__akoriLang);
    const value = (lead || trail ? pre : '') + (lead || trail ? out.replace(/^\s+/, '') : out);
    if (node.nodeValue !== value) node.nodeValue = value;
    node.__akLast = node.nodeValue;
    syncIcon(node, 'L', lead);
    syncIcon(node, 'R', trail);
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
    if (root.nodeType === 3) { if (root.parentNode && !SKIP.has(root.parentNode.tagName) && !root.parentNode.closest('svg')) translateTextNode(root); return; }
    if (root.nodeType !== 1 || SKIP.has(root.tagName)) return;
    translateAttrs(root);
    const tw = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    let n;
    while ((n = tw.nextNode())) {
      if (n.nodeType === 3) { if (n.parentNode && !SKIP.has(n.parentNode.tagName) && !n.parentNode.closest('svg')) translateTextNode(n); }
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
  let dark = true, pref = null;
  try {
    dark = localStorage.getItem('akori-theme') !== 'light';
    pref = localStorage.getItem('akori-sidebar');
  } catch (e) {}
  const small = window.innerWidth <= 900;
  const collapsed = window.innerWidth <= 700 ? false : (pref ? pref === 'closed' : small);
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
        const icons = el.querySelectorAll ? el.querySelectorAll(':scope > .ak-ico') : [];
        if (el.childElementCount && el.childElementCount !== icons.length) continue;
        const t = (el.textContent || '').trim();
        const st = icons.length ? icons[0].dataset.s : null;
        const want = !t || t.length > 400 ? null : (st === 'err' || /^[⚠❌⛔]/.test(t) ? 'ak-err' : (st === 'ok' || /^✅/.test(t) ? 'ak-ok' : null));
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
      if (window.innerWidth <= 900 && window.innerWidth > 700 && ev.target.closest('#sidebar .navbtn')) {
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
  document.body.classList.add('sb-instant');            // réduction / ouverture au clic : immédiate, sans animation
  window.__sbSuppress = true;                           // le curseur est encore sur le menu : il ne le rouvre pas avant d'en sortir
  document.body.classList.remove('sb-hover');
  window.setTimeout(() => document.body.classList.remove('sb-instant'), 420);
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
  // le texte du bouton est dans le dernier <span> (le premier peut être l'icône ajoutée par la traduction)
  const setLabel = (btn, text) => { const sp = [...btn.querySelectorAll(':scope > span:not(.ak-ico)')].pop(); if (sp) sp.textContent = text; };
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
    } else if (act === 'aedit' || act === 'adel') {
      sendHidden('admin-pick', 'admin-pick-btn', act + ':' + ((cls.match(/ak-u-(\d+)/) || [])[1] || '0'));
      window.setTimeout(() => { const card = document.getElementById('admin-card'); if (card) card.scrollIntoView({ behavior: 'smooth', block: 'center' }); }, 250);
    }
  });
  // Ouverture du menu réduit au survol (gérée ici pour pouvoir la suspendre juste après un clic sur ☰)
  document.addEventListener('mouseover', (e) => {
    const inside = !!(e.target.closest && e.target.closest('#sidebar'));
    if (!inside) window.__sbSuppress = false;
    document.body.classList.toggle('sb-hover', inside && !window.__sbSuppress);
  });
  document.documentElement.addEventListener('mouseleave', () => { window.__sbSuppress = false; document.body.classList.remove('sb-hover'); });
  // Cartes du guide / boutons « Commencer » : ouvrent directement la page correspondante
  document.addEventListener('click', (event) => {
    const go = event.target.closest('.ak-go');
    if (go) {
      const m = go.className.match(/ak-t-(\w+)/);
      const nav = m && document.getElementById('nav-' + m[1]);
      if (nav) { event.preventDefault(); nav.click(); window.scrollTo({ top: 0, behavior: 'smooth' }); }
      return;
    }
    if (event.target.closest('#user-badge')) { const p = document.getElementById('nav-profile'); if (p) p.click(); }
  });
  // Page active dans le menu : le bouton dont le libellé correspond à l'onglet affiché
  window.setInterval(() => {
    const t = document.querySelector('#main-tabs [role=tab][aria-selected=true]');
    if (!t) return;
    const txt = t.textContent.trim();
    document.querySelectorAll('#sidebar .navbtn').forEach((b) => {
      const on = (b.textContent || '').trim() === txt;
      if (on !== b.classList.contains('nav-active')) b.classList.toggle('nav-active', on);
    });
  }, 200);
  // Halo qui suit le curseur sur le bandeau d'accueil
  document.addEventListener('mousemove', (event) => {
    const hero = event.target.closest && event.target.closest('.hero');
    if (!hero) return;
    const r = hero.getBoundingClientRect();
    hero.style.setProperty('--mx', (event.clientX - r.left) + 'px');
    hero.style.setProperty('--my', (event.clientY - r.top) + 'px');
  });
  // Écran unique : hauteur disponible pour le corps de page (en-tête séparé)
  const fitScreen = () => {
    const mc = document.getElementById('main-column');
    if (mc) {
      const top = Math.max(0, mc.getBoundingClientRect().top + (window.scrollY || 0));
      document.body.style.setProperty('--mc-top', top + 'px');
    }
    const mt = document.getElementById('main-tabs');
    if (mt) document.body.style.setProperty('--body-h', mt.clientHeight + 'px');
    const active = document.querySelector('#sidebar .navbtn.nav-active');
    const title = document.getElementById('ak-page-title');
    if (active && title) {
      const t = (active.textContent || '').trim();
      if (t && title.textContent !== t) {
        title.textContent = t;
        const tab = document.getElementById('topbar-title');
        if (tab) { tab.classList.remove('tab-anim'); void tab.offsetWidth; tab.classList.add('tab-anim'); }
      }
    }
    document.body.classList.toggle('ak-narrow', (mt ? mt.clientWidth : window.innerWidth) < 760);
    document.body.classList.toggle('ak-short', window.innerHeight < 600);
    document.body.classList.toggle('ak-sbtight', window.innerHeight < 760);
    // Paramètres et Profil ne dépendent pas du cours : le sélecteur « Cours actif » est masqué
    document.body.classList.toggle('no-docsel', ['tab-settings', 'tab-profile'].some((id) => { const t = document.getElementById(id); return t && t.offsetParent !== null; }));
    const lw = document.getElementById('login-wrap');
    document.body.classList.toggle('on-login', !!(lw && lw.offsetParent !== null));
  };
  window.addEventListener('resize', fitScreen);
  window.setInterval(fitScreen, 300);
  // Guide de l'accueil : la fiche s'ouvre dans une fenêtre centrée, posée sur la page (jamais décalée par la carte)
  const gcClose = (modal) => {
    if (!modal || modal.classList.contains('out')) return;
    modal.classList.add('out');
    window.setTimeout(() => modal.remove(), 180);
  };
  document.addEventListener('click', (event) => {
    const sum = event.target.closest && event.target.closest('details.guide-card > summary');
    if (!sum) return;
    event.preventDefault();
    const card = sum.parentElement;
    const det = card.querySelector('.gc-detail');
    if (!det) return;
    document.querySelectorAll('body > .gc-modal').forEach((m) => m.remove());
    const modal = document.createElement('div');
    modal.className = 'gc-modal'; modal.setAttribute('role', 'dialog'); modal.setAttribute('aria-modal', 'true');
    const box = document.createElement('div'); box.className = 'gc-mbox';
    const head = document.createElement('div'); head.className = 'gc-mhead';
    const ico = card.querySelector('.gc-ico'); if (ico) head.appendChild(ico.cloneNode(true));
    const t = document.createElement('div'); t.className = 'gc-detail-title'; t.textContent = (card.querySelector('.gc-body b') || {}).textContent || '';
    head.appendChild(t);
    const x = document.createElement('button'); x.type = 'button'; x.className = 'gc-close'; x.setAttribute('aria-label', 'Fermer'); x.textContent = '✕';
    box.appendChild(x); box.appendChild(head);
    const body = document.createElement('div'); body.className = 'gc-mbody'; body.innerHTML = det.innerHTML;
    box.appendChild(body); modal.appendChild(box); document.body.appendChild(modal);
    const onKey = (e) => { if (e.key === 'Escape') { gcClose(modal); document.removeEventListener('keydown', onKey); } };
    document.addEventListener('keydown', onKey);
    modal.addEventListener('click', (e) => {
      if (e.target === modal || e.target.closest('.gc-close') || e.target.closest('.ak-go')) { gcClose(modal); document.removeEventListener('keydown', onKey); }
    });
    x.focus();
  }, true);
  // Mes dossiers : suppression d'un PDF après confirmation
  document.addEventListener('click', (event) => {
    const del = event.target.closest && event.target.closest('.doc-del');
    if (!del) return;
    event.preventDefault(); event.stopPropagation();
    const card = del.closest('.folder-card');
    const nameEl = card && card.querySelector('.folder-card-name');
    const name = nameEl ? (nameEl.getAttribute('title') || nameEl.textContent || '').trim() : '';
    if (!name) return;
    const en = window.__akoriLang === 'en';
    const old = document.getElementById('ak-confirm'); if (old) old.remove();
    const wrap = document.createElement('div'); wrap.id = 'ak-confirm';
    wrap.innerHTML = '<div class="box" role="dialog" aria-modal="true"><h3></h3><p></p><div class="row"><button type="button" class="no"></button><button type="button" class="danger yes"></button></div></div>';
    wrap.querySelector('h3').textContent = en ? 'Delete this PDF?' : 'Supprimer ce PDF ?';
    wrap.querySelector('p').textContent = (en ? '« ' + name + ' », its index, its history and its revision data will be erased. This cannot be undone.'
                                              : '« ' + name + ' », son index, son historique et ses révisions seront effacés. Cette action est définitive.');
    wrap.querySelector('.no').textContent = en ? 'Cancel' : 'Annuler';
    wrap.querySelector('.yes').textContent = en ? 'Delete' : 'Supprimer';
    document.body.appendChild(wrap);
    const close = () => wrap.remove();
    wrap.addEventListener('click', (e) => { if (e.target === wrap || e.target.closest('.no')) close(); });
    wrap.querySelector('.yes').addEventListener('click', () => { close(); sendHidden('del-doc-box', 'del-doc-btn', name); });
    document.addEventListener('keydown', function esc(e) { if (e.key === 'Escape') { close(); document.removeEventListener('keydown', esc); } });
    wrap.querySelector('.no').focus();
  }, true);
  // Connexion : onglets « Se connecter / Créer un compte » posés sur la carte
  document.addEventListener('click', (event) => {
    const tab = event.target.closest && event.target.closest('.auth-tab');
    if (!tab) return;
    const target = document.getElementById(tab.dataset.tab === 'register' ? 'go-register' : 'go-login');
    if (target) target.click();
  });
  // Bouton « Continuer » : indicateur d'attente dans le bouton, jusqu'à la réponse
  document.addEventListener('click', (event) => {
    const btn = event.target.closest && event.target.closest('#li-btn, #su-btn');
    if (!btn) return;
    btn.classList.add('busy');
    window.setTimeout(() => btn.classList.remove('busy'), 12000);
  });
  const authWatch = () => {
    const reg = document.getElementById('register-panel');
    const regOn = !!(reg && reg.offsetParent !== null);
    document.body.classList.toggle('auth-login', !regOn);
    document.querySelectorAll('.auth-tab').forEach((t) => t.classList.toggle('on', (t.dataset.tab === 'register') === regOn));
    ['login-status', 'register-status'].forEach((id) => {
      const st = document.getElementById(id);
      const msg = st && ((st.querySelector('.prose, .md') || {}).textContent || '').trim();
      if (st && st.__akTxt !== msg) { const had = st.__akTxt !== undefined; st.__akTxt = msg; if (had && msg) document.querySelectorAll('#li-btn, #su-btn').forEach((b) => b.classList.remove('busy')); }
    });
    if (!document.getElementById('login-wrap') || document.getElementById('login-wrap').offsetParent === null) document.querySelectorAll('#li-btn, #su-btn').forEach((b) => b.classList.remove('busy'));
    // Téléphone : le bouton d'import ouvre le gestionnaire de fichiers (PDF) et non la galerie
    document.querySelectorAll('#documents-row input[type=file], #add-document-tile input[type=file], input[type=file][accept*=".pdf"]').forEach((inp) => {
      if (inp.accept !== 'application/pdf,.pdf') inp.accept = 'application/pdf,.pdf';
    });
  };
  window.setInterval(authWatch, 250);
  // Progression : la vue d'ensemble ouvre le détail par cours (carrousel)
  const gpOpen = (src) => {
    const tpl = src.closest('.ak-pane, .tabitem, body').querySelector('.gp-modal');
    if (!tpl) return;
    const old = document.querySelector('body > .gp-modal'); if (old) old.remove();
    const modal = tpl.cloneNode(true); document.body.appendChild(modal);
    const cards = [...modal.querySelectorAll('.gp-card')];
    const pager = [...modal.querySelectorAll('.gp-pager i')];
    const at = ((src.closest('.gp-open') || src).className.match(/gp-at-(\d+)/) || [])[1];
    let cur = at && +at < cards.length ? +at : 0;
    const show = () => {
      const n = cards.length;
      cards.forEach((c, i) => {
        c.classList.remove('pos0', 'posL', 'posR');
        if (i === cur) c.classList.add('pos0');
        else if (n > 1 && i === (cur - 1 + n) % n) c.classList.add('posL');
        else if (n > 2 && i === (cur + 1) % n) c.classList.add('posR');
        else if (n === 2 && i !== cur) c.classList.add('posR');
      });
      pager.forEach((p, i) => p.classList.toggle('on', i === cur));
    };
    const go = (d) => { cur = (cur + d + cards.length) % cards.length; show(); };
    const close = () => { modal.remove(); document.removeEventListener('keydown', onKey); };
    const onKey = (e) => { if (e.key === 'Escape') close(); if (e.key === 'ArrowLeft') go(-1); if (e.key === 'ArrowRight') go(1); };
    document.addEventListener('keydown', onKey);
    modal.addEventListener('click', (e) => {
      if (e.target.closest('.gp-close') || e.target === modal) return close();
      if (e.target.closest('.gp-prev') || e.target.closest('.gp-card.posL')) return go(-1);
      if (e.target.closest('.gp-next') || e.target.closest('.gp-card.posR')) return go(1);
    });
    let sx = null;
    modal.addEventListener('pointerdown', (e) => { sx = e.clientX; });
    modal.addEventListener('pointerup', (e) => { if (sx === null) return; const dx = e.clientX - sx; sx = null; if (Math.abs(dx) > 60) go(dx < 0 ? 1 : -1); });
    show();
    const closeBtn = modal.querySelector('.gp-close'); if (closeBtn) closeBtn.focus();
  };
  document.addEventListener('click', (e) => { const h = e.target.closest && e.target.closest('.gp-open'); if (h) gpOpen(h); });
  document.addEventListener('keydown', (e) => { if ((e.key === 'Enter' || e.key === ' ') && e.target.closest && e.target.closest('.gp-open')) { e.preventDefault(); gpOpen(e.target); } });
  // Session perdue (serveur redémarré, page restée ouverte) : on recharge une fois pour se reconnecter automatiquement
  window.setInterval(() => {
    const txt = [...document.querySelectorAll('#upload-status, #plan-status, #login-status, .ak-status, [id$="-status"]')].map((e) => e.textContent || '').join(' ');
    if (!/Session expirée|Session expired/.test(txt)) return;
    let last = 0; try { last = +(sessionStorage.getItem('akori-reloaded') || 0); } catch (e) {}
    if (Date.now() - last < 60000) return;
    try { sessionStorage.setItem('akori-reloaded', String(Date.now())); } catch (e) {}
    window.setTimeout(() => location.reload(), 1500);
  }, 700);
  // Plan de révision : emploi du temps intelligent (semaine à l'écran, détail d'une séance ou d'un jour au toucher)
  const edtShow = (edt, w) => {
    const weeks = [...edt.querySelectorAll('.edt-week')];
    if (!weeks.length) return;
    w = Math.max(0, Math.min(weeks.length - 1, w));
    weeks.forEach((x, i) => x.classList.toggle('on', i === w));
    edt.dataset.cur = String(w);
    const lbl = edt.querySelector('.edt-wl'); if (lbl) lbl.textContent = weeks[w].dataset.label || '';
    const p = edt.querySelector('.edt-prev'), n = edt.querySelector('.edt-next');
    if (p) p.disabled = w === 0; if (n) n.disabled = w === weeks.length - 1;
  };
  const edtModal = (tpl) => {
    const old = document.querySelector('body > .edt-modal'); if (old) old.remove();
    const modal = document.createElement('div');
    modal.className = 'edt-modal'; modal.setAttribute('role', 'dialog'); modal.setAttribute('aria-modal', 'true');
    const box = document.createElement('div'); box.className = 'edt-mbox';
    const x = document.createElement('button'); x.type = 'button'; x.className = 'edt-mclose'; x.setAttribute('aria-label', 'Fermer'); x.textContent = '✕';
    box.appendChild(x); box.appendChild(tpl.content.cloneNode(true)); modal.appendChild(box);
    if (document.body.classList.contains('akori-dark')) modal.classList.add('dark');
    document.body.appendChild(modal);
    const close = () => { modal.remove(); document.removeEventListener('keydown', onKey); };
    const onKey = (e) => { if (e.key === 'Escape') close(); };
    document.addEventListener('keydown', onKey);
    modal.addEventListener('click', (e) => {
      if (e.target === modal || e.target.closest('.edt-mclose')) return close();
      const go = e.target.closest('.pl-go');
      if (go) { close(); planGo(go); }
    });
    x.focus();
  };
  const planGo = (go) => {
    const m = go.className.match(/ak-t-(\w+)/);
    const doc = go.dataset.doc || '';
    const sel = document.querySelector('#doc-select input');
    if (doc && !(sel && sel.value === doc)) sendHidden('plan-go-box', 'plan-go-btn', doc);
    const nav = m && document.getElementById('nav-' + m[1]);
    window.setTimeout(() => { if (nav) nav.click(); window.scrollTo({ top: 0, behavior: 'smooth' }); }, doc ? 260 : 0);
  };
  document.addEventListener('click', (e) => {
    const t = e.target.closest ? e.target : null;
    if (!t) return;
    if (t.closest('.edt-edit')) { const b = document.getElementById('plan-edit-btn'); if (b) b.click(); return; }
    const edt = t.closest('.edt');
    if (!edt) return;
    if (t.closest('.edt-del')) {
      const en = window.__akoriLang === 'en';
      if (window.confirm(en ? 'Delete this revision plan?' : 'Supprimer ce plan de révision ?')) { const b = document.getElementById('plan-clear'); if (b) b.click(); }
      return;
    }
    if (t.closest('.edt-prev')) return edtShow(edt, (+edt.dataset.cur || 0) - 1);
    if (t.closest('.edt-next')) return edtShow(edt, (+edt.dataset.cur || 0) + 1);
    const cell = t.closest('.edt-blk, .edt-dh, .edt-exam');
    const tpl = cell && cell.querySelector(':scope > template');
    if (tpl) edtModal(tpl);
  });
  // Glisser horizontalement sur l'emploi du temps : semaine précédente / suivante
  let edtX = null, edtY = null;
  document.addEventListener('pointerdown', (e) => { const w = e.target.closest && e.target.closest('.edt-weeks'); edtX = w ? e.clientX : null; edtY = e.clientY; });
  document.addEventListener('pointerup', (e) => {
    if (edtX === null) return;
    const dx = e.clientX - edtX, dy = e.clientY - edtY; edtX = null;
    const edt = e.target.closest && e.target.closest('.edt');
    if (edt && Math.abs(dx) > 70 && Math.abs(dy) < 60) edtShow(edt, (+edt.dataset.cur || 0) + (dx < 0 ? 1 : -1));
  });
  // Comptes récents sur l'écran de connexion (comme Messenger)
  const ACC_KEY = 'akori-accounts';
  const accLoad = () => { try { return JSON.parse(localStorage.getItem(ACC_KEY) || '[]'); } catch (e) { return []; } };
  const accSave = (list) => { try { localStorage.setItem(ACC_KEY, JSON.stringify(list.slice(0, 4))); } catch (e) {} };
  const accRemember = () => {
    const badge = document.querySelector('#user-badge .user-badge');
    const mail = badge && badge.querySelector('.ub-mail');
    if (!mail) return;
    const email = (mail.textContent || '').trim();
    const name = ((badge.querySelector('.ub-text b') || {}).textContent || email).trim();
    const img = badge.querySelector('img.ub-avatar');
    const entry = { email, name, avatar: img ? img.getAttribute('src') : '' };
    const list = accLoad().filter((a) => a.email !== email);
    const prev = accLoad().find((a) => a.email === email);
    if (prev && prev.name === entry.name && prev.avatar === entry.avatar && accLoad()[0] && accLoad()[0].email === email) return;
    list.unshift(entry); accSave(list);
  };
  const setField = (input, value) => {
    const proto = input.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  };
  const esc = (t) => String(t || '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const accRender = () => {
    const box = document.getElementById('ak-accounts');
    if (!box) return;
    const list = accLoad();
    const sig = JSON.stringify(list.map((a) => [a.email, a.name, (a.avatar || '').length]));
    if (box.dataset.sig === sig) return;
    box.dataset.sig = sig;
    if (!list.length) { box.innerHTML = ''; return; }
    const cards = list.map((a, i) => '<div class="acc" data-i="' + i + '"><button type="button" class="acc-x" title="Retirer">×</button>' +
      (a.avatar ? '<img class="acc-av" src="' + esc(a.avatar) + '" alt="">' : '<div class="acc-av">' + esc((a.name || a.email || '?')[0].toUpperCase()) + '</div>') +
      '<div class="acc-name">' + esc(a.name || a.email) + '</div></div>').join('');
    box.innerHTML = '<div class="acc-title">Comptes récents</div><div class="acc-row">' + cards +
      '<div class="acc acc-add"><div class="acc-av">+</div><div class="acc-name">Autre compte</div></div></div><div class="acc-pick"></div>';
  };
  document.addEventListener('click', (event) => {
    const acc = event.target.closest && event.target.closest('#ak-accounts .acc');
    if (!acc) return;
    const panel = document.getElementById('login-panel');
    const user = document.querySelector('#li-user input, #li-user textarea');
    const pass = document.querySelector('#li-pass input');
    const list = accLoad();
    if (event.target.closest('.acc-x')) {
      const i = +acc.dataset.i; list.splice(i, 1); accSave(list);
      if (panel) panel.classList.remove('picked'); accRender(); return;
    }
    document.querySelectorAll('#ak-accounts .acc').forEach((x) => x.classList.toggle('on', x === acc));
    if (acc.classList.contains('acc-add')) {
      if (panel) panel.classList.remove('picked');
      if (user) { setField(user, ''); user.focus(); }
      return;
    }
    const a = list[+acc.dataset.i];
    if (!a || !user) return;
    setField(user, a.email);
    const pick = document.querySelector('#ak-accounts .acc-pick');
    if (pick) pick.innerHTML = 'Continuer en tant que <b>' + esc(a.name || a.email) + '</b>';
    if (panel) panel.classList.add('picked');
    if (pass) window.setTimeout(() => pass.focus(), 60);
  });
  window.setInterval(() => { accRemember(); accRender(); }, 700);
  // Téléphone : menu en tiroir (bouton ☰ flottant), pages pleine largeur
  const mq = window.matchMedia('(max-width: 700px)');
  const applyMode = () => {
    document.body.classList.toggle('m-mode', mq.matches);
    if (mq.matches) document.body.classList.remove('sidebar-collapsed'); else document.body.classList.remove('m-nav-open');
  };
  if (mq.addEventListener) mq.addEventListener('change', applyMode); else mq.addListener(applyMode);
  applyMode();
  if (!document.getElementById('m-burger')) {
    const burger = document.createElement('button');
    burger.id = 'm-burger'; burger.type = 'button'; burger.setAttribute('aria-label', 'Menu');
    burger.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>';
    const back = document.createElement('div'); back.id = 'm-backdrop';
    document.body.appendChild(burger); document.body.appendChild(back);
    burger.addEventListener('click', () => document.body.classList.toggle('m-nav-open'));
    back.addEventListener('click', () => document.body.classList.remove('m-nav-open'));
    document.addEventListener('click', (e) => { if (e.target.closest && e.target.closest('#sidebar .navbtn')) window.setTimeout(() => document.body.classList.remove('m-nav-open'), 120); });
  }
  window.setInterval(() => {
    document.body.classList.toggle('in-app', !!document.getElementById('sidebar'));
    if (mq.matches) document.body.classList.remove('sidebar-collapsed');
  }, 400);
  // Mindmap : zoom (+, −, ajuster, Ctrl + molette, pincement à deux doigts) et déplacement en faisant glisser
  const mmSet = (sc, z, cx, cy) => {
    const old = +sc.dataset.z || 1;
    z = Math.max(1, Math.min(5, z));
    const rx = (sc.scrollLeft + (cx ?? sc.clientWidth / 2)) / (sc.scrollWidth || 1);
    const ry = (sc.scrollTop + (cy ?? sc.clientHeight / 2)) / (sc.scrollHeight || 1);
    sc.style.setProperty('--mmz', z); sc.dataset.z = z;
    sc.scrollLeft = rx * sc.scrollWidth - (cx ?? sc.clientWidth / 2);
    sc.scrollTop = ry * sc.scrollHeight - (cy ?? sc.clientHeight / 2);
    const lbl = sc.closest('.mm-wrap').querySelector('.mm-zl'); if (lbl) lbl.textContent = Math.round(z * 100) + ' %';
    sc.classList.toggle('zoomed', z > 1.01);
    return old;
  };
  document.addEventListener('click', (e) => {
    const zb = e.target.closest && e.target.closest('.mm-z');
    if (!zb) return;
    const sc = zb.closest('.mm-wrap').querySelector('.mm-scroll');
    const z = +sc.dataset.z || 1;
    mmSet(sc, zb.dataset.z === 'in' ? z * 1.3 : (zb.dataset.z === 'out' ? z / 1.3 : 1));
  });
  document.addEventListener('wheel', (e) => {
    const sc = e.target.closest && e.target.closest('.mm-scroll');
    if (!sc || !(e.ctrlKey || e.metaKey)) return;
    e.preventDefault();
    const r = sc.getBoundingClientRect();
    mmSet(sc, (+sc.dataset.z || 1) * (e.deltaY < 0 ? 1.12 : 1 / 1.12), e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });
  let mmDrag = null; const mmTouches = new Map();
  document.addEventListener('pointerdown', (e) => {
    const sc = e.target.closest && e.target.closest('.mm-scroll');
    if (!sc) return;
    window.__mmDragged = false;
    mmTouches.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (mmTouches.size === 2) {
      const [a, b] = [...mmTouches.values()];
      mmDrag = { sc, pinch: Math.hypot(a.x - b.x, a.y - b.y), z: +sc.dataset.z || 1 };
    } else {
      mmDrag = { sc, x: e.clientX, y: e.clientY, l: sc.scrollLeft, t: sc.scrollTop };
    }
  });
  document.addEventListener('pointermove', (e) => {
    if (!mmDrag) return;
    if (mmTouches.has(e.pointerId)) mmTouches.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const sc = mmDrag.sc;
    if (mmDrag.pinch && mmTouches.size === 2) {
      const [a, b] = [...mmTouches.values()];
      const r = sc.getBoundingClientRect();
      mmSet(sc, mmDrag.z * Math.hypot(a.x - b.x, a.y - b.y) / mmDrag.pinch, (a.x + b.x) / 2 - r.left, (a.y + b.y) / 2 - r.top);
      window.__mmDragged = true; return;
    }
    if (mmDrag.pinch) return;
    const dx = e.clientX - mmDrag.x, dy = e.clientY - mmDrag.y;
    if (Math.abs(dx) + Math.abs(dy) > 5) { window.__mmDragged = true; sc.classList.add('drag'); }
    sc.scrollLeft = mmDrag.l - dx; sc.scrollTop = mmDrag.t - dy;
  });
  const mmEnd = (e) => {
    mmTouches.delete(e.pointerId);
    if (mmDrag && mmTouches.size === 0) { mmDrag.sc.classList.remove('drag'); mmDrag = null; window.setTimeout(() => { window.__mmDragged = false; }, 0); }
  };
  document.addEventListener('pointerup', mmEnd); document.addEventListener('pointercancel', mmEnd);
  // Chat : heure relative des messages (« à l'instant », « il y a 5 min », « il y a 2 h », « hier à 21:40 »)
  const relTime = (iso) => {
    const d = new Date(iso); if (isNaN(d)) return null;
    const en = window.__akoriLang === 'en';
    const s = (Date.now() - d.getTime()) / 1000;
    const hm = d.toLocaleTimeString(en ? 'en-GB' : 'fr-FR', { hour: '2-digit', minute: '2-digit' });
    if (s < 60) return en ? 'just now' : "à l'instant";
    if (s < 3600) { const m = Math.floor(s / 60); return en ? `${m} min ago` : `il y a ${m} min`; }
    const y = new Date(); y.setDate(y.getDate() - 1);
    if (s < 6 * 3600 || d.toDateString() === new Date().toDateString()) { const h = Math.floor(s / 3600); return en ? `${h} h ago` : `il y a ${h} h`; }
    if (d.toDateString() === y.toDateString()) return en ? `yesterday at ${hm}` : `hier à ${hm}`;
    return d.toLocaleDateString(en ? 'en-GB' : 'fr-FR', { day: '2-digit', month: '2-digit' }) + (en ? ' at ' : ' à ') + hm;
  };
  const tickTimes = () => document.querySelectorAll('.msg-time[class*="ts-"]').forEach((el) => {
    const m = el.className.match(/ts-(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/);
    if (!m) return;
    const t = relTime(`${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}`); if (t && el.textContent !== t) el.textContent = t;
  });
  window.setInterval(tickTimes, 2000);
  // Page de présentation : apparition au défilement, ancres, bouton « Se connecter »
  const lrIO = 'IntersectionObserver' in window ? new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); lrIO.unobserve(e.target); } }), { threshold: .12 }) : null;
  const scanLr = () => document.querySelectorAll('.lr:not(.in):not([data-lr])').forEach((el) => { el.dataset.lr = '1'; if (lrIO) lrIO.observe(el); else el.classList.add('in'); });
  new MutationObserver(scanLr).observe(document.body, { childList: true, subtree: true });
  scanLr();
  document.addEventListener('click', (event) => {
    const login = event.target.closest && event.target.closest('.ld-login');
    if (login) { const b = document.getElementById('land-login-btn'); if (b) { b.click(); window.scrollTo({ top: 0 }); } return; }
    const anchor = event.target.closest && event.target.closest('.ld-links a, .ld a[href^="#ld-"]');
    if (anchor) { event.preventDefault(); const t = document.getElementById(anchor.getAttribute('href').slice(1)); if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
    const br = event.target.closest && event.target.closest('.mm-bnode');
    if (br && !window.__mmDragged) br.closest('.mm-br').classList.toggle('mm-closed');
    const opt = event.target.closest && event.target.closest('button.opt-click');
    if (opt) { const m = opt.className.match(/ak-o-(\d+)/); if (m) { opt.classList.add('selected'); sendHidden('quiz-pick-box', 'quiz-pick-btn', m[1]); } return; }
    if (event.target.closest && event.target.closest('.ak-quiz-next')) quizNext();
  });
  // Quiz : glisser vers la gauche (ou flèche droite) pour passer à la suite une fois la correction affichée
  const quizNext = () => {
    const card = document.querySelector('.quiz-card');
    const next = document.getElementById('quiz_next');
    if (!card || !next || !card.querySelector('.quiz-verdict')) return;
    card.classList.add('swipe-out');
    window.setTimeout(() => next.click(), 190);
  };
  let swX = null, swY = null;
  document.addEventListener('pointerdown', (event) => {
    const card = event.target.closest && event.target.closest('.quiz-card');
    if (!card || event.target.closest('button')) { swX = null; return; }
    swX = event.clientX; swY = event.clientY;
  });
  document.addEventListener('pointerup', (event) => {
    if (swX === null) return;
    const dx = event.clientX - swX, dy = event.clientY - swY;
    swX = null;
    if (dx < -70 && Math.abs(dy) < 70) quizNext();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowRight' && document.querySelector('.quiz-card .quiz-verdict') && !/INPUT|TEXTAREA/.test((document.activeElement || {}).tagName || '')) quizNext();
  });
  // Compteurs animés (0 -> valeur) à l'apparition des chiffres
  const countUp = (el) => {
    if (el.__cu || !el.firstChild || el.firstChild.nodeType !== 3) return;
    const m = (el.textContent || '').trim().match(/^(\d+)(\s*%?)$/);
    el.__cu = true;
    if (!m || +m[1] === 0) return;
    const to = +m[1], suffix = m[2], node = el.firstChild, t0 = performance.now();
    const step = (t) => {
      const k = Math.min(1, (t - t0) / 800);
      node.nodeValue = Math.round(to * (1 - Math.pow(1 - k, 3))) + suffix;
      if (k < 1) requestAnimationFrame(step);
    };
    node.nodeValue = '0' + suffix;
    requestAnimationFrame(step);
  };
  const scanCounters = (root) => {
    if (!root || root.nodeType !== 1) return;
    if (root.classList && root.classList.contains('cu')) countUp(root);
    if (root.querySelectorAll) root.querySelectorAll('.cu').forEach(countUp);
  };
  new MutationObserver((muts) => muts.forEach((m) => m.addedNodes.forEach(scanCounters))).observe(document.body, { childList: true, subtree: true });
  scanCounters(document.body);
  // Champs de connexion : pas de suggestions de saisie automatique du navigateur (plus de surbrillance gênante)
  const hardenAuth = () => {
    [['li-user', 'off'], ['su-user', 'off'], ['li-pass', 'new-password'], ['su-pass', 'new-password'], ['su-pass2', 'new-password']].forEach(([id, ac]) => {
      const box = document.getElementById(id);
      const input = box && box.querySelector('input, textarea');
      if (!input || input.__akHard) return;
      input.__akHard = true;
      input.setAttribute('autocomplete', ac);
      input.setAttribute('name', 'ak_' + id + '_' + Math.random().toString(36).slice(2, 7));
      input.setAttribute('autocapitalize', 'off'); input.setAttribute('autocorrect', 'off'); input.setAttribute('spellcheck', 'false');
      input.setAttribute('data-lpignore', 'true'); input.setAttribute('data-form-type', 'other');
    });
  };
  new MutationObserver(hardenAuth).observe(document.body, { childList: true, subtree: true });
  hardenAuth();
  // Date d'examen et heure de début : sélecteurs natifs placés devant le champ texte
  // (le champ texte, masqué, reste la source de vérité côté serveur).
  const nativePicker = (id, type, setup) => {
    const box = document.getElementById(id);
    const area = box && box.querySelector('textarea, input:not(.ak-datepick)');
    if (!area || area.__akPick) return;
    area.__akPick = true;
    const pick = document.createElement('input');
    pick.type = type; pick.className = 'ak-datepick ak-pick-' + type;
    setup(pick);
    area.style.display = 'none';
    area.parentNode.insertBefore(pick, area);
    const push = () => {
      const setter = Object.getOwnPropertyDescriptor(area.constructor.prototype, 'value').set;
      setter.call(area, pick.value);
      area.dispatchEvent(new Event('input', { bubbles: true }));
      area.dispatchEvent(new Event('change', { bubbles: true }));
    };
    pick.addEventListener('input', push); pick.addEventListener('change', push);
    pick.value = area.value || '';
    setInterval(() => { if (document.activeElement !== pick && pick.value !== area.value) pick.value = area.value || ''; }, 300);
  };
  const pickers = () => {
    nativePicker('plan-date', 'date', (p) => { p.min = new Date().toISOString().slice(0, 10); });
    nativePicker('pw-start', 'time', (p) => { p.step = 300; p.required = true; });
  };
  new MutationObserver(pickers).observe(document.body, { childList: true, subtree: true });
  pickers();

  // Recherche instantanée dans la liste des utilisateurs
  document.addEventListener('input', (event) => {
    const box = event.target.closest && event.target.closest('#admin-search');
    if (!box) return;
    const q = (event.target.value || '').toLowerCase().replace(/[^\p{L}\p{N}@. _-]/gu, '').trim();
    document.querySelectorAll('#admin-users .ad-table tbody tr').forEach((tr) => {
      tr.style.display = !q || tr.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
  });
};
"""

# Gradio UI


# =============================================================================
# CONNEXION, SESSIONS ET CONSOLE D'ADMINISTRATION
# =============================================================================
def _avatar_html(key, css="ub-avatar"):
    """Photo de profil si elle existe, sinon l'initiale du prénom."""
    u = _users().get(key, {}) if key else {}
    name = _display_name(key) if key else ""
    photo = u.get("avatar") or ""
    if photo.startswith("data:image/"):
        return f"<img class='{css}' src='{photo}' alt='{_escape_html(name)}'>"
    return f"<div class='{css}'>{_escape_html(name[:1].upper() or '?')}</div>"


def _user_badge_html(key):
    if not key:
        return ""
    u = _users().get(key, {})
    name = _display_name(key)
    role = "Administrateur" if u.get("role") == "admin" else "Utilisateur"
    return (f"<div class='user-badge' title='Mon profil'>{_avatar_html(key)}"
            f"<div class='ub-text'><b>{_escape_html(name)}</b><span>{role}</span></div>"
            f"<span class='ub-mail'>{_escape_html(str(u.get('username', key)))}</span></div>")


# ---------------- profil : photo, prénom, niveau, objectif
PROF_LEVELS = [("Collège", "college"), ("Lycée", "lycee"), ("Études supérieures", "superieur")]
PROF_GOALS = [("Réviser mes cours", "revise"), ("Préparer un examen", "exam"), ("Mieux comprendre un chapitre", "understand")]


def _profile_card(key):
    if not key:
        return ""
    u = _users().get(key, {})
    photo = u.get("avatar") or ""
    name = _display_name(key)
    face = (f"<img class='prof-avatar' src='{photo}' alt=''>" if photo.startswith("data:image/")
            else f"<div class='prof-avatar'>{_escape_html(name[:1].upper() or '?')}</div>")
    return (f"<div class='prof-card'>{face}<div><b>{_escape_html(name)}</b><small>{_escape_html(str(u.get('username', key)))}</small>"
            f"<small>Ajoutez une photo pour personnaliser votre espace.</small></div></div>")


def profile_view():
    key = _CUR_USER.get()
    if not key:
        return "", "", None, None, ""
    p = _profile(key)
    return _profile_card(key), p["name"], p["level"] or None, p["goal"] or None, ""


def _make_avatar(path):
    """Photo -> carré 192 px en JPEG, sous forme d'adresse data: (petite, stockée avec le compte)."""
    import base64
    from PIL import Image, ImageOps   # Pillow est installé avec Gradio
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        side = min(im.size)
        left, top = (im.width - side) // 2, (im.height - side) // 2
        im = im.crop((left, top, left + side, top + side)).resize((192, 192))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=84, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def profile_photo_upload(file_path, doc_name=None):
    key = _CUR_USER.get()
    if not key:
        return gr.update(), gr.update(), "⚠️ Session expirée : rechargez la page pour vous reconnecter.", gr.update()
    path = getattr(file_path, "name", file_path)
    if not path:
        return gr.update(), gr.update(), "", gr.update()
    try:
        if os.path.getsize(path) > 8 * 1024 * 1024:
            return gr.update(), gr.update(), "⚠️ Image trop volumineuse (8 Mo maximum).", gr.update()
        _save_profile(key, avatar=_make_avatar(path))
    except Exception:
        return gr.update(), gr.update(), "⚠️ Image illisible : choisissez un fichier JPG ou PNG.", gr.update()
    _log("profile_photo")
    return _profile_card(key), _user_badge_html(key), "✅ Photo de profil enregistrée.", dashboard_html(doc_name)


def profile_photo_remove(doc_name=None):
    key = _CUR_USER.get()
    if key:
        _save_profile(key, avatar="")
    return _profile_card(key), _user_badge_html(key), "✅ Photo retirée.", dashboard_html(doc_name)


def profile_save(name, level, goal, doc_name=None):
    key = _CUR_USER.get()
    if not key:
        return gr.update(), gr.update(), "⚠️ Session expirée : rechargez la page pour vous reconnecter.", gr.update()
    name = str(name or "").strip()
    if not name or len(name) > 40:
        return gr.update(), gr.update(), "⚠️ Ton prénom est trop long (40 caractères maximum)." if name else "⚠️ Entre ton prénom et choisis ton niveau.", gr.update()
    _save_profile(key, name=name, level=level or "", goal=goal or "")
    _log("profile_save")
    return _profile_card(key), _user_badge_html(key), "✅ Profil enregistré.", dashboard_html(doc_name)


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
    for index, (key, u) in enumerate(sorted(users.items()), start=1):
        st = _user_stats(key)
        total_docs += st["docs"]
        total_size += st["size"]
        is_admin = u.get("role") == "admin"
        name = _display_name(key)
        ident = str(u.get("username", key))
        rows.append(
            "<tr>"
            f"<td class='ad-no'>{index}</td>"
            f"<td><div class='ad-profile'>{_avatar_html(key, 'ad-avatar')}"
            f"<span><b>{_escape_html(name or ident)}</b><small>{st['docs']} cours · {st['chats']} échanges · {_format_file_size(st['size'])}</small></span></div></td>"
            f"<td>{_escape_html(ident)}</td>"
            f"<td><span class='ad-pill {'ad-admin' if is_admin else 'ad-user'}'>{'Administrateur' if is_admin else 'Utilisateur'}</span></td>"
            "<td><span class='ad-pw' title='Le mot de passe est chiffré : il ne peut être lu par personne, pas même par l’administrateur.'>"
            "••••••••<span class='ad-pwnote'>🔒 Chiffré</span></span></td>"
            f"<td><span class='ad-pill {'ad-on' if u.get('active', True) else 'ad-off'}'>{'Actif' if u.get('active', True) else 'Désactivé'}</span></td>"
            f"<td>{_escape_html((u.get('last_login') or '—')[:16].replace('T', ' '))}<small class='ad-sub'>{int(u.get('logins', 0))} connexions</small></td>"
            "<td class='ad-actions'>"
            f"<button type='button' class='ak-act ak-a-aedit ak-u-{index - 1}'>📝 <span>Modifier</span></button>"
            f"<button type='button' class='ak-act ak-a-adel ak-u-{index - 1}'>🗑 <span>Supprimer</span></button></td>"
            "</tr>"
        )
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
        f"<div><span>Stockage</span><b>{('⚠️ Temporaire : comptes et cours effacés à chaque redémarrage' if STORAGE_KIND == 'Temporaire' else '✅ ' + _escape_html(STORAGE_KIND))}</b></div>"
        + (f"<div><span>Dernière sauvegarde</span><b>{_escape_html(_HF_STATE['last'] or 'en attente')}{(' · ⚠️ ' + _escape_html(_HF_STATE['error'])) if _HF_STATE['error'] else ''}</b></div>" if HF_SYNC else "") +
        f"<div><span>Dossier de données</span><b>{_escape_html(DATA_DIR)}</b></div>"
        f"<div><span>Taille PDF maximale</span><b>{int(MAX_PDF_MB)} Mo</b></div>"
        f"<div><span>Modèle d'embeddings</span><b>{_escape_html(EMBED_MODEL_NAME)} ({EMBED_DEVICE})</b></div>"
        f"<div><span>Seuil de pertinence</span><b>{MIN_SIMILARITY:.2f}</b></div>"
        f"<div><span>OCR (PDF scannés)</span><b>{_escape_html(_ocr_label())}</b></div>"
        f"{perf_html}"
        "</div>"
    )
    logs = [[r["t"].replace("T", " "), r["event"], r["user"],
             r["detail"] + (f" · réf. {(r.get('m') or {}).get('rid')}" if (r.get("m") or {}).get("rid") else "")]
            for r in log[:200]]
    table_html = (
        "<div class='ad-table-wrap'><table class='ad-table'><thead><tr><th>N°</th><th>Profil</th><th>E-mail</th><th>Rôle</th>"
        "<th>Mot de passe</th><th>Statut</th><th>Dernière connexion</th><th></th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )
    return kpi_html, table_html, gr.update(choices=[u.get("username", k) for k, u in sorted(users.items())]), logs, service


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


def _admin_user_keys():
    return sorted(_users().keys())


def admin_user_card(target):
    """Fiche de l'utilisateur sélectionné (le mot de passe est chiffré : il n'est jamais lisible)."""
    key = _target_key(target)
    if not key:
        return "<div class='ad-card ad-card-empty'>Sélectionnez un utilisateur dans la liste ou cliquez sur « Modifier ».</div>"
    u = _users()[key]
    st = _user_stats(key)
    name = _display_name(key)
    is_admin = u.get("role") == "admin"
    return (
        "<div class='ad-card'>"
        f"<div class='ad-card-head'>{_avatar_html(key, 'ad-avatar ad-avatar-lg')}"
        f"<div><b>{_escape_html(name or key)}</b><small>{_escape_html(str(u.get('username', key)))}</small></div>"
        f"<span class='ad-pill {'ad-admin' if is_admin else 'ad-user'}'>{'Administrateur' if is_admin else 'Utilisateur'}</span>"
        f"<span class='ad-pill {'ad-on' if u.get('active', True) else 'ad-off'}'>{'Actif' if u.get('active', True) else 'Désactivé'}</span></div>"
        "<div class='ad-card-grid'>"
        f"<div><span>Créé le</span><b>{_escape_html((u.get('created_at') or '—')[:16].replace('T', ' '))}</b></div>"
        f"<div><span>Dernière connexion</span><b>{_escape_html((u.get('last_login') or '—')[:16].replace('T', ' '))}</b></div>"
        f"<div><span>Connexions</span><b>{int(u.get('logins', 0))}</b></div>"
        f"<div><span>Cours</span><b>{st['docs']}</b></div>"
        f"<div><span>Échanges</span><b>{st['chats']}</b></div>"
        f"<div><span>Stockage</span><b>{_format_file_size(st['size'])}</b></div>"
        "</div>"
        "<div class='ad-card-pw'><span class='ad-pw'>••••••••</span><span>🔒 Mot de passe chiffré : personne ne peut le lire. "
        "Définissez-en un nouveau ou générez un mot de passe temporaire.</span></div>"
        "</div>"
    )


def admin_pick(payload):
    """Clic sur « Modifier » / « Supprimer » dans la liste : sélectionne l'utilisateur et affiche sa fiche."""
    denied = _admin_denied()
    if denied:
        return gr.update(), gr.update(), denied, gr.update()
    try:
        action, index = str(payload).split(":")[:2]
        key = _admin_user_keys()[int(index)]
    except (ValueError, IndexError):
        return gr.update(), gr.update(), "", gr.update()
    username = _users()[key].get("username", key)
    hint = ("ℹ️ Pour supprimer « {} » : cochez la confirmation, puis cliquez sur « Supprimer le compte »." if action == "adel"
            else "ℹ️ Modification de « {} » : choisissez une action ci-dessous.").format(username)
    return gr.update(value=username), admin_user_card(username), hint, gr.update(value="")


def admin_temp_password(target):
    denied = _admin_denied()
    if denied:
        return _admin_pack(denied)
    key = _target_key(target)
    if not key:
        return _admin_pack("⚠️ Sélectionnez un utilisateur.")
    password = secrets.token_urlsafe(8)
    with _STORE_LOCK:
        users = _users()
        users[key]["salt"], users[key]["hash"] = _hash_password(password)
        _write_json(USERS_FILE, users)
    _revoke_user_tokens(key)
    _log("admin_temp_password", _CUR_USER.get(), key)
    return _admin_pack(f"✅ Mot de passe temporaire de « {key} » : {password} — à lui communiquer (affiché une seule fois).")


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
        _io_flush(ACTIVITY_LOG)
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
      with gr.Column(visible=True, elem_id="landing") as landing_col:
        gr.HTML(landing_html())
        land_login_btn = gr.Button("login", elem_id="land-login-btn", elem_classes="akori-hidden")
      with gr.Column(visible=False, elem_id="login-wrap") as login_wrap:
        # Connexion façon Claude : logo, grande accroche, carte sobre ; comptes récents façon Messenger
        with gr.Row(elem_id="auth-top"):
            gr.HTML(f"<div class='auth-logo'><img src='{LOGO_DATA_URI}' alt='AKORI'><span>AKORI</span></div>")
            back_land = gr.Button("Découvrir AKORI", elem_id="back-land", elem_classes="link-btn", scale=0, min_width=150)
        gr.HTML("<div class='auth-hero'><h1>Vos cours,<br>votre réussite</h1>"
                "<p>Un assistant de révision qui ne répond qu'avec vos propres documents.</p></div>")
        gr.HTML("<div class='auth-tabs' role='tablist'><button type='button' class='auth-tab on' data-tab='login'>Se connecter</button>"
                "<button type='button' class='auth-tab' data-tab='register'>Créer un compte</button></div>", elem_id="auth-tabs-bar")
        with gr.Column(visible=True, elem_id="login-panel", elem_classes="auth-panel auth-card") as login_panel:
            gr.HTML("<div id='ak-accounts'></div>")
            li_user = gr.Textbox(label="E-mail", placeholder="Votre e-mail", elem_id="li-user")
            li_pass = gr.Textbox(label="Mot de passe", type="password", placeholder="Votre mot de passe", elem_id="li-pass")
            li_remember = gr.State(True)
            li_btn = gr.Button("Continuer", elem_id="li-btn")
            login_status = gr.Markdown("", elem_id="login-status", elem_classes="auth-status")
            forgot_btn = gr.Button("Mot de passe oublié ?", elem_classes="link-btn")
            gr.HTML("<div class='auth-or'><span>OU</span></div>")
            go_register = gr.Button("Créer un compte", elem_classes="auth-alt", elem_id="go-register")
        with gr.Column(visible=False, elem_id="register-panel", elem_classes="auth-panel auth-card") as register_panel:
            gr.HTML("<div class='auth-card-title'>Créer un compte</div>")
            su_user = gr.Textbox(label="E-mail", placeholder="prenom.nom@gmail.com", elem_id="su-user", info="Une adresse réelle que vous consultez.")
            su_pass = gr.Textbox(label="Mot de passe", type="password", placeholder="8 caractères minimum", elem_id="su-pass")
            su_pass2 = gr.Textbox(label="Confirmer le mot de passe", type="password", placeholder="Retapez le mot de passe", elem_id="su-pass2")
            su_remember = gr.State(True)
            su_btn = gr.Button("Créer mon compte", elem_id="su-btn")
            register_status = gr.Markdown("", elem_id="register-status", elem_classes="auth-status")
            gr.HTML("<div class='auth-or'><span>OU</span></div>")
            go_login = gr.Button("J'ai déjà un compte", elem_classes="auth-alt", elem_id="go-login")
        gr.HTML("<div class='auth-legal'>Vos documents restent dans votre espace personnel. AKORI ne répond qu'à partir de vos cours.</div>")

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
                nav_oral = gr.Button("Colle orale", elem_id="nav-oral", elem_classes="navbtn")
                nav_summary = gr.Button("Résumé", elem_id="nav-summary", elem_classes="navbtn")
                nav_mindmap = gr.Button("Mindmap", elem_id="nav-mindmap", elem_classes="navbtn")
                nav_assistant = gr.Button("Assistant IA", elem_id="nav-assistant", elem_classes="navbtn")
                nav_progress = gr.Button("Progression", elem_id="nav-progress", elem_classes="navbtn")
                nav_reviewq = gr.Button("À revoir", elem_id="nav-reviewq", elem_classes="navbtn")
                nav_plan = gr.Button("Plan de révision", elem_id="nav-plan", elem_classes="navbtn")
                nav_history = gr.Button("Historique", elem_id="nav-history", elem_classes="navbtn")
                nav_admin = gr.Button("Administration", elem_id="nav-admin", elem_classes="navbtn", visible=False)
                nav_settings = gr.Button("Paramètres", elem_id="nav-settings", elem_classes="navbtn")
                nav_logout = gr.Button("Déconnexion", elem_id="nav-logout", elem_classes="navbtn")
                user_badge = gr.HTML("", elem_id="user-badge")

            with gr.Column(scale=4, min_width=280, elem_id="main-column"):
                # En-tête : page active + cours actif (séparé du corps de la page)
                with gr.Row(elem_id="topbar"):
                    gr.HTML("<div class='pt-wrap'><span class='pt-ico'></span><div><div id='ak-page-title'>Accueil</div>"
                            "<div class='pt-sub'>AKORI · Espace de révision</div></div></div>", elem_id="topbar-title")
                    doc_selector = gr.Dropdown(label="Cours actif", choices=list(documents_db.keys()), value=(next(iter(documents_db), None)),
                                               scale=0, min_width=260, elem_id="doc-select")

                with gr.Tabs(elem_id="main-tabs") as tabs:
                    with gr.Tab("Accueil", id="home") as tab_home:
                        home_html = gr.HTML(dashboard_html(doc_selector.value))
                        start_review = gr.Button("Commencer la révision →", variant="primary", elem_classes="akori-hidden")

                    with gr.Tab("Mes dossiers", id="courses") as tab_courses:
                        with gr.Row(elem_classes="ak-cols", equal_height=False):
                            with gr.Column(scale=3, min_width=300, elem_classes="ak-pane"):
                                gr.HTML("<div class='pane-title'>Mes documents</div>")
                                with gr.Row(elem_id="upload-opts"):
                                    upload_title = gr.Textbox(label="Titre du cours (facultatif)", placeholder="Ex. : Réseaux — Chapitre 2", scale=3, min_width=200, elem_id="upload-title")
                                    upload_mod = gr.Dropdown(label="Ranger dans le module", choices=_module_choices(), value="", scale=2, min_width=160, elem_id="upload-mod")
                                with gr.Row(elem_id="documents-row", equal_height=True):
                                    docs_html = gr.HTML(documents_html_v16(doc_selector.value), scale=3)
                                    pdf_input = gr.UploadButton(
                                        "＋\nAjouter des PDF",
                                        file_types=None,  # contrôle fait côté serveur (signature %PDF) : Android donne parfois des noms sans « .pdf »
                                        file_count="multiple",
                                        type="filepath",
                                        elem_id="add-document-tile",
                                        scale=1,
                                    )
                                upload_status = gr.Markdown("", elem_id="upload-status")
                                course_info = gr.HTML(course_overview_html(doc_selector.value), elem_classes="akori-hidden")
                                del_doc_box = gr.Textbox(elem_id="del-doc-box", elem_classes="akori-hidden", show_label=False, container=False)
                                del_doc_btn = gr.Button("del", elem_id="del-doc-btn", elem_classes="akori-hidden")
                            with gr.Column(scale=2, min_width=280, elem_classes="ak-pane"):
                                gr.HTML("<div class='pane-title'>Modules · semestres</div>")
                                modules_box = gr.HTML(modules_html(doc_selector.value))
                                with gr.Row(elem_id="module-row"):
                                    mod_name = gr.Textbox(label="Nouveau module", placeholder="Ex. : Réseaux", scale=3, elem_id="mod-name")
                                    mod_sem = gr.Dropdown(label="Semestre", choices=SEMESTERS, value=SEMESTERS[0], scale=1, min_width=115, elem_id="mod-sem")
                                mod_create = gr.Button("Créer le module", variant="primary", elem_id="mod-create")
                                with gr.Row(elem_id="module-row2"):
                                    mod_pick = gr.Dropdown(label="Module du cours actif", choices=_module_choices(), value="", scale=3, elem_id="mod-pick")
                                    mod_del = gr.Button("Supprimer", scale=1, min_width=120, elem_id="mod-del")
                                mod_status = gr.Markdown("", elem_id="mod-status")

                    with gr.Tab("Réviser", id="review") as tab_review:
                        review_cards = gr.HTML(course_overview_html(doc_selector.value))
                        with gr.Row(elem_classes="ak-toolbar"):
                            review_summary = gr.Button("Résumé", variant="secondary")
                            review_flash = gr.Button("Flashcards", variant="primary")
                            review_quiz = gr.Button("Quiz / QCM", variant="primary")
                            review_chat = gr.Button("Assistant IA", variant="secondary")

                    with gr.Tab("Flashcards", id="flashcards"):
                        with gr.Row(elem_classes="ak-toolbar"):
                            generate_flash = gr.Button("✦ Générer les flashcards", variant="primary", scale=0, min_width=230)
                            flash_status = gr.Markdown("", elem_id="flash-status")
                        flash_view = gr.HTML(flashcard_view([], 0, False))
                        with gr.Row(visible=False, elem_classes="ak-toolbar") as flash_actions:
                            flash_reveal = gr.Button("Afficher la réponse", variant="primary")
                            flash_known = gr.Button("✓ Je savais", variant="primary")
                            flash_review = gr.Button("↻ À revoir")

                    with gr.Tab("Quiz / QCM", id="quiz"):
                        with gr.Row(elem_classes="ak-toolbar"):
                            generate_quiz_btn = gr.Button("✦ Générer le quiz", variant="primary", scale=0, min_width=200)
                            quiz_restart = gr.Button("🔁 Rejouer", scale=0, min_width=130)
                            quiz_status = gr.Markdown("")
                        quiz_view_box = gr.HTML(quiz_view([], 0))
                        quiz_choice = gr.Radio(
                            choices=[], label="Choisissez votre réponse", container=False,
                            interactive=True, visible=False, elem_classes="quiz-radio-group",
                        )
                        quiz_pick = gr.Textbox(elem_id="quiz-pick-box", elem_classes="akori-hidden", show_label=False, container=False)
                        quiz_pick_btn = gr.Button("pick", elem_id="quiz-pick-btn", elem_classes="akori-hidden")
                        quiz_validate = gr.Button("Valider la réponse", elem_id="quiz_validate", elem_classes="akori-hidden")
                        quiz_next = gr.Button(NEXT_LABEL, elem_id="quiz_next", elem_classes="akori-hidden")

                    with gr.Tab("Colle orale", id="oral"):
                        oral_state = gr.State({})
                        with gr.Row(elem_classes="ak-cols", equal_height=False):
                            with gr.Column(scale=1, min_width=300, elem_classes="ak-pane"):
                                oral_card = gr.HTML(oral_card_html(None))
                                oral_answer = gr.Textbox(lines=5, max_lines=16, label="Votre réponse", placeholder="Écrivez ou dictez votre réponse…", elem_id="oral-answer")
                                with gr.Row(elem_classes="ak-toolbar"):
                                    oral_new = gr.Button("Nouvelle question", elem_id="oral-new")
                                    oral_mic = gr.Button("🎤 Dicter", elem_id="oral-mic")
                                    oral_submit = gr.Button("Corriger", variant="primary", elem_id="oral-submit")
                                oral_status = gr.Markdown("", elem_id="oral-status")
                            with gr.Column(scale=1, min_width=300, elem_classes="ak-pane"):
                                oral_result = gr.HTML(ORAL_EMPTY)
                                oral_history = gr.HTML("")

                    with gr.Tab("Résumé", id="summary"):
                        with gr.Row(elem_classes="ak-toolbar"):
                            summary_btn = gr.Button("✦ Générer le résumé", variant="primary", scale=0, min_width=220, elem_id="sum-gen")
                            gr.HTML("<div class='tb-hint'>Les points clés du cours actif, conservés avec le document.</div>")
                        summary_output = gr.Markdown("Sélectionnez un cours puis lancez la génération.", elem_id="summary-out")

                    with gr.Tab("Mindmap", id="mindmap") as tab_mindmap:
                        with gr.Row(elem_classes="ak-toolbar"):
                            mm_btn = gr.Button("✦ Générer la mindmap", variant="primary", elem_id="mm-gen", scale=0, min_width=220)
                            mm_status = gr.Markdown("", elem_id="mm-status")
                        mm_box = gr.HTML(mindmap_html(None))

                    with gr.Tab("Assistant IA", id="assistant"):
                        with gr.Row(elem_classes="ak-toolbar", elem_id="scope-row"):
                            gr.HTML("<div class='tb-hint'>Chercher dans :</div>")
                            chat_scope = gr.Radio(choices=[("Ce cours", "doc"), ("Mon module", "module"), ("Tout mon espace", "all")], value="doc",
                                                  show_label=False, container=False, elem_id="chat-scope", scale=0)
                        chatbot = gr.Chatbot(
                            value=_chat_history_for_ui(doc_selector.value) if doc_selector.value else [],
                            height=470,
                            elem_id="assistant-chatbot",
                            avatar_images=(None, _logo_favicon_path()),
                            placeholder=("<div class='chat-empty'><img src='" + LOGO_DATA_URI + "' alt='AKORI'>"
                                         "<b>Une question sur ce cours ?</b>"
                                         "<span>AKORI répond à partir de ses pages et cite le passage utilisé.</span></div>"),
                            show_label=False,
                            render_markdown=True,
                            line_breaks=True,
                            latex_delimiters=[
                                {"left": "$$", "right": "$$", "display": True},
                                {"left": "$", "right": "$", "display": False},
                                {"left": "\\[", "right": "\\]", "display": True},
                                {"left": "\\(", "right": "\\)", "display": False},
                            ],
                        )
                        with gr.Row(elem_id="assistant-bar"):
                            chat_mic = gr.Button("🎤", elem_id="chat-mic", scale=0, min_width=44)
                            msg_input = gr.Textbox(placeholder="Posez votre question…", show_label=False, container=False, scale=5, elem_id="assistant-input")
                            send_btn = gr.Button("➤", variant="primary", scale=0, min_width=44, elem_id="assistant-send")

                    with gr.Tab("Progression", id="progress"):
                        with gr.Row(elem_classes="ak-cols", equal_height=False):
                            with gr.Column(scale=1, min_width=300, elem_classes="ak-pane"):
                                gr.HTML("<div class='pane-title'>Vue d'ensemble</div>")
                                global_progress_output = gr.HTML(global_progress_html(doc_selector.value))
                            with gr.Column(scale=1, min_width=300, elem_classes="ak-pane"):
                                gr.HTML("<div class='pane-title'>Cours actif</div>")
                                progress_output = gr.HTML(progress_detail_html(doc_selector.value))

                    with gr.Tab("À revoir", id="reviewq") as tab_reviewq:
                        pins_box = gr.HTML(pins_view(doc_selector.value))

                    with gr.Tab("Paramètres", id="settings", elem_id="tab-settings"):
                        with gr.Row(elem_classes="ak-cols", equal_height=False):
                            with gr.Column(scale=1, min_width=300, elem_classes="ak-pane"):
                                with gr.Row(elem_classes="set-card"):
                                    gr.HTML("<div class='set-text'><div class='set-title'>Mode sombre</div><div class='set-desc'>Thème clair ou sombre.</div></div>")
                                    theme_toggle = gr.Button(" ", elem_id="theme-toggle-btn", scale=0, min_width=84)
                                with gr.Row(elem_classes="set-card"):
                                    gr.HTML("<div class='set-text'><div class='set-title'>Langue</div><div class='set-desc'>Interface et réponses de l'IA.</div></div>")
                                    lang_choice = gr.Radio(choices=LANG_CHOICES, value="Français", show_label=False, container=False, elem_id="lang-choice", scale=0)
                            with gr.Column(scale=1, min_width=300, elem_classes="ak-pane"):
                                gr.HTML(about_html())

                    with gr.Tab("Profil", id="profile", elem_id="tab-profile"):
                        with gr.Row(elem_classes="ak-cols", equal_height=False):
                            with gr.Column(scale=1, min_width=280, elem_classes="ak-pane"):
                                prof_card = gr.HTML("")
                                with gr.Row(elem_classes="ak-toolbar"):
                                    prof_photo = gr.UploadButton("Ajouter une photo", file_types=["image"], file_count="single", type="filepath", elem_id="prof-photo")
                                    prof_remove = gr.Button("Retirer la photo", elem_id="prof-remove")
                            with gr.Column(scale=1, min_width=280, elem_classes="ak-pane"):
                                prof_name = gr.Textbox(label="Prénom", placeholder="Ton prénom", elem_id="prof-name")
                                prof_level = gr.Radio(choices=PROF_LEVELS, label="Niveau", elem_id="prof-level")
                                prof_goal = gr.Radio(choices=PROF_GOALS, label="Objectif", elem_id="prof-goal")
                                prof_save = gr.Button("Enregistrer", variant="primary", elem_id="prof-save")
                                prof_status = gr.Markdown("", elem_id="prof-status")

                    with gr.Tab("Plan de révision", id="plan") as tab_plan:
                        pw_step = gr.State(1)
                        # Assistant en 5 étapes : contexte, calendrier, matières (PDF importés), rythme, fonctions IA
                        with gr.Column(visible=True, elem_id="plan-wizard") as pw_col:
                            pw_head = gr.HTML(_pw_head(1), elem_id="pw-head")
                            with gr.Row(elem_id="pw-stage", equal_height=False):
                                pw_mascot = gr.HTML(_pw_mascot("wave"), elem_id="pw-mascot", scale=0, min_width=120)
                                with gr.Column(elem_id="pw-fields", scale=1, min_width=240):
                                    with gr.Column(visible=True, elem_classes="pw-step") as pw_s1:
                                        pw_context = gr.Radio(PW_CONTEXT_CHOICES, value="exam", show_label=False, container=False, elem_id="pw-context")
                                        plan_name = gr.Textbox(label="Intitulé (facultatif)", placeholder="Ex. : Partiel de réseaux", elem_id="plan-name")
                                    with gr.Column(visible=False, elem_classes="pw-step") as pw_s2:
                                        plan_date = gr.Textbox(label="Date de l'examen", placeholder="AAAA-MM-JJ", elem_id="plan-date",
                                                               info="Facultative en apprentissage libre (plan sur 4 semaines).")
                                        with gr.Row(elem_classes="pw-row"):
                                            plan_minutes = gr.Slider(10, 240, value=40, step=5, label="Minutes par jour", elem_id="plan-minutes", scale=3)
                                            pw_start = gr.Textbox(value="18:00", label="Heure de début", elem_id="pw-start", scale=1, min_width=180, max_lines=1)
                                    with gr.Column(visible=False, elem_classes="pw-step") as pw_s3:
                                        pw_docs = gr.Dropdown(choices=[], value=[], multiselect=True, label="Matières (vos PDF importés)",
                                                              info="Tapez pour rechercher une matière.", elem_id="pw-docs")
                                        pw_acts = gr.CheckboxGroup(PW_ACT_CHOICES, value=list(PLAN_ACTS), label="Activités", elem_id="pw-acts")
                                    with gr.Column(visible=False, elem_classes="pw-step") as pw_s4:
                                        pw_intensity = gr.Radio(PW_INTENSITY_CHOICES, value="mid", label="Intensité", elem_id="pw-intensity")
                                        pw_rest = gr.CheckboxGroup(PW_REST_CHOICES, value=["6"], label="Jours de repos", elem_id="pw-rest")
                                    with gr.Column(visible=False, elem_classes="pw-step") as pw_s5:
                                        pw_ai = gr.CheckboxGroup(PW_AI_CHOICES, value=["spaced", "prio"], label="Fonctions IA", elem_id="pw-ai")
                                        gr.HTML("<div class='pw-ai-help'><p><b>Répétition espacée</b> : chaque activité revient à intervalles croissants.</p>"
                                                "<p><b>Priorisation IA</b> : les matières et les pages les plus fragiles passent en premier.</p>"
                                                "<p><b>Alertes de surcharge</b> : AKORI signale les journées qui dépassent ton temps.</p></div>")
                                    pw_err = gr.Markdown("", elem_id="plan-status")
                            with gr.Row(elem_id="pw-nav"):
                                pw_prev_btn = gr.Button("‹ Précédent", elem_id="pw-prev", visible=False, scale=0, min_width=130)
                                pw_cancel_btn = gr.Button("Annuler", elem_id="pw-cancel", scale=0, min_width=120)
                                pw_next_btn = gr.Button("Suivant", elem_id="pw-next", scale=0, min_width=120)
                                pw_gen_btn = gr.Button("Générer le plan final", elem_id="pw-gen", visible=False, scale=0, min_width=220)
                        # Emploi du temps intelligent (affiché une fois le plan généré)
                        with gr.Column(visible=False, elem_id="plan-edt") as edt_col:
                            plan_box = gr.HTML("")
                        plan_go_box = gr.Textbox(elem_id="plan-go-box", elem_classes="akori-hidden", show_label=False, container=False)
                        plan_go_btn = gr.Button("go", elem_id="plan-go-btn", elem_classes="akori-hidden")
                        plan_edit_btn = gr.Button("edit", elem_id="plan-edit-btn", elem_classes="akori-hidden")
                        plan_del = gr.Button("del", elem_id="plan-clear", elem_classes="akori-hidden")

                    with gr.Tab("Historique", id="history") as tab_history:
                        journal_box = gr.HTML(history_page_html())
                        history_box = gr.HTML(history_view(doc_selector.value), elem_classes="akori-hidden")


    # ------------------------------------------------------------ vue : administration (une seule page)
    with gr.Column(visible=False, elem_id="admin-view") as admin_view:
        with gr.Row(elem_id="admin-top"):
            gr.HTML("<div class='admin-title'><div class='admin-badge'>🛡</div><div><h1>Console d'administration</h1><p>Suivi, maintenance et gestion des comptes AKORI</p></div></div>")
            admin_refresh_btn = gr.Button("↻ Actualiser", scale=0, min_width=130)
            admin_back = gr.Button("← Retour à l'application", variant="primary", scale=0, min_width=220)
        admin_kpis = gr.HTML()
        with gr.Tabs(elem_id="admin-tabs"):
            with gr.Tab("Utilisateurs"):
                with gr.Row(elem_id="admin-list-head"):
                    gr.HTML("<div class='ad-list-title'>Liste des utilisateurs</div>")
                    admin_search = gr.Textbox(placeholder="Rechercher un utilisateur", show_label=False, container=False, elem_id="admin-search", scale=0, min_width=280)
                admin_table = gr.HTML(elem_id="admin-users")
                admin_pick_payload = gr.Textbox(elem_id="admin-pick", elem_classes="akori-hidden", show_label=False, container=False)
                admin_pick_btn = gr.Button("pick", elem_id="admin-pick-btn", elem_classes="akori-hidden")
                admin_card = gr.HTML(admin_user_card(None), elem_id="admin-card")
                with gr.Row():
                    admin_user_sel = gr.Dropdown(label="Utilisateur", choices=[], scale=2, elem_id="admin-user-sel")
                    admin_newpwd = gr.Textbox(label="Nouveau mot de passe", type="password", scale=2, elem_id="admin-newpwd")
                with gr.Row():
                    admin_btn_toggle = gr.Button("Activer / Désactiver")
                    admin_btn_reset = gr.Button("Réinitialiser le mot de passe")
                    admin_btn_role = gr.Button("Promouvoir / Rétrograder")
                    admin_btn_temp = gr.Button("Générer un mot de passe temporaire")
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
    nav_profile = gr.Button("profil", elem_id="nav-profile", elem_classes="akori-hidden")
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

    def _prepare_pdf(path, title="", n=1, total=1):
        """Vérifie que le fichier est bien un PDF (signature %PDF) et lui donne un nom propre.

        Renvoie (chemin, None) ou (None, message d'erreur)."""
        path = getattr(path, "name", path) or ""
        base = os.path.basename(path)
        try:
            if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
                return None, f"⚠️ « {base or 'fichier'} » est vide ou illisible."
            with open(path, "rb") as fh:
                head = fh.read(1024)
        except OSError:
            return None, f"⚠️ Impossible de lire « {base} »."
        if b"%PDF" not in head:
            return None, f"⚠️ « {base} » n'est pas un PDF valide."
        clean = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', " ", (title or "").strip())
        clean = re.sub(r"\s+", " ", clean).strip(" .")[:80]
        if clean:
            name = clean + (f" ({n})" if total > 1 else "") + ".pdf"
        else:
            stem = base[:-4] if base.lower().endswith(".pdf") else base
            stem = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', " ", stem).strip(" .") or "document"
            name = stem + ".pdf"
        if name != base:
            dst_dir = tempfile.mkdtemp(prefix="akori_up_")
            dst = os.path.join(dst_dir, name)
            shutil.copyfile(path, dst)
            path = dst
        return path, None

    def upload_many(files, lang=None, module_id="", title=""):
        """Un ou plusieurs PDF d'un coup, rangés dans le module choisi, avec un titre facultatif."""
        try:
            yield from _upload_many(files, lang, module_id, title)
        except Exception as exc:  # jamais de message « ⏳ » bloqué : on affiche l'erreur
            print("upload error:", repr(exc))
            yield (f"⚠️ Import interrompu : {exc}",) + tuple([gr.update()] * (10 + REVISION_N))

    def _upload_many(files, lang=None, module_id="", title=""):
        if not isinstance(files, (list, tuple)):
            files = [files] if files else []
        title = (title or "").strip()
        if not _CUR_USER.get():   # serveur redémarré alors que la page était ouverte : on recharge pour se reconnecter
            yield ("⚠️ Session expirée : rechargez la page pour vous reconnecter.",) + tuple([gr.update()] * (10 + REVISION_N))
            return
        total, done, errors = len(files), [], []
        if not total:
            yield ("⚠️ Aucun fichier reçu. Réessayez avec un fichier PDF.",) + tuple([gr.update()] * (10 + REVISION_N))
            return
        for n, raw in enumerate(files, 1):
            f, err = _prepare_pdf(raw, title, n, total)
            if err:
                errors.append(err)
                yield (err,) + tuple([gr.update()] * (10 + REVISION_N))
                continue
            last = None
            for out in upload_and_refresh(f, lang):
                last = out
                msg = out[0] if isinstance(out[0], str) else ""
                if total > 1 and isinstance(out[0], str):
                    out = (f"({n}/{total}) {msg}",) + tuple(out[1:])
                yield out + (gr.update(),)
            sel = os.path.basename(f)
            if sel and sel in documents_db:
                done.append(sel)
                changed = False
                if title:
                    documents_db[sel]["title"] = title + (f" ({n})" if total > 1 else "")
                    changed = True
                if module_id and module_id in _space()["modules"]:
                    documents_db[sel]["module"] = module_id
                    changed = True
                if changed:
                    _save_history(sel)
                    yield (gr.update(), gr.update(), documents_html_v16(sel), dashboard_html(sel), course_overview_html(sel),
                           course_overview_html(sel), global_progress_html(sel), gr.update(), gr.update(), gr.update(),
                           *([gr.update()] * REVISION_N), modules_html(sel))
        if total > 1 or (errors and done):
            msg = f"✅ {len(done)} document(s) importé(s) sur {total}." + (("  \n" + "  \n".join(errors)) if errors else "")
            yield (msg,) + tuple([gr.update()] * (9 + REVISION_N)) + ((modules_html(done[-1]) if done else gr.update()),)

    # Retour immédiat dès la sélection du PDF, puis import par étapes (extraction, indexation par lots).
    upload_event = pdf_input.upload(
        lambda: "⏳ Analyse du PDF… extraction, découpage et indexation en cours.",
        inputs=None, outputs=[upload_status], queue=False
    )
    upload_event.then(
        upload_many, inputs=[pdf_input, lang_choice, upload_mod, upload_title],
        outputs=[upload_status, doc_selector, docs_html, home_html, course_info,
                 review_cards, global_progress_output, progress_output, chatbot, history_box] + REVISION_OUT + [modules_box],
        queue=True, show_progress="hidden", concurrency_limit=3
    ).then(lambda: "", None, upload_title, queue=False)

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

    del_doc_btn.click(delete_document, [del_doc_box, doc_selector], [upload_status, doc_selector, docs_html, modules_box, del_doc_box],
                      queue=False).then(refresh_all, inputs=[doc_selector, lang_choice],
                                        outputs=[home_html, docs_html, course_info, review_cards, global_progress_output, progress_output,
                                                 chatbot, history_box] + REVISION_OUT, queue=False, show_progress="hidden")

    # Navigation vers les onglets
    nav_home.click(lambda: gr.Tabs(selected="home"), outputs=tabs, queue=False)
    nav_courses.click(lambda: gr.Tabs(selected="courses"), outputs=tabs, queue=False)
    nav_review.click(lambda: gr.Tabs(selected="review"), outputs=tabs, queue=False)
    nav_flash.click(lambda: gr.Tabs(selected="flashcards"), outputs=tabs, queue=False)
    nav_quiz.click(lambda: gr.Tabs(selected="quiz"), outputs=tabs, queue=False)
    nav_summary.click(lambda: gr.Tabs(selected="summary"), outputs=tabs, queue=False)
    nav_oral.click(lambda: gr.Tabs(selected="oral"), outputs=tabs, queue=False)
    nav_profile.click(lambda: gr.Tabs(selected="profile"), outputs=tabs, queue=False).then(
        profile_view, None, [prof_card, prof_name, prof_level, prof_goal, prof_status], queue=False)
    PROF_OUT = [prof_card, user_badge, prof_status, home_html]
    prof_photo.upload(profile_photo_upload, [prof_photo, doc_selector], PROF_OUT, queue=False)
    prof_remove.click(profile_photo_remove, [doc_selector], PROF_OUT, queue=False)
    prof_save.click(profile_save, [prof_name, prof_level, prof_goal, doc_selector], PROF_OUT, queue=False)
    oral_new.click(oral_new_handler, [doc_selector, lang_choice, oral_state], [oral_state, oral_card, oral_answer, oral_result, oral_status], show_progress="hidden")
    oral_submit.click(oral_grade_handler, [oral_state, oral_answer, doc_selector, lang_choice], [oral_result, oral_status, oral_history], show_progress="hidden").then(
        after_session_refresh, None, [plan_box, journal_box], queue=False, show_progress="hidden")
    oral_mic.click(None, None, None, js=ORAL_MIC_JS, queue=False)
    chat_mic.click(None, None, None, js=CHAT_MIC_JS, queue=False)
    doc_selector.change(oral_reset, [doc_selector], [oral_state, oral_card, oral_answer, oral_result, oral_status, oral_history], queue=False)
    nav_assistant.click(lambda: gr.Tabs(selected="assistant"), outputs=tabs, queue=False)
    nav_progress.click(lambda: gr.Tabs(selected="progress"), outputs=tabs, queue=False)
    nav_reviewq.click(lambda: gr.Tabs(selected="reviewq"), outputs=tabs, queue=False)
    nav_history.click(lambda: gr.Tabs(selected="history"), outputs=tabs, queue=False)
    nav_plan.click(lambda: gr.Tabs(selected="plan"), outputs=tabs, queue=False)
    nav_mindmap.click(lambda: gr.Tabs(selected="mindmap"), outputs=tabs, queue=False)
    mm_btn.click(lambda: "⏳ Construction de la mindmap…", outputs=[mm_status], queue=False).then(
        mindmap_generate_handler, [doc_selector, lang_choice], [mm_box, mm_status], show_progress="hidden", concurrency_limit=2).then(
        after_session_refresh, None, [plan_box, journal_box], queue=False, show_progress="hidden")
    tab_mindmap.select(mindmap_view, [doc_selector, lang_choice], [mm_box, mm_status], queue=False, show_progress="hidden")
    doc_selector.change(mindmap_view, [doc_selector, lang_choice], [mm_box, mm_status], queue=False, show_progress="hidden")
    PW_VIEW = [pw_step, pw_head, pw_mascot, pw_s1, pw_s2, pw_s3, pw_s4, pw_s5, pw_prev_btn, pw_next_btn, pw_gen_btn, pw_err]
    PW_FIELDS = [pw_context, plan_name, plan_date, plan_minutes, pw_start, pw_docs, pw_acts, pw_intensity, pw_rest, pw_ai]
    PLAN_OPEN_OUT = [pw_col, edt_col, plan_box] + PW_VIEW + PW_FIELDS
    tab_plan.select(plan_open, None, PLAN_OPEN_OUT, queue=False, show_progress="hidden")
    plan_edit_btn.click(plan_edit, None, PLAN_OPEN_OUT, queue=False, show_progress="hidden")
    plan_del.click(plan_clear, None, PLAN_OPEN_OUT, queue=False, show_progress="hidden")
    tab_history.select(history_page_html, None, [journal_box], queue=False, show_progress="hidden")
    nav_plan.click(plan_open, None, PLAN_OPEN_OUT, queue=False, show_progress="hidden")
    pw_next_btn.click(pw_next, [pw_step, pw_context, plan_date, pw_docs, pw_acts, pw_rest], PW_VIEW, queue=False, show_progress="hidden")
    pw_prev_btn.click(pw_prev, [pw_step], PW_VIEW, queue=False, show_progress="hidden")
    pw_cancel_btn.click(pw_cancel, None, [pw_col, edt_col] + PW_VIEW, queue=False, show_progress="hidden")
    pw_gen_btn.click(pw_generate, PW_FIELDS, [pw_col, edt_col, plan_box] + PW_VIEW, queue=False, show_progress="hidden")
    plan_go_btn.click(lambda d: gr.update(value=d) if d in documents_db else gr.update(), [plan_go_box], [doc_selector],
                      queue=False, show_progress="hidden")
    mod_create.click(module_create, [mod_name, mod_sem, doc_selector], [mod_pick, mod_status, modules_box, mod_name], queue=False)
    mod_pick.input(module_assign, [doc_selector, mod_pick], [mod_status, modules_box], queue=False)
    mod_del.click(module_delete, [mod_pick, doc_selector], [mod_pick, mod_status, modules_box], queue=False)
    doc_selector.change(module_for_doc, [doc_selector], [mod_pick, modules_box], queue=False)
    _refresh_up = lambda: gr.update(choices=_module_choices())
    tab_courses.select(_refresh_up, None, [upload_mod], queue=False, show_progress="hidden")
    mod_create.click(_refresh_up, None, [upload_mod], queue=False)
    mod_del.click(_refresh_up, None, [upload_mod], queue=False)
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
        inputs=[pending_message, doc_selector, chat_scope],
        outputs=[chatbot, pending_message],
        show_progress="hidden",
    ).then(history_view, inputs=[doc_selector], outputs=[history_box], queue=False).then(
        after_session_refresh, None, [plan_box, journal_box], queue=False, show_progress="hidden")

    submit_event = msg_input.submit(
        _ui_chat_user_message,
        inputs=[msg_input, chatbot, doc_selector],
        outputs=[msg_input, chatbot, pending_message],
        queue=False,
    )
    submit_event.then(
        repondre_akori_chat_stream,
        inputs=[pending_message, doc_selector, chat_scope],
        outputs=[chatbot, pending_message],
        show_progress="hidden",
    ).then(history_view, inputs=[doc_selector], outputs=[history_box], queue=False).then(
        after_session_refresh, None, [plan_box, journal_box], queue=False, show_progress="hidden")

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
    ).then(lambda d: (global_progress_html(d), progress_detail_html(d)), inputs=[doc_selector], outputs=[global_progress_output, progress_output], queue=False).then(
        after_session_refresh, None, [plan_box, journal_box], queue=False, show_progress="hidden")
    flash_review.click(
        lambda c, i, d, lg: flashcard_mark_handler(c, i, "review", d, lg),
        inputs=[flashcards_state, flash_index, doc_selector, lang_choice],
        outputs=[flash_index, flash_revealed, flash_view, flash_actions, generate_flash],
        queue=False,
    ).then(lambda d: (global_progress_html(d), progress_detail_html(d)), inputs=[doc_selector], outputs=[global_progress_output, progress_output], queue=False).then(
        after_session_refresh, None, [plan_box, journal_box], queue=False, show_progress="hidden")

    # Quiz
    quiz_outputs = [quiz_state, quiz_index, quiz_validated, quiz_view_box, quiz_status, quiz_choice, quiz_answers, quiz_next]
    generate_quiz_btn.click(
        lambda: "⏳ Génération du quiz…", outputs=[quiz_status], queue=False
    ).then(quiz_generate_handler, inputs=[doc_selector, lang_choice], outputs=quiz_outputs, show_progress="minimal")
    quiz_restart.click(quiz_restart_handler, inputs=[quiz_state, doc_selector], outputs=quiz_outputs, queue=False)
    quiz_pick_btn.click(
        quiz_validate_handler,
        inputs=[quiz_state, quiz_index, quiz_pick, quiz_validated, quiz_answers, doc_selector],
        outputs=[quiz_validated, quiz_view_box, quiz_status, quiz_answers, quiz_choice, quiz_next],
        queue=False,
    ).then(lambda: "", None, quiz_pick, queue=False, show_progress="hidden")
    quiz_next.click(
        quiz_next_handler,
        inputs=[quiz_state, quiz_index, quiz_validated, quiz_answers, doc_selector],
        outputs=[quiz_index, quiz_validated, quiz_view_box, quiz_status, quiz_choice, quiz_next],
        queue=False,
    ).then(lambda d: (global_progress_html(d), progress_detail_html(d)), inputs=[doc_selector], outputs=[global_progress_output, progress_output], queue=False).then(
        after_session_refresh, None, [plan_box, journal_box], queue=False, show_progress="hidden")

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
            _session("summary", d)
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
        lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), ""), None,
        [login_panel, register_panel, login_wrap, landing_col, li_pass], queue=False)
    _after_enter(token_in.change(do_auto_login, [token_in], ENTER_OUT))
    token_out.change(None, token_out, None, js=TOKEN_STORE_JS)
    demo.load(None, None, token_in, js=GET_TOKEN_JS)

    # page de présentation -> connexion
    land_login_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), "", ""), None,
                         [landing_col, login_wrap, login_panel, register_panel, login_status, register_status], queue=False)
    back_land.click(lambda: (gr.update(visible=True), gr.update(visible=False)), None, [landing_col, login_wrap], queue=False)

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
    admin_btn_temp.click(admin_temp_password, [admin_user_sel], ADMIN_OUT)
    admin_pick_btn.click(admin_pick, [admin_pick_payload], [admin_user_sel, admin_card, admin_msg, admin_newpwd], queue=False)
    admin_user_sel.change(admin_user_card, [admin_user_sel], [admin_card], queue=False)
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
                tries = 0
                while len(entry.setdefault("oral_bank", [])) < 5 and tries < 8:     # banque de questions de colle orale
                    tries += 1
                    try:
                        passage = _oral_prepare(entry)
                        if passage:
                            entry["oral_bank"].append(_oral_ask(passage, langs[0]))
                    except Exception as e:
                        print(f"   ⚠️ {name} : question de colle orale impossible : {e}")
                        break
                print(f"   🎤 {name} : {len(entry['oral_bank'])} questions de colle orale")
                demo_questions = [q.strip() for q in os.getenv("AKORI_DEMO_QUESTIONS", "").split("|") if q.strip()] or [
                    "De quoi parle ce cours ?", "Quelles sont les notions principales de ce cours ?",
                    "Quel est l'objectif de ce document ?"]
                for question in demo_questions:
                    before = len(entry["history"])
                    try:
                        list(_ui_chat_user_message(question, [], name))
                        for _ in repondre_akori_chat_stream(question, name):
                            pass
                        print(f"   💬 {name} : réponse mémorisée « {question[:50]} »")
                    except Exception as e:
                        print(f"   ⚠️ {name} : question de démonstration impossible : {e}")
                    del entry["history"][before:]          # l'historique de démonstration reste vierge ; la réponse est en cache
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
    _in_colab = "google.colab" in sys.modules or os.path.isdir("/content")
    _share = (os.getenv("AKORI_SHARE") == "1") or (_in_colab and os.getenv("AKORI_SHARE") != "0")
    _server = {} if _in_colab else {"server_name": "0.0.0.0", "server_port": int(os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT") or 7860)}
    demo.queue(default_concurrency_limit=4, max_size=64).launch(
        share=_share,
        **_server,
        debug=False,
        show_error=True,
        head='<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"><meta name="theme-color" content="#4f6df5"><meta name="mobile-web-app-capable" content="yes"><meta name="apple-mobile-web-app-capable" content="yes"><meta name="apple-mobile-web-app-title" content="AKORI"><meta name="google" content="notranslate"><meta http-equiv="Content-Language" content="fr"><script>document.documentElement.setAttribute("translate","no");document.documentElement.lang="fr";</script>',
        theme=theme_akori,
        css=custom_css,
        js=AKORI_NAV_JS,
        favicon_path=_logo_favicon_path(),
    )
