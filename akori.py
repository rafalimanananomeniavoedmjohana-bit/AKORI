import hashlib
import html
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
)


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


# --- 2. STOCKAGE PERSISTANT ---
DATA_DIR = os.getenv(
    "AKORI_DATA_DIR", os.path.join(os.getcwd(), "akori_data_store")
)
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

documents_db = {}


def _doc_id_from_filename(filename):
  return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]


def _doc_folder(filename):
  return os.path.join(DOCUMENTS_DIR, _doc_id_from_filename(filename))


def _doc_metadata_path(filename):
  return os.path.join(_doc_folder(filename), "metadata.json")


def _doc_index_path(filename):
  return os.path.join(_doc_folder(filename), "faiss_index")
def clean_latex_artifacts(text: str) -> str:
    """Nettoie ou convertit les artefacts LaTeX bruts pour un affichage lisible."""
    # Exemple : transformer les \text{...} en texte simple ou harmoniser les délimiteurs
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    # Vous pouvez aussi nettoyer les symboles superflus si besoin
    return text


def _save_document(filename):
  data = documents_db.get(filename)
  if not data:
    return

  folder = _doc_folder(filename)
  os.makedirs(folder, exist_ok=True)

  data["vector_db"].save_local(_doc_index_path(filename))

  metadata = {
      "filename": filename,
      "title": data.get("title", ""),
      "chunks_count": int(data.get("chunks_count", 0)),
      "file_size": int(data.get("file_size", 0)),
      "added_at": data.get("added_at", ""),
      "history": data.get("history", []),
      "progress": data.get("progress", {"flashcards": {}, "quiz_attempts": []}),
  }

  with open(_doc_metadata_path(filename), "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)


def _save_history(filename):
  """Sauvegarde uniquement l'historique JSON, sans réécrire l'index FAISS."""
  data = documents_db.get(filename)
  if not data:
    return

  folder = _doc_folder(filename)
  os.makedirs(folder, exist_ok=True)

  metadata = {
      "filename": filename,
      "title": data.get("title", ""),
      "chunks_count": int(data.get("chunks_count", 0)),
      "file_size": int(data.get("file_size", 0)),
      "added_at": data.get("added_at", ""),
      "history": data.get("history", []),
      "progress": data.get("progress", {"flashcards": {}, "quiz_attempts": []}),
  }

  with open(_doc_metadata_path(filename), "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)


def _load_persisted_documents():
  documents_db.clear()
  if not os.path.exists(DOCUMENTS_DIR):
    return

  for entry in os.listdir(DOCUMENTS_DIR):
    folder = os.path.join(DOCUMENTS_DIR, entry)
    if not os.path.isdir(folder):
      continue

    metadata_path = os.path.join(folder, "metadata.json")
    index_path = os.path.join(folder, "faiss_index")

    if not os.path.exists(metadata_path):
      continue

    try:
      with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

      filename = metadata.get("filename")
      if not filename or not os.path.exists(index_path):
        continue

      vector_db = FAISS.load_local(
          index_path,
          embed_model,
          allow_dangerous_deserialization=True,
      )

      documents_db[filename] = {
          "vector_db": vector_db,
          "history": metadata.get("history", []),
          "chunks_count": int(metadata.get("chunks_count", 0)),
          "full_text": "",
          "title": metadata.get("title", ""),
          "file_size": int(metadata.get("file_size", 0)),
          "added_at": metadata.get("added_at", ""),
          "progress": metadata.get("progress", {"flashcards": {}, "quiz_attempts": []}),
      }
    except Exception as e:
      print(f"⚠️ Impossible de restaurer '{entry}': {e}")


_load_persisted_documents()



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
    return """<div class='progress-empty'><b>Votre progression commencera ici.</b><br>Importez un cours pour créer votre premier suivi. Tant qu'aucun cours n'est chargé, la progression reste à <b>0 %</b>.</div>"""
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
def ajouter_et_indexer_pdf(file_path):
  if not file_path:
    return (
        "⚠️ Aucun fichier sélectionné.",
        gr.update(choices=list(documents_db.keys())),
        generer_html_documents(),
    )

  filename = os.path.basename(file_path)

  try:
    with fitz.open(file_path) as doc:
      texte_complet = "".join(page.get_text() for page in doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80
    )
    chunks = text_splitter.split_text(texte_complet)

    vector_db = FAISS.from_texts(chunks, embed_model)

    previous = documents_db.get(filename, {})
    same_content = previous.get("file_size") == os.path.getsize(file_path) and previous.get("chunks_count") == len(chunks)
    documents_db[filename] = {
        "vector_db": vector_db,
        "history": previous.get("history", []) if same_content else [],
        "progress": previous.get("progress", _default_progress()) if same_content else _default_progress(),
        "chunks_count": len(chunks),
        "full_text": texte_complet,
        "title": _extract_document_title(texte_complet),
        "file_size": os.path.getsize(file_path),
        "added_at": previous.get("added_at") if same_content and previous.get("added_at") else datetime.now().isoformat(timespec="seconds"),
    }

    _save_document(filename)
    docs_list = list(documents_db.keys())

    return (
        f"✅ '{filename}' indexé avec succès ({len(chunks)} fragments).",
        gr.update(choices=docs_list, value=filename),
        generer_html_documents(),
    )
  except Exception as e:
    return (
        f"❌ Erreur lors de l'indexation : {str(e)}",
        gr.update(choices=list(documents_db.keys())),
        generer_html_documents(),
    )


def generer_html_documents():
  if not documents_db:
    return (
        "<p style='color: #94a3b8;'>Aucun document chargé pour le moment.</p>"
    )

  html = "<div style='display: flex; flex-direction: column; gap: 10px;'>"
  for name, data in documents_db.items():
    html += f"""
        <div style='background-color: #161e2e; padding: 12px 16px; border-radius: 8px; border: 1px solid #243046; display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <b style='color: #f8fafc;'>📄 {name}</b><br>
                <small style='color: #94a3b8;'>PDF • {data['chunks_count']} fragments indexés</small>
            </div>
            <span style='background-color: #334155; color: #cbd5e1; padding: 3px 8px; border-radius: 12px; font-size: 12px;'>Indexé</span>
        </div>
        """
  html += "</div>"
  return html


def _chat_history_for_ui(doc_name):
  data = documents_db.get(doc_name, {})
  raw = data.get("history", [])
  result = []
  for item in raw:
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


def _appel_gemini_securise(user_prompt, system_instruction, max_retries=2):
  """Exécute l'appel Gemini avec tentative automatique en cas d'erreur 503/429.

  Bascule sur le modèle de secours si nécessaire.
  """
  models_to_try = _models_chain()

  for model_target in models_to_try:
    for attempt in range(max_retries):
      try:
        _wait_before_gemini_request()
        return client.models.generate_content_stream(
            model=model_target,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=500,
            ),
        )
      except APIError as e:
        if e.code in [503, 429]:
          sleep_time = (1.5**attempt) + random.uniform(0.2, 0.6)
          print(
              f"⚠️ Modèle {model_target} saturé ({e.code}). Tentative"
              f" {attempt+1}/{max_retries} dans {sleep_time:.1f}s..."
          )
          time.sleep(sleep_time)
        else:
          raise e

  raise Exception(
      "Tous les modèles Gemini sont actuellement saturés. Veuillez réessayer"
      " dans 10 secondes."
  )


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
      historique[-1]["content"] = (
          f"Le titre du document est : **{title}**"
          if title else "⚠️ Le titre du document est indisponible."
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
        "Pour les formules, utilise un format lisible compatible Markdown/LaTeX."
    )
    user_prompt = (
        f"CONTEXTE:\n{contexte_brut}\n\nDEMANDE UTILISATEUR:\n{message_str}"
    )
    input_tokens_est = _estimate_tokens(user_prompt)

    stream_response = _appel_gemini_securise(
        user_prompt, system_instruction
    )
    session_usage["input_tokens"] += input_tokens_est

    # Streaming visuel lissé : Gemini envoie des fragments de tailles variables.
    # On regroupe les fragments très courts avant de rafraîchir l'interface afin
    # d'éviter un effet de clignotement tout en gardant une impression naturelle.
    reponse = ""
    buffer = ""
    last_ui_update = time.monotonic()
    MIN_STREAM_CHARS = 10
    MAX_STREAM_DELAY = 0.05

    for chunk in stream_response:
      texte_chunk = getattr(chunk, "text", "") or ""
      if not texte_chunk:
        continue

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

    # Dernier état garanti après la fin du flux.
    yield _chat_history_for_ui(doc_selectionne), ""

  except Exception as e:
    historique[-1]["content"] = f"⚠️ {str(e)}"
    _save_history(doc_selectionne)
    yield _chat_history_for_ui(doc_selectionne), ""


# --- 5. NOUVELLE INTERFACE AKORI ---
# Interface orientée "AI Study Assistant" : cours -> révision -> flashcards/quiz -> assistant -> progression.

TOKEN_SESSION_LIMIT = 50000


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
        <div class='guide-item'><b>1. Chargez vos dossiers</b><span>Glissez vos PDF dans <i>Mes dossiers</i> pour activer l'indexation FAISS.</span></div>
        <div class='guide-item'><b>2. Lancement roulette</b><span>Générez au moins 7 flashcards. Les questions défilent automatiquement en boucle.</span></div>
        <div class='guide-item'><b>3. Clic &amp; réponse effacée</b><span>Cliquez sur la carte pour <b>stopper/relancer</b>. La réponse s'efface à chaque relance.</span></div>
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
    <div class='course-grid'>{cards}</div>
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
    for model_target in _models_chain():
        for attempt in range(attempts_per_model):
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

    if last_error is not None and getattr(last_error, "code", None) == 429:
        raise RuntimeError(
            "Quota ou limite temporaire Gemini atteinte. Attendez un peu avant de relancer la génération."
        ) from last_error
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


def flashcard_view(cards, index=0, revealed=False, doc_name=None, results=None):
    cards = normalize_flashcards(cards)
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
        answer_html = f"<div style='margin-top:20px;padding:15px;background:#eef2ff;border-radius:12px;border-left:4px solid #4f46e5;color:#1e293b;'><b>💡 Réponse :</b><br><br>{answer}</div>"
    else:
        answer_html = "<div style='margin-top:20px;text-align:center;color:#94a3b8;'>Cliquez sur le bouton <b>« Afficher la réponse »</b>.</div>"

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
  <div class="badge">🎯 CARTE {index + 1} / {total}</div>
  <div id="status" class="status">🎰 Roulette active</div>
  <div id="question" class="q-text">{question}</div>
  <div style="width:100%;">{answer_html}</div>
  <div id="hint" class="hint">Cliquez sur la carte pour l'arrêter.</div>
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
  statusEl.innerText = "🎰 Roulette active";
  statusEl.style.background = "#fef3c7"; statusEl.style.color = "#b45309";
  hintEl.innerText = "Cliquez sur la carte pour l'arrêter.";
  timer = setInterval(() => {{ qEl.innerText = cards[Math.floor(Math.random() * cards.length)].question; }}, 120);
}}
function stop() {{
  running = false;
  if (timer) clearInterval(timer);
  statusEl.innerText = "⏸ Roulette arrêtée";
  statusEl.style.background = "#dcfce7"; statusEl.style.color = "#15803d";
  hintEl.innerText = isRevealed ? "Passez à la carte suivante." : "Cliquez sur le bouton pour afficher la réponse.";
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


def _flash_generate_fail(message):
    return [], 0, False, flashcard_view([], 0, False), message, gr.update(visible=False), gr.update(value=GENERATE_LABEL)


def flashcard_generate_handler(doc_name):
    """Génère une NOUVELLE série (7 flashcards minimum) à chaque appel."""
    if not doc_name or doc_name not in documents_db:
        return _flash_generate_fail("⚠️ Aucun document sélectionné.")

    document = documents_db[doc_name]
    context = "\n\n".join(preparer_contexte_global(document))
    if not context.strip():
        return _flash_generate_fail("❌ Document vide.")

    cards = []
    last_error = None
    for attempt in range(2):
        try:
            prompt = f"""CONTEXTE DU COURS:\n{context}\n\n
Crée {MIN_FLASHCARDS + 1} flashcards (questions/réponses courtes) pour réviser les concepts clés de ce cours.
Il en faut au moins {MIN_FLASHCARDS}, avec des questions toutes différentes.
Retourne UNIQUEMENT un JSON valide avec la structure exacte suivante :
{{"flashcards": [{{"question": "...", "answer": "..."}}]}}"""
            raw = _gemini_structured(
                prompt,
                "Tu es AKORI, assistant académique. Base-toi uniquement sur le contexte fourni et n'invente rien.",
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

    if last_error is not None and not cards:
        return _flash_generate_fail(f"❌ Erreur : {last_error}")
    if len(cards) < MIN_FLASHCARDS:
        return _flash_generate_fail(
            f"⚠️ Seulement {len(cards)} flashcard(s) obtenue(s) : il en faut au moins {MIN_FLASHCARDS}. Relancez la génération.")

    document["flashcards"] = cards
    view = flashcard_view(cards, 0, False, doc_name, get_flashcard_results(doc_name, cards))
    return (cards, 0, False, view, f"✅ {len(cards)} flashcards prêtes !",
            gr.update(visible=True), gr.update(value=GENERATE_LABEL))


def flashcard_reveal_handler(cards, index, doc_name):
    cards = normalize_flashcards(cards)
    return True, flashcard_view(cards, index, True, doc_name, get_flashcard_results(doc_name, cards))


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
      <p style="margin-top:18px;">Cliquez sur <b>« 🔄 Régénérer (7 minimum) »</b> pour obtenir une nouvelle série.</p>
    </div>
    """


def flashcard_mark_handler(cards, index, mark_type, doc_name):
    cards = normalize_flashcards(cards)
    if not cards:
        return 0, False, flashcard_view([], 0, False), gr.update(visible=False), gr.update()
    index = int(index)
    if index >= len(cards):
        # Session déjà terminée : rien d'autre à faire que régénérer.
        return index, False, gr.update(), gr.update(visible=False), gr.update(value=REGENERATE_LABEL)
    save_flashcard_result(doc_name, cards[index]["question"], mark_type)
    next_index = index + 1
    results = get_flashcard_results(doc_name, cards)
    if next_index >= len(cards):
        st = flashcard_statistics(cards, results)
        return (next_index, False,
                flashcard_end_game_view(doc_name, st["total"], st["known"], st["review"]),
                gr.update(visible=False), gr.update(value=REGENERATE_LABEL))
    return (next_index, False, flashcard_view(cards, next_index, False, doc_name, results),
            gr.update(visible=True), gr.update())


# ============================================================
# QUIZ — une question à la fois, réponses cliquables
# ============================================================
def generate_quiz_v12(doc_name, count=5):
    if not doc_name or doc_name not in documents_db:
        return [], "⚠️ Sélectionnez d'abord un cours."
    context = "\n\n".join(preparer_contexte_global(documents_db[doc_name]))
    prompt = f"""CONTEXTE DU COURS:\n{context}\n\nCrée exactement {count} questions QCM de révision. Une seule bonne réponse par question.\nRetourne uniquement un JSON valide: {{\"questions\":[{{\"question\":\"...\",\"options\":[\"...\",\"...\",\"...\",\"...\"],\"answer\":0,\"explanation\":\"...\",\"topic\":\"notion abordée (2-3 mots)\"}}]}}\nanswer est l'index 0-3 de la bonne option."""
    try:
        data = _extract_json_from_text(_gemini_structured(prompt, "Tu es AKORI, assistant académique. Base-toi uniquement sur le contexte fourni et n'invente rien."))
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


def quiz_generate_handler(doc_name):
    qs, status = generate_quiz_v12(doc_name)
    return (qs, 0, False, quiz_view(qs, 0, False, []), status,
            _quiz_radio(qs, 0), [None] * len(qs), gr.update(value=NEXT_LABEL))


def quiz_restart_handler(questions):
    questions = questions or []
    return (questions, 0, False, quiz_view(questions, 0, False, []), "🔁 Quiz relancé.",
            _quiz_radio(questions, 0), [None] * len(questions), gr.update(value=NEXT_LABEL))


def quiz_validate_handler(questions, index, choice, validated, answers):
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
    return (len(questions), True, quiz_result_view(questions, answers), f"🎯 Quiz terminé : {score}/{len(questions)}.",
            gr.update(choices=[], value=None, visible=False), gr.update(value=NEXT_LABEL))


def history_view(doc_name):
    if not doc_name or doc_name not in documents_db:
        return "<div class='empty-study'>Aucun historique pour le moment.</div>"
    history = documents_db[doc_name].get("history", [])
    blocks = []
    for item in history:
        role = item.get("role", "")
        content = _escape_html(item.get("content", ""))
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
#sidebar .sidebar-note { color:#8993a7!important;font-size:10px;line-height:1.45;margin-top:18px; }

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


/* ===== Menu latéral : fixe, largeur définie, rail d'icônes + survol ===== */
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

/* NB : Gradio recopie chaque règle avec un préfixe ".gradio-container … .contain" qui lui donne
   plus de poids. Les états (replié / survol) passent donc par des variables CSS. */
:root { --sb-open:264px; --sb-rail:72px; --sb-left:max(16px, calc((100vw - 1280px)/2 + 32px)); }
body { --sb-w:var(--sb-open); --main-ml:calc(var(--sb-open) + 16px); }
body.sidebar-collapsed { --sb-w:var(--sb-rail); --main-ml:calc(var(--sb-rail) + 16px); }

/* mise en page stable : toutes les pages ont la même largeur et la barre de défilement est toujours là */
html { overflow-y:scroll!important; }
.gradio-container:has(#main-column), .main:has(#main-column), .wrap:has(#main-column), .contain:has(#main-column),
.column:has(> .row > #main-column), .row:has(> #main-column) { width:100%!important; }
.row:has(> #main-column) { flex-wrap:nowrap!important; }

#sidebar {
  position:fixed!important;top:14px!important;left:var(--sb-left)!important;
  width:var(--sb-w)!important;min-width:var(--sb-w)!important;max-width:var(--sb-w)!important;flex:none!important;
  height:calc(100vh - 28px)!important;max-height:calc(100vh - 28px)!important;min-height:0!important;
  display:flex!important;flex-direction:column!important;gap:2px!important;--layout-gap:2px;
  padding-left:var(--sb-pad,16px)!important;padding-right:var(--sb-pad,16px)!important;
  overflow-x:hidden!important;overflow-y:auto!important;scrollbar-width:thin;
  box-shadow:var(--sb-shadow, 0 8px 30px rgba(42,55,90,.06))!important;z-index:60;
  transition:width .3s cubic-bezier(.4,0,.2,1), min-width .3s cubic-bezier(.4,0,.2,1),
             max-width .3s cubic-bezier(.4,0,.2,1), padding .3s ease, box-shadow .3s ease;
}
#main-column {
  flex:1 1 0!important;min-width:0!important;width:auto!important;
  margin-left:var(--main-ml)!important;
  transition:margin-left .3s cubic-bezier(.4,0,.2,1);
  overflow-wrap:break-word;word-break:normal;
}
#main-tabs .tabitem { min-height:calc(100vh - 150px)!important;width:100%!important; }

/* barre du haut : le titre et le sélecteur de cours se replient proprement, sans casser les mots */
#topbar { flex-wrap:wrap!important;gap:8px 16px!important;align-items:center!important; }
#topbar > * { flex:1 1 240px!important;min-width:min(240px,100%)!important; }
#topbar h3 { white-space:normal!important;word-break:normal!important;overflow-wrap:normal!important;margin:0!important; }

#sidebar .block { padding:0!important;margin:0!important;border:0!important;min-height:0!important; }
#sidebar .navbtn {
  display:flex!important;align-items:center!important;justify-content:var(--nav-jc, flex-start)!important;gap:var(--nav-gap, 12px)!important;
  font-size:var(--nav-fs, 13px)!important;min-height:40px!important;margin:1px 0!important;padding:0 var(--nav-px, 12px)!important;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:none!important;
}
#sidebar .navbtn::before { display:inline-block;width:24px;flex:none;text-align:center;font-size:17px;line-height:1; }
#sidebar #nav-settings { margin-top:auto!important;border-top:1px solid var(--ak-line)!important; }

/* textes du menu : toujours entiers, jamais coupés */
#sidebar .brand { margin:2px 4px 14px!important;align-items:center;justify-content:var(--brand-jc, flex-start)!important; }
#sidebar .brand > div:last-child { min-width:0;display:var(--lab-d, block)!important; }
#sidebar .brand { width:100%; }
#sidebar .brand-mark { flex:none!important;width:42px!important;min-width:42px!important;height:42px!important; }
#sidebar .brand-name { white-space:nowrap; }
#sidebar .brand-sub { white-space:normal!important;overflow-wrap:normal;word-break:normal;font-size:10px!important;line-height:1.35!important; }
#sidebar .nav-title { white-space:nowrap;display:var(--lab-d, block)!important; }
#sidebar .sidebar-note { white-space:normal!important;overflow-wrap:normal;word-break:normal;margin-top:8px!important;font-size:10.5px!important;line-height:1.45!important;display:var(--lab-d, block)!important; }

#sidebar #sidebar-toggle {
  width:44px!important;min-width:44px!important;max-width:44px!important;height:40px!important;flex:none!important;
  padding:0!important;font-size:20px!important;align-self:var(--tog-as, flex-start)!important;margin:0 0 6px 2px!important;
  display:flex!important;align-items:center!important;justify-content:center!important;
}

/* menu replié : seules les icônes restent visibles ... */
body.sidebar-collapsed #sidebar:not(:hover) {
  --sb-pad:10px; --nav-fs:0px; --nav-gap:0px; --nav-px:0px; --nav-jc:center;
  --lab-d:none; --brand-jc:center; --tog-as:center;
}
body.sidebar-collapsed #sidebar:hover { --sb-w:var(--sb-open); --sb-shadow:0 18px 50px rgba(30,41,80,.22); }

/* écrans étroits : rail fixe, ouverture en surimpression */
@media (max-width:900px) {
  :root { --sb-left:10px; }
  body { --sb-w:var(--sb-rail); --main-ml:calc(var(--sb-rail) + 12px); }
  body.sidebar-collapsed { --main-ml:calc(var(--sb-rail) + 12px); }
  body:not(.sidebar-collapsed) #sidebar { --sb-w:var(--sb-open); --sb-shadow:0 18px 50px rgba(30,41,80,.22); }
  body.sidebar-collapsed #sidebar:not(:hover), body:not(.sidebar-collapsed) #sidebar { --sb-pad:10px; }
  body:not(.sidebar-collapsed) #sidebar { --sb-pad:16px; }
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
"""


THEME_APPLY_JS = r"""
(choice) => {
  const dark = String(choice).includes('Sombre');
  document.body.classList.toggle('akori-dark', dark);
  document.body.classList.toggle('dark', dark);
  try { localStorage.setItem('akori-theme', dark ? 'dark' : 'light'); } catch (e) {}
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
  if (!window.__akoriSidebarBound) {
    window.__akoriSidebarBound = true;
    // Le menu est fixe : on l'aligne sur le bord gauche réel de la grille Gradio.
    const alignSidebar = () => {
      const main = document.querySelector('#main-column');
      if (!main || !main.parentElement) return;
      const left = main.parentElement.getBoundingClientRect().left;
      document.documentElement.style.setProperty('--sb-left', Math.max(8, Math.round(left)) + 'px');
    };
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
  return dark ? '🌙 Sombre' : '☀️ Clair';
}
"""

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

theme_akori = gr.themes.Soft(primary_hue="indigo", secondary_hue="purple", neutral_hue="slate")

with gr.Blocks(title="AKORI — AI Study Assistant") as demo:
    flashcards_state = gr.State([])
    flash_index = gr.State(0)
    flash_revealed = gr.State(False)
    quiz_state = gr.State([])
    quiz_index = gr.State(0)
    quiz_validated = gr.State(False)
    quiz_answers = gr.State([])

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
            nav_settings = gr.Button("Paramètres", elem_id="nav-settings", elem_classes="navbtn")
            gr.Markdown("<div class='sidebar-note'>Vos documents, historiques et progressions sont conservés localement dans AKORI.</div>")

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
                    gr.Markdown("### Apparence")
                    theme_choice = gr.Radio(
                        choices=["☀️ Clair", "🌙 Sombre"], value="☀️ Clair",
                        label="Thème de l'interface", elem_id="theme-choice",
                    )
                    gr.Markdown("Votre choix est mémorisé dans ce navigateur. Le bouton ☰ en haut à gauche ouvre et ferme le menu latéral.")

                with gr.Tab("Historique", id="history"):
                    gr.Markdown("## Historique du cours actif")
                    history_box = gr.HTML(history_view(doc_selector.value))

    # Upload : la sélection du PDF déclenche directement extraction + chunking + indexation.
    def upload_and_refresh(file_path):
        file_path = getattr(file_path, "name", file_path)
        status, selector_update, _ = ajouter_et_indexer_pdf(file_path)
        selected = os.path.basename(file_path) if file_path else None
        # The upload event updates every dependent view in one transaction.
        # Keep the output count exactly aligned with the Gradio listener.
        return (
            status, selector_update, documents_html_v16(selected),
            dashboard_html(selected), course_overview_html(selected),
            course_overview_html(selected), global_progress_html(selected),
            progress_detail_html(selected), _chat_history_for_ui(selected),
            history_view(selected)
        )

    # Stage 1: immediate visual feedback. Stage 2: extraction + chunking +
    # embeddings + FAISS, with the selected PDF becoming the active course.
    upload_event = pdf_input.upload(
        lambda: "⏳ Analyse du PDF… extraction, découpage et indexation en cours.",
        inputs=None, outputs=[upload_status], queue=False
    )
    upload_event.then(
        upload_and_refresh, inputs=[pdf_input],
        outputs=[upload_status, doc_selector, docs_html, home_html, course_info,
                 review_cards, global_progress_output, progress_output, chatbot, history_box],
        queue=True, show_progress="minimal", concurrency_limit=1
    )

    def refresh_all(d):
        return dashboard_html(d), documents_html_v16(d), course_overview_html(d), course_overview_html(d), global_progress_html(d), progress_detail_html(d), _chat_history_for_ui(d), history_view(d)

    doc_selector.change(refresh_all, inputs=[doc_selector], outputs=[home_html, docs_html, course_info, review_cards, global_progress_output, progress_output, chatbot, history_box])

    # Navigation vers les onglets
    nav_home.click(lambda: gr.Tabs(selected="home"), outputs=tabs)
    nav_courses.click(lambda: gr.Tabs(selected="courses"), outputs=tabs)
    nav_review.click(lambda: gr.Tabs(selected="review"), outputs=tabs)
    nav_flash.click(lambda: gr.Tabs(selected="flashcards"), outputs=tabs)
    nav_quiz.click(lambda: gr.Tabs(selected="quiz"), outputs=tabs)
    nav_summary.click(lambda: gr.Tabs(selected="summary"), outputs=tabs)
    nav_assistant.click(lambda: gr.Tabs(selected="assistant"), outputs=tabs)
    nav_progress.click(lambda: gr.Tabs(selected="progress"), outputs=tabs)
    nav_reviewq.click(lambda: gr.Tabs(selected="reviewq"), outputs=tabs)
    nav_history.click(lambda: gr.Tabs(selected="history"), outputs=tabs)
    nav_settings.click(lambda: gr.Tabs(selected="settings"), outputs=tabs)
    start_review.click(lambda: gr.Tabs(selected="review"), outputs=tabs)
    review_flash.click(lambda: gr.Tabs(selected="flashcards"), outputs=tabs)
    review_quiz.click(lambda: gr.Tabs(selected="quiz"), outputs=tabs)
    review_summary.click(lambda: gr.Tabs(selected="summary"), outputs=tabs)
    review_chat.click(lambda: gr.Tabs(selected="assistant"), outputs=tabs)

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
    )

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
    )

    # Flashcards
    generate_flash.click(
        lambda: "⏳ Génération des flashcards…", outputs=[flash_status], queue=False,
    ).then(
        flashcard_generate_handler,
        inputs=[doc_selector],
        outputs=[flashcards_state, flash_index, flash_revealed, flash_view, flash_status, flash_actions, generate_flash],
        show_progress="minimal",
        concurrency_limit=1,
    )
    flash_reveal.click(
        flashcard_reveal_handler,
        inputs=[flashcards_state, flash_index, doc_selector],
        outputs=[flash_revealed, flash_view],
        queue=False,
    )
    flash_known.click(
        lambda c, i, d: flashcard_mark_handler(c, i, "known", d),
        inputs=[flashcards_state, flash_index, doc_selector],
        outputs=[flash_index, flash_revealed, flash_view, flash_actions, generate_flash],
        queue=False,
    ).then(lambda d: (global_progress_html(d), progress_detail_html(d)), inputs=[doc_selector], outputs=[global_progress_output, progress_output], queue=False)
    flash_review.click(
        lambda c, i, d: flashcard_mark_handler(c, i, "review", d),
        inputs=[flashcards_state, flash_index, doc_selector],
        outputs=[flash_index, flash_revealed, flash_view, flash_actions, generate_flash],
        queue=False,
    ).then(lambda d: (global_progress_html(d), progress_detail_html(d)), inputs=[doc_selector], outputs=[global_progress_output, progress_output], queue=False)

    # Quiz
    quiz_outputs = [quiz_state, quiz_index, quiz_validated, quiz_view_box, quiz_status, quiz_choice, quiz_answers, quiz_next]
    generate_quiz_btn.click(
        lambda: "⏳ Génération du quiz…", outputs=[quiz_status], queue=False
    ).then(quiz_generate_handler, inputs=[doc_selector], outputs=quiz_outputs, show_progress="minimal")
    quiz_restart.click(quiz_restart_handler, inputs=[quiz_state], outputs=quiz_outputs, queue=False)
    quiz_validate.click(
        quiz_validate_handler,
        inputs=[quiz_state, quiz_index, quiz_choice, quiz_validated, quiz_answers],
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
    theme_choice.input(None, theme_choice, None, js=THEME_APPLY_JS)
    menu_btn.click(None, None, None, js=SIDEBAR_TOGGLE_JS)
    demo.load(None, None, theme_choice, js=THEME_INIT_JS)

    # Résumé — conserve le comportement existant, mais l'affiche dans son propre espace.
    def summary_handler(d):
        if not d or d not in documents_db:
            return "⚠️ Sélectionnez d'abord un cours."
        context = "\n\n".join(preparer_contexte_global(documents_db[d]))
        prompt = f"CONTEXTE DU COURS:\n{context}\n\nDEMANDE: Fais un résumé synthétique des points clés principaux de ce document."
        try:
            response = _gemini_structured(prompt, "Tu es AKORI. Résume uniquement les informations présentes dans le contexte fourni. Retourne un JSON {\"summary\":\"...\"}.")
            data = _extract_json_from_text(response)
            return data.get("summary", response) if isinstance(data, dict) else response
        except Exception as e:
            return f"⚠️ {e}"
    summary_btn.click(
        lambda: "⏳ Génération du résumé…", outputs=summary_output, queue=False
    ).then(summary_handler, inputs=doc_selector, outputs=summary_output)

    demo.load(lambda: (dashboard_html(doc_selector.value), documents_html_v16(doc_selector.value), course_overview_html(doc_selector.value), course_overview_html(doc_selector.value), global_progress_html(doc_selector.value), progress_detail_html(doc_selector.value), _chat_history_for_ui(doc_selector.value), history_view(doc_selector.value)), outputs=[home_html, docs_html, course_info, review_cards, global_progress_output, progress_output, chatbot, history_box])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(
        share=True,
        debug=False,
        show_error=False,
        theme=theme_akori,
        css=custom_css,
        js=AKORI_NAV_JS,
    )
