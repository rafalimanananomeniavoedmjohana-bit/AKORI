"""Régénère build.py, demo.py, notebooks/AKORI_Build.ipynb et notebooks/AKORI_Demo.ipynb : une seule cellule de code chacun,
contenant tout akori.py. À relancer après chaque modification de akori.py :  python tools/make_notebooks.py"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = open(os.path.join(ROOT, "akori.py"), encoding="utf-8").read()

PREP = '''!pip -q install gradio google-genai pymupdf langchain-community langchain-text-splitters faiss-cpu sentence-transformers
import os
from google.colab import drive, userdata
drive.mount("/content/drive")                         # tout est enregistré dans MyDrive/AKORI

def secret(name, default=""):
    try:
        return userdata.get(name) or default
    except Exception:
        return default

os.environ["GEMINI_API_KEY"] = secret("GEMINI_API_KEY")   # Colab > icône clé > Secrets
'''
BUILD_ENV = '''os.environ["AKORI_MODE"] = "build"
os.environ["AKORI_BUILD_USER"] = "demo@akori.app"        # compte qui reçoit les cours préparés
os.environ["AKORI_BUILD_PASSWORD"] = "ChangezMoi123"     # 8 caractères minimum
os.environ["AKORI_BUILD_LANGS"] = "fr,en"                # langues des résumés pré-générés
# PDF à indexer : MyDrive/AKORI/a_indexer/
'''
DEMO_ENV = '''for name in ("AKORI_ADMIN_PASSWORD", "AKORI_RESET_ADMIN", "AKORI_MIN_SIMILARITY", "AKORI_EMBED_MODEL"):
    if secret(name):
        os.environ[name] = secret(name)
'''

def notebook(title, env):
    src = PREP + env + "\n# ======================= code AKORI =======================\n" + CODE
    return {"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [title]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)},
    ], "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}

os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
for name, title, env in (("Build", "# AKORI · BUILD (une seule fois)\nIndexe les PDF de `MyDrive/AKORI/a_indexer/` et pré-génère résumé, flashcards et quiz.", BUILD_ENV),
                         ("Demo", "# AKORI · DEMO\nLance l'application avec les cours déjà préparés par Build.", DEMO_ENV)):
    with open(os.path.join(ROOT, "notebooks", f"AKORI_{name}.ipynb"), "w", encoding="utf-8") as f:
        json.dump(notebook(title, env), f, ensure_ascii=False, indent=1)
print("notebooks générés")

# build.py / demo.py : scripts autonomes (tout le code AKORI), à lancer avec `python build.py` / `python demo.py`
BUILD_HEAD = ("# AKORI · BUILD : à lancer une seule fois (python build.py). Fichier généré par tools/make_notebooks.py.\n"
              "# Indexe les PDF de MyDrive/AKORI/a_indexer/ et pré-génère résumé, flashcards et quiz.\n"
              "import os\nos.environ.setdefault('AKORI_MODE', 'build')\n"
              "os.environ.setdefault('AKORI_BUILD_USER', 'demo@akori.app')\n"
              "os.environ.setdefault('AKORI_BUILD_PASSWORD', 'ChangezMoi123')\n\n")
DEMO_HEAD = ("# AKORI · DEMO : lance l'application (python demo.py). Fichier généré par tools/make_notebooks.py.\n\n")
for fname, head in (("build.py", BUILD_HEAD), ("demo.py", DEMO_HEAD)):
    with open(os.path.join(ROOT, fname), "w", encoding="utf-8") as f:
        f.write(head + CODE)
print("build.py et demo.py générés")
