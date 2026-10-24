---
title: AKORI
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AKORI — Assistant Knowledge Organized to Revise Intelligently

Assistant de révision à partir de vos PDF : assistant qui cite ses sources (RAG : BGE-M3 + FAISS + Gemini), flashcards, quiz, colle orale, mindmap, résumé, plan de révision et suivi des lacunes. Un seul fichier : `akori.py`.

## Lancer

| Où | Commande |
|---|---|
| Google Colab | exécuter `akori.py` (lien public `gradio.live`, données sur Drive) |
| Serveur / PC | `pip install -r requirements.txt` puis `python akori.py` (port 7860) |
| Docker | `docker build -t akori .` puis `docker run -p 7860:7860 -v akori:/data -e GEMINI_API_KEY=... -e AKORI_ADMIN_PASSWORD=... akori` |

## Déployer pour tout le monde (téléphone compris) : Hugging Face Spaces

1. Créer un compte sur huggingface.co, puis **New Space** → SDK **Docker** → *Blank*.
2. Envoyer ce dépôt dans le Space (`git remote add space https://huggingface.co/spaces/<compte>/akori` puis `git push space main`), ou connecter le dépôt GitHub.
3. Dans **Settings → Variables and secrets**, ajouter les *secrets* :
   - `GEMINI_API_KEY` : votre clé Gemini (commence par `AIza`) ; facultatif : `GEMINI_API_KEYS` (plusieurs clés séparées par des virgules) pour répartir la charge ;
   - `AKORI_ADMIN_PASSWORD` : mot de passe du compte `admin` (sinon un mot de passe aléatoire est écrit dans les *logs* au premier démarrage).
4. **Données persistantes** : sans disque, tout est effacé à chaque redémarrage du Space. Dans **Settings → Persistent storage**, activer un stockage (payant) : il est monté sur `/data`, AKORI l'utilise automatiquement.
5. Le Space fournit une adresse `https://<compte>-akori.hf.space` à partager : elle s'ouvre sur n'importe quel téléphone, sans installation. Sur le téléphone, « Ajouter à l'écran d'accueil » crée une icône AKORI.

Variables utiles : `AKORI_SHARE=1` (lien Gradio public hors Colab), `AKORI_EMAIL_CHECK=0` (ne pas vérifier le domaine des e-mails), `AKORI_OCR=tesseract|gemini|off`, `AKORI_EDITOR=<nom affiché dans À propos>`.

Remarque : sur un processeur sans GPU, BGE-M3 indexe lentement un gros PDF (quelques minutes) ; les réponses, elles, restent rapides. Le plan gratuit de Hugging Face (2 vCPU, 16 Go) suffit pour quelques dizaines d'utilisateurs.
