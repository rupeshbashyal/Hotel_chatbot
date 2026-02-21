# Hotel Sunrise Chatbot

A multilingual intent-based chatbot for **Hotel Sunrise**, built with scikit-learn and Django. It answers common guest questions about booking, WiFi, parking, menu, check-in/check-out times, and supports English and Nepali.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-line interface (CLI)](#command-line-interface-cli)
  - [Web interface (Django)](#web-interface-django)
- [How It Works](#how-it-works)
- [Training Data](#training-data)
- [API](#api)
- [Configuration](#configuration)
- [Security Notes](#security-notes)
- [License](#license)

---

## Features

| Feature | Description |
|--------|-------------|
| **Intent classification** | TF-IDF + Logistic Regression to map user messages to intents (greeting, booking, wifi, etc.). |
| **Spelling / typo handling** | Uses RapidFuzz to correct minor typos against the training vocabulary (score > 80). |
| **Confidence threshold** | Replies with *"Sorry, I didn't understand..."* when model confidence is below 0.25. |
| **Multilingual** | Training examples in English and Nepali (e.g. नमस्ते, wifi छ?, कोठा बुक गर्न मिल्छ?). |
| **Varied greetings** | Random greeting responses for a more natural feel. |
| **Chat logging** | CLI conversations are appended to `chatlog.txt`. |
| **Web UI** | Django app with a simple chat page and CSRF-secured POST endpoint. |

---

## Project Structure

```
Hotel_chatbot/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore
├── bot.py                     # Core chatbot logic (CLI + get_response)
├── data.json                  # Intent training data
├── chatlog.txt                # CLI chat log (optional to commit)
└── chatbot_project/          # Django web app
    ├── manage.py              # Adds project root to path, runs Django
    ├── chat/                   # Django app
    │   ├── views.py            # home, get_bot_response
    │   └── urls.py             # /, /get/
    ├── chatbot_project/
    │   ├── settings.py         # Django settings, PROJECT_ROOT for bot import
    │   └── urls.py             # Root URLconf
    └── templates/
        └── index.html         # Chat UI (CSRF cookie + fetch)
```

---

## Requirements

- **Python:** 3.10+ recommended  
- **Dependencies:** See `requirements.txt`

| Package | Purpose |
|---------|---------|
| Django | Web framework for the chat UI and API. |
| scikit-learn | TF-IDF vectorizer and Logistic Regression for intent classification. |
| rapidfuzz | Fuzzy string matching for spelling/typo correction. |

---

## Installation

1. **Clone or download** the project and go to the project root:
   ```bash
   cd Hotel_chatbot
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   # Windows (cmd)
   venv\Scripts\activate.bat
   # Linux / macOS
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Django (web app) only – run migrations:**
   ```bash
   cd chatbot_project
   python manage.py migrate
   ```

---

## Usage

### Command-line interface (CLI)

Run the bot in the terminal. Use `help` for topic hints and `exit` to quit.

```bash
# From project root (Hotel_chatbot)
python bot.py
```

- **help** – Prints: *"You can ask about booking, wifi, parking, menu, check-in/check-out time."*
- **exit** – Ends the session.

Conversations are appended to `chatlog.txt` in the project root.

### Web interface (Django)

1. Start the server (from project root or from `chatbot_project`):
   ```bash
   cd chatbot_project
   python manage.py runserver
   ```
   Or, using the venv Python explicitly:
   ```bash
   path\to\Hotel_chatbot\venv\Scripts\python.exe manage.py runserver
   ```

2. Open **http://127.0.0.1:8000/** in your browser.

3. Type messages and click **Send** (or press Enter). The same `get_response` logic from `bot.py` is used.

**Note:** The first page load sets the CSRF cookie; if you previously saw “Sorry, something went wrong”, refresh the page and try again.

---

## How It Works

1. **Training (at import time)**  
   - `data.json` is loaded; each key is an **intent**, each value is a list of **example phrases**.  
   - Text is cleaned (lowercase, strip).  
   - TF-IDF features (unigrams + bigrams) are computed with `TfidfVectorizer(ngram_range=(1, 2))`.  
   - A **Logistic Regression** model is trained to predict intent from these features.  
   - A **vocabulary** (unique training texts) is kept for spelling correction.

2. **At runtime (each user message)**  
   - Input is cleaned and optionally corrected with RapidFuzz against the vocabulary.  
   - It is transformed with the same vectorizer and passed to the model.  
   - If `predict_proba` max is below **0.25**, the bot replies with *"Sorry, I didn't understand. Could you please rephrase?"*  
   - Otherwise, the predicted intent is mapped to a reply (from the `responses` dict in `bot.py`). Greeting replies are chosen at random.

---

## Training Data

`data.json` is a single JSON object: **intent name → list of example phrases**.

Example:

```json
{
  "greeting": ["hello", "hi", "namaste", "नमस्ते"],
  "checkin_time": ["check in time", "when can i check in"],
  "checkout_time": ["check out time", "when is checkout"],
  "wifi": ["do you have wifi", "internet available", "wifi छ?"],
  "parking": ["parking available", "car parking free"],
  "menu": ["show menu", "what food do you serve"],
  "booking": ["how to book room", "i want to reserve", "कोठा बुक गर्न मिल्छ?"],
  "goodbye": ["bye", "thank you", "thanks"]
}
```

- Add or edit intents and examples here.  
- Ensure every intent in `data.json` has a corresponding entry in the `responses` dictionary in `bot.py`; otherwise the bot may return `None`.

---

## API

### POST `/get/`

Returns the bot’s reply for a given message.

- **Request:** JSON body: `{ "message": "user text here" }`  
- **Headers:**  
  - `Content-Type: application/json`  
  - `X-CSRFToken: <cookie csrftoken>` (required; cookie is set when loading the home page)
- **Response:** JSON: `{ "response": "Bot reply text" }`

Example:

```bash
curl -X POST http://127.0.0.1:8000/get/ \
  -H "Content-Type: application/json" \
  -H "X-CSRFToken: YOUR_CSRF_TOKEN" \
  -d "{\"message\": \"hello\"}"
```

---

## Configuration

- **Bot data paths**  
  In `bot.py`, paths are derived from the script location:
  - `DATA_JSON` = directory of `bot.py` + `data.json`  
  - `CHATLOG_TXT` = directory of `bot.py` + `chatlog.txt`  
  So the bot works both when run as `python bot.py` and when imported by Django.

- **Django**
  - **Secret key:** Set `DJANGO_SECRET_KEY` in the environment for production; otherwise the default in `settings.py` is used.  
  - **Project root for `bot`:** `chatbot_project/settings.py` adds `PROJECT_ROOT` (parent of `chatbot_project`) to `sys.path`; `manage.py` also adds the same path so `from bot import get_response` works when running from `chatbot_project`.

- **Logs**  
  To avoid committing chat logs, add `chatlog.txt` to `.gitignore` (optional).

---

## Security Notes

- Do **not** deploy with `DEBUG = True` or a hardcoded `SECRET_KEY`.  
- Use `DJANGO_SECRET_KEY` and set `ALLOWED_HOSTS` for production.  
- The `/get/` endpoint is protected by Django’s CSRF middleware; the web UI sends the token from the cookie.

---

## License

This project is for educational and demonstration purposes. Use and modify as needed.
