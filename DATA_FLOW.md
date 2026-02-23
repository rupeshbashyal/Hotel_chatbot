## Hotel Sunrise Chatbot – Data Flow

This document explains, step by step, how data moves through the **Hotel Sunrise** chatbot, from the moment a user types a message to the final response – for both the **web UI (Django)** and the **CLI**.

---

## 1. High-level overview

### 1.1 Main components

- **`data.json`**  
  Intent training data: `"intent" -> [example phrases...]`.

- **`bot.py`**  
  Core chatbot logic:
  - Loads and preprocesses training data
  - Trains TF‑IDF + Logistic Regression model
  - Handles text preprocessing, spelling correction, synonyms
  - Chooses responses

- **Django project (`chatbot_project/`)**
  - `chat/views.py` – HTTP views connecting the web UI to `bot.get_response`
  - `chat/urls.py` – Routes `/` and `/get/`
  - `templates/index.html` – Frontend chat UI (HTML/CSS/JS)

---

## 2. Training pipeline (runs at import time in `bot.py`)

This happens once when Python imports `bot.py` (CLI start or Django server start).

### 2.1 Load raw training data

1. `bot.py` locates `data.json` using a path relative to itself:
   - `BOT_DIR = Path(__file__).resolve().parent`
   - `DATA_JSON = BOT_DIR / "data.json"`
2. It opens `DATA_JSON` and loads it as a Python dict:
   - Keys: intent names (e.g. `"greeting"`, `"booking"`, `"wifi"`)
   - Values: lists of example phrases for each intent.

### 2.2 Preprocess examples

For every example phrase:

- The text is passed through `clean_text`:
  1. **Expand contractions** (`expand_contractions`)
     - `"don't"` → `"do not"`, `"what's"` → `"what is"`, `"can't"` → `"cannot"`, etc.
  2. **Lowercase**  
     - `"Hello"` → `"hello"`.
  3. **Remove punctuation** (`remove_punctuation`)
     - Replaces hyphens with spaces (`"check-in"` → `"check in"`).
     - Removes other punctuation: `"wifi??"` → `"wifi"`.
  4. **Normalize whitespace** (`normalize_whitespace`)
     - Collapses multiple spaces and strips ends: `"  room   available  "` → `"room available"`.

The result is a **cleaned training phrase**.

### 2.3 Build feature matrix

1. All cleaned phrases are stored in `texts`.
2. Their corresponding intents are stored in `labels`.
3. A TF‑IDF vectorizer is created:
   - `vectorizer = TfidfVectorizer(ngram_range=(1, 2))`
   - Uses 1‑gram and 2‑gram features (single words and word pairs).
4. `X = vectorizer.fit_transform(texts)` converts all training phrases into numeric vectors.

### 2.4 Train classifier

- A **Logistic Regression** model is created and trained:
  - `model = LogisticRegression()`
  - `model.fit(X, labels)`
- The model learns to map TF‑IDF vectors to intents (e.g. `"wifi"` → `wifi` intent).

### 2.5 Build vocabularies for fuzzy matching

From the cleaned training texts:

- **Phrase vocabulary** (`phrase_vocabulary`)
  - Unique full phrases (e.g. `"do you have wifi"`, `"check in time"`).

- **Word vocabulary** (`word_vocabulary`)
  - Unique words split from all phrases (e.g. `"wifi"`, `"check"`, `"parking"`).

These are used by RapidFuzz for spelling correction.

---

## 3. Request pipeline – Web UI (Django)

This is what happens when a user uses the web chat page.

### 3.1 User types a message (browser)

1. The user opens `http://127.0.0.1:8000/`.
2. Django view `home` renders `templates/index.html`, which:
   - Shows the styled chat card, a welcome message and quick‑reply buttons.
   - Loads a CSRF cookie (via `@ensure_csrf_cookie` in `home` view).
3. When the user types a message and clicks **Send** (or presses Enter):
   - JavaScript in `index.html` calls `sendMessage()`:
     - **Adds user bubble** to the chat UI.
     - Reads CSRF token from cookie `csrftoken`.
     - Sends a `fetch` POST request to `/get/`:

       ```js
       fetch("/get/", {
         method: "POST",
         headers: {
           "Content-Type": "application/json",
           "X-CSRFToken": csrfToken
         },
         body: JSON.stringify({ message: message })
       })
       ```

   - Shows a **typing indicator** while waiting.

### 3.2 Django view receives request

1. The request hits `chat/urls.py`:

   ```python
   path('get/', views.get_bot_response, name='get_response')
   ```

2. `views.get_bot_response` is called:
   - It parses the JSON body:

     ```python
     data = json.loads(request.body)
     message = data.get("message", "")
     ```

   - Calls the core bot function:

     ```python
     response = get_response(message)
     ```

   - Returns a JSON response:

     ```python
     return JsonResponse({"response": response})
     ```

### 3.3 JavaScript displays bot reply

1. The browser receives JSON: `{"response": "<bot text>"}`.
2. JavaScript:
   - Removes the typing indicator.
   - Adds a **bot bubble** to the chat UI with the reply.

---

## 4. Request pipeline – CLI (terminal)

When you run `python bot.py`:

1. The same training pipeline (Section 2) runs once.
2. A loop starts:

   ```python
   while True:
       user = input("You: ")
       ...
       response = get_response(user)
       print("Bot:", response)
       log_chat(user, response)
   ```

3. Special commands:
   - `"help"` – prints what topics the bot can answer.
   - `"exit"` – prints goodbye and breaks the loop.
4. Each exchange is appended to `chatlog.txt` (for later review).

The **core processing** uses the exact same `get_response` function as the web UI.

---

## 5. Core NLP pipeline in `get_response`

This is the heart of the chatbot – used by both CLI and Django.

### 5.1 Input cleaning

1. `clean_text(user_input)`:
   - Expands contractions (e.g. `"what's wifi password"` → `"what is wifi password"`).
   - Lowercases.
   - Removes punctuation and normalizes spaces.
2. If the result is empty:

   ```python
   if not user_input:
       return "Please type a message. I can help with booking, wifi, parking, menu, and more!"
   ```

### 5.2 Typo and synonym normalization

1. **Common typo fixes** (`fix_common_typos`)
   - Word‑by‑word replacements:
     - `"wify"` → `"wifi"`, `"parkng"` → `"parking"`, `"bookng"` → `"booking"`, etc.
     - Some romanized Nepali fixes, e.g. `"xa"` → `"cha"`.

2. **Synonym mapping** (`apply_synonyms`)
   - Maps related words to a canonical form:
     - `"reserve"`/`"reservation"` → `"book"`/`"booking"`
     - `"internet"`, `"wireless"`, `"net"` → `"wifi"`
     - `"restaurant"`, `"cuisine"` → `"menu"`
     - `"address"`, `"directions"` → `"location"`
     - `"phone"`, `"call"`, `"email"` → `"contact"`
     - `"hi"`, `"hey"`, `"namaste"` → `"hello"`

This step reduces variation so the classifier sees more consistent inputs.

### 5.3 Spelling & fuzzy matching

`correct_spelling(user_input, phrase_vocabulary, word_vocabulary)`:

1. **Phrase-level fuzzy match**
   - Uses RapidFuzz `process.extractOne` with `fuzz.token_set_ratio`:
   - Compares the whole user input against all training phrases.
   - If the best match has score ≥ `FUZZ_THRESHOLD` (75), that phrase is used as the corrected input.

   Example: `"wifi pasword"` → best phrase is `"wifi password"`.

2. **Word-level correction**
   - If phrase match is not strong enough:
     - Splits the text into words.
     - For each word (length ≥ 2), finds nearest word in `word_vocabulary` using `fuzz.ratio`.
     - If score ≥ 75, replaces the word with the matched vocabulary word.

   Example: `"parkng available"`  
   → `"parking available"`.

The output is a **clean, normalized, and corrected** input string.

### 5.4 Vectorization and classification

1. Convert user input to TF‑IDF features:

   ```python
   X_test = vectorizer.transform([user_input])
   ```

2. Compute prediction probabilities:

   ```python
   confidence = model.predict_proba(X_test).max()
   ```

3. If confidence is below threshold (0.25):

   ```python
   if confidence < 0.25:
       return "Sorry, I didn’t understand. Could you please rephrase?"
   ```

4. Otherwise, predict the intent:

   ```python
   intent = model.predict(X_test)[0]
   ```

### 5.5 Response selection

1. The predicted intent is used to look up replies in the `responses` dict.
2. Many intents have **multiple response variants** (lists):
   - For lists, a random reply is chosen:

     ```python
     reply = responses.get(intent)
     if isinstance(reply, list):
         return random.choice(reply)
     return reply
     ```

3. The final string is returned to:
   - The CLI loop (printed to terminal), or
   - The Django view (wrapped in JSON and sent to the browser).

---

## 6. End-to-end sequence diagrams (textual)

### 6.1 Web request

1. **User**: types `"wify pasword"` in browser and clicks **Send**.
2. **Browser (JS)**:
   - Adds *You* bubble.
   - Sends `POST /get/` with JSON `{ "message": "wify pasword" }` and CSRF token.
3. **Django `get_bot_response`**:
   - Parses JSON, extracts `"wify pasword"`.
   - Calls `get_response("wify pasword")`.
4. **`get_response` in `bot.py`**:
   - Cleans text → `"wify pasword"`.
   - Fixes typos: `"wify"` → `"wifi"`.
   - Applies synonyms (if any).
   - Fuzzy spelling correction → `"wifi password"`.
   - Vectorizes & classifies as `wifi` intent.
   - Randomly picks one `wifi` reply.
5. **Django**: returns JSON `{"response": "Free WiFi is available. The password is at reception."}`.
6. **Browser (JS)**:
   - Removes typing indicator.
   - Adds **Hotel Sunrise** bubble with that text.

### 6.2 CLI request

1. **User (terminal)**: types `"room xa"` while running `python bot.py`.
2. CLI loop:
   - Sends `"room xa"` to `get_response`.
3. `get_response`:
   - Cleans text → `"room xa"`.
   - Applies typo fix `"xa"` → `"cha"` → `"room cha"`.
   - Synonyms: `"room"` → `"booking"`.
   - Fuzzy matching / TF‑IDF classification → `booking` intent.
   - Picks a booking response.
4. CLI prints: `Bot: You can book via reception, phone, or our website. We'd love to host you!`
5. Conversation is logged to `chatlog.txt`.

---

## 7. Key design choices

- **Shared core (`get_response`)**  
  Both web and CLI use the same function, so improvements benefit all interfaces.

- **Simple but effective ML**  
  TF‑IDF + Logistic Regression is fast, interpretable, and easy to retrain.

- **Layered text processing**  
  Contractions → punctuation → whitespace → typos → synonyms → fuzzy match → classifier.

- **Graceful degradation**  
  Low-confidence answers fall back to a polite *“I didn’t understand”* with a rephrase request.

This is the complete data flow of the Hotel Sunrise chatbot. You can modify any stage (training data, preprocessing, model, responses, or UI) to evolve the assistant further.

