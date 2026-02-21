import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from rapidfuzz import process
import random

# ---------------------------
# TEXT CLEANING FUNCTION
# ---------------------------
def clean_text(text):
    return text.lower().strip()


# ---------------------------
# LOAD TRAINING DATA
# ---------------------------
with open("data.json", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []

for intent, examples in data.items():
    for example in examples:
        texts.append(clean_text(example))
        labels.append(intent)

# ---------------------------
# CONVERT TEXT TO NUMBERS
# ---------------------------
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

# ---------------------------
# TRAIN MODEL
# ---------------------------
model = LogisticRegression()
model.fit(X, labels)

# ---------------------------
# VOCABULARY
# ---------------------------
vocabulary = list(set(texts))

# ---------------------------
# BOT RESPONSES
# ---------------------------
responses = {
    "greeting": ["🙏 Welcome to Hotel Sunrise!",
        "Hello! How can I assist you today?",
        "Namaste! How may I help you?"],
    "checkin_time": "Check-in starts at 2 PM.",
    "checkout_time": "Check-out time is 12 PM.",
    "wifi": "Yes, we provide free high-speed WiFi.",
    "parking": "Yes, free and secure parking is available.",
    "menu": "We serve Nepali, Indian & Continental dishes.",
    "booking": "You can book a room via reception or our website.",
    "goodbye": "Thank you for contacting us. Have a wonderful day!"
}

# ---------------------------
# CORRECT SPELLING
# ---------------------------
def correct_spelling(user_input, vocabulary):
    match, score, _ = process.extractOne(user_input, vocabulary)
    if score > 80:
        return match
    return user_input

# ---------------------------
# RESPONSE FUNCTION
# ---------------------------
def get_response(user_input):
    # clean input
    user_input = clean_text(user_input)

    # correct spelling / typos
    user_input = correct_spelling(user_input, vocabulary)

    # convert to numbers
    X_test = vectorizer.transform([user_input])

    # confidence score
    confidence = model.predict_proba(X_test).max()

    if confidence < 0.25:
        return "Sorry, I didn’t understand. Could you please rephrase?"

    intent = model.predict(X_test)[0]

    reply = responses.get(intent)
    if isinstance(reply, list):
        return random.choice(reply)
    return reply

# ---------------------------
# CHAT LOOP
# ---------------------------
print(" Hotel Support Bot Ready! (type 'exit' to stop)\n")

# ---------------------------
# LOG CHAT
# ---------------------------
def log_chat(user, bot):
    with open("chatlog.txt", "a", encoding="utf-8") as f:
        f.write(f"You: {user}\nBot: {bot}\n\n")

# ---------------------------
# CHAT LOOP
# ---------------------------
while True:
    user = input("You: ")
    if user.lower() == "help":
        print("Bot: You can ask about booking, wifi, parking, menu, or check-in time.")
        continue

    if user.lower() == "exit":
        print("Bot: Goodbye!")
        break

    response = get_response(user)
    print("Bot:", response)
    log_chat(user, response)

