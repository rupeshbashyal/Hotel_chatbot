import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
# BOT RESPONSES
# ---------------------------
responses = {
    "greeting": "Welcome to Hotel Sunrise! How can I help you?",
    "checkin_time": "Check-in starts at 2 PM.",
    "checkout_time": "Check-out time is 12 PM.",
    "wifi": "Yes, we provide free high-speed WiFi.",
    "parking": "Yes, free and secure parking is available.",
    "menu": "We serve Nepali, Indian & Continental dishes.",
    "booking": "You can book a room via reception or our website.",
    "goodbye": "Thank you for contacting us. Have a wonderful day!"
}

# ---------------------------
# RESPONSE FUNCTION
# ---------------------------
def get_response(user_input):
    user_input = clean_text(user_input)
    X_test = vectorizer.transform([user_input])

    # confidence score
    confidence = model.predict_proba(X_test).max()

    if confidence < 0.40:
        return "Sorry, I didn’t understand. Could you please rephrase?"

    intent = model.predict(X_test)[0]
    return responses.get(intent, "I am not sure how to respond to that.")


# ---------------------------
# CHAT LOOP
# ---------------------------
print(" Hotel Support Bot Ready! (type 'exit' to stop)\n")

while True:
    user = input("You: ")

    if user.lower() == "exit":
        print("Bot: Goodbye!")
        break

    response = get_response(user)
    print("Bot:", response)

