import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

with open("data.json") as f:
    data = json.load(f)

texts = []
labels = []

for intent, examples in data.items():
    for ex in examples:
        texts.append(ex)
        labels.append(intent)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

responses = {
    "greeting": "Welcome to Hotel Sunrise 😊",
    "checkin_time": "Check-in starts at 2 PM.",
    "wifi": "Yes, free WiFi is available.",
    "parking": "Yes, free parking available.",
    "menu": "We serve Nepali & Continental dishes.",
    "booking": "You can book via reception or website.",
    "goodbye": "Thank you! Have a nice day."
}

while True:
    user = input("You: ")
    if user.lower() == "exit":
        break

    X_test = vectorizer.transform([user])
    intent = model.predict(X_test)[0]

    print("Bot:", responses[intent])

    