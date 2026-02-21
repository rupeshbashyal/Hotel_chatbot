import json
import random
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from rapidfuzz import process, fuzz

# Paths relative to this file so bot works from CLI or when imported by Django
BOT_DIR = Path(__file__).resolve().parent
DATA_JSON = BOT_DIR / "data.json"
CHATLOG_TXT = BOT_DIR / "chatlog.txt"

# ---------------------------
# CONTRACTIONS (expand before processing)
# ---------------------------
CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
    "can't": "cannot", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "haven't": "have not",
    "hasn't": "has not", "hadn't": "had not", "we're": "we are",
    "you're": "you are", "they're": "they are", "i'm": "i am",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "what's": "what is", "that's": "that is", "there's": "there is",
    "i've": "i have", "we've": "we have", "you've": "you have",
    "they've": "they have", "i'd": "i would", "we'd": "we would",
    "you'd": "you would", "they'd": "they would", "i'll": "i will",
    "we'll": "we will", "you'll": "you will", "they'll": "they will",
    "let's": "let us", "who's": "who is", "it'll": "it will",
}

# ---------------------------
# TEXT PREPROCESSING
# ---------------------------
def expand_contractions(text):
    """Expand contractions: don't -> do not"""
    text_lower = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text_lower = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, text_lower, flags=re.IGNORECASE)
    return text_lower


def remove_punctuation(text):
    """Remove punctuation, keep letters, numbers, spaces. Replace hyphens with space."""
    text = re.sub(r"-", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return text


def normalize_whitespace(text):
    """Collapse multiple spaces and strip."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text):
    """Full preprocessing pipeline for training and inference."""
    if not text or not isinstance(text, str):
        return ""
    text = expand_contractions(text)
    text = text.lower()
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------
# LOAD TRAINING DATA
# ---------------------------
with open(DATA_JSON, encoding="utf-8") as f:
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
# VOCABULARY (phrases + words for better fuzzy matching)
# ---------------------------
phrase_vocabulary = list(set(texts))
word_vocabulary = set()
for phrase in texts:
    word_vocabulary.update(phrase.split())
word_vocabulary = list(word_vocabulary)
# Combined: phrases first (prefer phrase match), then words
vocabulary = phrase_vocabulary + word_vocabulary

# ---------------------------
# SYNONYMS (map to canonical form for better intent matching)
# ---------------------------
SYNONYMS = {
    "reserve": "book", "reservation": "booking", "reserving": "booking",
    "internet": "wifi", "wireless": "wifi", "network": "wifi", "net": "wifi",
    "car": "parking", "vehicle": "parking", "park": "parking",
    "food": "menu", "restaurant": "menu", "eat": "menu", "cuisine": "menu",
    "room": "booking", "rooms": "booking", "accommodation": "booking",
    "checkin": "check in", "check-in": "check in",
    "checkout": "check out", "check-out": "check out",
    "arrival": "check in", "arrive": "check in",
    "departure": "check out", "leave": "check out", "vacate": "check out",
    "address": "location", "where": "location", "directions": "location",
    "phone": "contact", "number": "contact", "call": "contact", "email": "contact",
    "thx": "thanks", "thankyou": "thanks", "ty": "thanks",
    "bye": "goodbye", "goodbye": "goodbye", "see ya": "goodbye",
    "hi": "hello", "hey": "hello", "helo": "hello", "hii": "hello",
    "namaste": "hello", "namaskar": "hello",
}

# ---------------------------
# BOT RESPONSES
# ---------------------------
responses = {
    "greeting": [
        "🙏 Welcome to Hotel Sunrise! How can I assist you today?",
        "Hello! Namaste! How may I help you?",
        "Hi there! Welcome. What would you like to know?",
        "Good day! How can I make your stay better?",
        "Namaste! Welcome to Hotel Sunrise. How may I help you?",
        "Hello! Thanks for reaching out. Ask me about booking, wifi, parking, menu, or timings!",
        "Hi! Welcome. I can help with check-in, checkout, food, parking & more."
    ],
    "checkin_time": [
        "Check-in starts at 2 PM. You can collect your keys from reception.",
        "You may check in from 2:00 PM onwards.",
        "Check-in time is 2 PM. Early arrival? We'll store your luggage.",
        "Our check-in begins at 2 PM. See you soon!",
        "Guests can check in from 2 PM. Reception is open 24/7.",
        "Check-in from 2 PM. Need early check-in? Contact reception."
    ],
    "checkout_time": [
        "Check-out time is 12 PM. Please vacate your room by noon.",
        "You can check out until 12:00 PM.",
        "Check-out is at 12 PM. Need a late checkout? Ask at reception.",
        "Please complete check-out by 12 PM. Thank you!",
        "Check-out by 12 PM. Late checkout may incur extra charges.",
        "Room must be vacated by 12 PM. Luggage storage available if needed."
    ],
    "wifi": [
        "Yes! We provide free high-speed WiFi throughout the hotel.",
        "Free WiFi is available. The password is at reception.",
        "We have complimentary WiFi in all rooms and common areas.",
        "Yes, free internet! Ask reception for the WiFi password.",
        "Complimentary WiFi in all areas. Password available at front desk.",
        "Free high-speed WiFi for all guests. Connect and ask reception for the code."
    ],
    "parking": [
        "Yes, free and secure parking is available for guests.",
        "We offer complimentary parking. It's safe and monitored.",
        "Free parking is available on-site. Plenty of space!",
        "Yes! Secure, free parking for all our guests.",
        "Complimentary parking for guests. CCTV monitored for safety.",
        "Free car and bike parking. Ask reception for the parking area."
    ],
    "menu": [
        "We serve Nepali, Indian & Continental dishes. Room service available.",
        "Our restaurant offers Nepali, Indian, and Continental cuisine.",
        "We have a full menu: Nepali, Indian & Continental. Breakfast 7–10 AM.",
        "Nepali, Indian & Continental food. Ask for the menu at reception.",
        "Restaurant open 7 AM–10 PM. Daal bhat, momo, curries & more!",
        "Breakfast 7–10 AM, lunch & dinner till 10 PM. Room service 24/7.",
        "Veg & non-veg options. Nepali thali, Indian curries, Continental dishes."
    ],
    "booking": [
        "You can book via reception, phone, or our website. We'd love to host you!",
        "Book a room at reception or online. Need help? Call us.",
        "Reserve through our website or visit reception. Rooms subject to availability.",
        "Book online or at reception. We'll be happy to assist you!",
        "Call reception or book online. Single & double rooms available.",
        "Reserve by phone, website, or in person. We'll confirm availability."
    ],
    "goodbye": [
        "Thank you for contacting us. Have a wonderful day! 🙏",
        "Thanks for reaching out. Enjoy your stay at Hotel Sunrise!",
        "Dhanyabad! Have a great day. See you soon!",
        "Thank you! Take care. We hope to see you again.",
        "Bye! Have a safe journey. Namaste! 🙏",
        "Thanks! Have a lovely day. We're here if you need anything.",
        "Dhanyabad! Safe travels. Come back soon! 🙏"
    ],
    "location": [
        "Hotel Sunrise is in the heart of the city. Full address at reception.",
        "We're centrally located. Ask reception for exact address and map.",
        "Easy to find! Contact reception for directions and landmarks.",
        "Central location with good transport links. Reception can share the address.",
        "We're near the main area. Call us for precise directions."
    ],
    "contact": [
        "Reception: 24/7. Call or visit for phone number and email.",
        "Contact details are on our website. Reception is always available.",
        "Call our reception anytime. Number available on our website and brochures.",
        "Reception handles all enquiries. Visit or call for contact info.",
        "We're here 24/7. Ask at reception for phone and email."
    ],
    "help": [
        "I can help with: booking, check-in/check-out times, WiFi, parking, menu, location & contact. What do you need?",
        "Ask me about: room booking, wifi, parking, food, check-in/out times, or hotel location!",
        "You can ask about: rooms, wifi, parking, restaurant menu, timings, address, or contact. What would you like to know?",
        "I assist with booking, wifi, parking, food, check-in/out, location & contact. How can I help?"
    ]
}

# ---------------------------
# COMMON TYPOS (explicit fixes before fuzzy match)
# ---------------------------
COMMON_TYPOS = {
    "wify": "wifi", "wfi": "wifi", "wifii": "wifi", "wi fi": "wifi",
    "internt": "internet", "interent": "internet",
    "parkng": "parking", "parkin": "parking",
    "bookng": "booking", "bookig": "booking", "reserv": "reserve",
    "chek": "check", "chek in": "check in", "chek out": "check out",
    "menue": "menu", "menuu": "menu",
    "rom": "room", "rooom": "room", "rrom": "room",
    "thnks": "thanks", "thaks": "thanks", "tanks": "thanks",
    "gud": "good", "gudbye": "goodbye", "gudby": "goodbye",
    "ca": "cha", "xa": "cha",  # romanized Nepali (wifi cha?, room xa?)
    "milxa": "milcha",  # "is available" in Nepali
}


def fix_common_typos(text):
    """Apply explicit typo fixes before fuzzy matching."""
    words = text.split()
    return " ".join(COMMON_TYPOS.get(w, w) for w in words)


# ---------------------------
# APPLY SYNONYMS
# ---------------------------
def apply_synonyms(text):
    """Replace synonym words with canonical form for better intent matching."""
    words = text.split()
    return " ".join(SYNONYMS.get(w, w) for w in words)


# ---------------------------
# SPELLING / FUZZY MATCHING (tuned threshold, word-level correction)
# ---------------------------
FUZZ_THRESHOLD = 75  # Lower = more lenient (catches more typos). 75-85 works well.

def correct_spelling(user_input, phrase_vocab, word_vocab):
    """Correct typos using phrase match first, then word-level correction."""
    # 1. Try full phrase match (best for complete sentences)
    if len(user_input) >= 3:
        match, score, _ = process.extractOne(
            user_input, phrase_vocab, scorer=fuzz.token_set_ratio
        )
        if score >= FUZZ_THRESHOLD:
            return match

    # 2. Word-level correction (for typos in individual words)
    words = user_input.split()
    corrected = []
    for w in words:
        if len(w) < 2:
            corrected.append(w)
            continue
        match, score, _ = process.extractOne(w, word_vocab, scorer=fuzz.ratio)
        if score >= FUZZ_THRESHOLD:
            corrected.append(match)
        else:
            corrected.append(w)
    return " ".join(corrected)

# ---------------------------
# RESPONSE FUNCTION
# ---------------------------
def get_response(user_input):
    # preprocess input
    user_input = clean_text(user_input)

    if not user_input:
        return "Please type a message. I can help with booking, wifi, parking, menu, and more!"

    # fix common typos (wify->wifi, parkng->parking, etc.)
    user_input = fix_common_typos(user_input)

    # apply synonyms (reserve->book, internet->wifi, etc.)
    user_input = apply_synonyms(user_input)

    # fuzzy spelling correction (phrase + word level, threshold 75)
    user_input = correct_spelling(user_input, phrase_vocabulary, word_vocabulary)

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
# LOG CHAT
# ---------------------------
def log_chat(user, bot):
    with open(CHATLOG_TXT, "a", encoding="utf-8") as f:
        f.write(f"You: {user}\nBot: {bot}\n\n")

# ---------------------------
# CHAT LOOP (CLI only)
# ---------------------------
if __name__ == "__main__":
    print(" Hotel Support Bot Ready! (type 'exit' to stop)\n")
    while True:
        user = input("You: ")
        if user.lower() == "help":
            print("Bot: You can ask about booking, wifi, parking, menu, check-in/check-out time, location, or contact.")
            continue

        if user.lower() == "exit":
            print("Bot: Goodbye!")
            break

        response = get_response(user)
        print("Bot:", response)
        log_chat(user, response)

