# --- 1. IMPORTS ---
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from googletrans import Translator
import speech_recognition as sr
import pyttsx3
import random
import re

# --- 2. KNOWLEDGE BASE & CONFIGURATION ---
# All university data is stored here for easy updates.
KNOWLEDGE_BASE = {
    "university_name": "National AI University",
    "programs": {
        "bs": {
            "list": ["BS Computer Science", "BS Software Engineering", "BS Information Technology"],
            "description": "Our BS programs are 4-year degrees. Admission requires at least 50% marks in Intermediate or an equivalent qualification. Classes are held in the morning sessions.",
            "merit_list_date": "October 15, 2025",
            "deadline_date": "September 30, 2025",
            "result_date": "October 5, 2025"
        },
        "ms": {
            "list": ["MS Computer Science", "MS Data Science", "MS Software Engineering"],
            "description": "Our MS programs are 2-year degrees, requiring a 16-year degree in a relevant field with at least a 2.5 CGPA. An admission test and an interview are mandatory.",
            "merit_list_date": "November 20, 2025",
            "deadline_date": "October 31, 2025",
            "result_date": "November 12, 2025"
        },
        "mphil": {
            "list": ["MPhil Computer Science", "MPhil Emerging Technologies"],
            "description": "Our MPhil programs are research-based, requiring 16 years of education with a 2.5+ CGPA. Students must clear the university entry test or a GAT test before admission.",
            "merit_list_date": "November 22, 2025",
            "deadline_date": "November 5, 2025",
            "result_date": "November 15, 2025"
        }
    },
    "admission_procedure": [
        "The admission process is simple:",
        "1. Fill out the online application form on our website.",
        "2. Upload your required documents (CNIC, transcripts, photos).",
        "3. Pay the admission processing fee.",
        "4. Appear for the mandatory entry test.",
        "5. Check the merit list and finalize your enrollment."
    ],
    "links": {
        "website": "www.university-admissions.edu.pk",
        "merit_list_section": "www.university-admissions.edu.pk/admissions/merit-list",
        "schedule": "www.university-admissions.edu.pk/students/schedule.pdf"
    },
    "fallbacks": [
        "I'm sorry, I didn't quite understand. I can help with program info, deadlines, and admission procedures.",
        "Could you please rephrase? I can provide details on BS, MS, or MPhil programs."
    ]
}

# --- 3. NLTK SETUP & INITIALIZATIONS ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except nltk.downloader.DownloadError:
    print("Downloading necessary NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("Downloads complete.")

lemmatizer = WordNetLemmatizer()

# --- 4. TEXT PRE-PROCESSING & TRANSLATION ---
def preprocess_text(text):
    """Tokenizes and lemmatizes text to its root form for better intent matching."""
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)

def translate_to_english(text):
    """Detects input language and translates to English if necessary."""
    translator = Translator()
    try:
        detected_lang = translator.detect(text).lang
        if detected_lang != 'en':
            print(f"(Translating from {detected_lang.upper()}...)")
            translated_text = translator.translate(text, dest='en').text
            return translated_text
        return text
    except Exception as e:
        print(f"Translation Error: {e}")
        return text # Fallback to original text on error

# --- 5. INTENT RECOGNITION ENGINE ---
training_data = {
    "greet": ["Hello", "Hi", "Hey"],
    "get_program_list": ["What programs do you offer?", "Show me a list of your programs", "What can I study?", "List all degrees"],
    "get_merit_list_info": ["When will the merit list be announced?", "merit list date", "When is the merit list coming?"],
    "get_deadline": ["What is the deadline to apply?", "What is the last date for admission?", "deadline"],
    "get_result_info": ["When will the entry test results be announced?", "entry test result", "When are results coming?"],
    "get_admission_procedure": ["What’s the admission process?", "How do I apply?", "admission procedure", "guide me on the admission procedure"],
    "get_bs_details": ["Tell me about BS programs", "Details for BS", "What is required for BS"],
    "get_ms_details": ["Tell me about MS programs", "Details for MS", "What is required for MS"],
    "get_mphil_details": ["Tell me about MPhil programs", "Details for MPhil", "What is required for MPhil"],
    "get_schedule": ["I’m already enrolled. Can I get my schedule?", "Where’s my class schedule?", "I am an enrolled student"],
    "goodbye": ["Bye", "Goodbye", "Exit", "Thanks bye"]
}

# Pre-process the training data itself for a more accurate model
corpus = []
intent_map = {}
for intent, phrases in training_data.items():
    for phrase in phrases:
        processed_phrase = preprocess_text(phrase)
        corpus.append(processed_phrase)
        intent_map[processed_phrase] = intent

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

def get_intent(user_input):
    """Identifies the user's intent after pre-processing the input."""
    processed_input = preprocess_text(user_input)
    user_vector = vectorizer.transform([processed_input])
    similarities = cosine_similarity(user_vector, X)
    max_similarity = similarities.max()
    
    if max_similarity < 0.35: # Threshold slightly increased for better accuracy
        return "unknown"
        
    closest_phrase_index = similarities.argmax()
    return intent_map[corpus[closest_phrase_index]]

# --- 6. ENTITY EXTRACTION ---
def extract_program_level(user_input):
    """Extracts a program level (bs, ms, mphil) from the user's input."""
    user_input = user_input.lower()
    if re.search(r'\bbs\b', user_input): return 'bs'
    if re.search(r'\bms\b', user_input): return 'ms'
    if re.search(r'\bmphil\b', user_input): return 'mphil'
    return None

# --- 7. RESPONSE GENERATION ---
def chatbot_response(user_input, conversation_state):
    """Generates a response based on intent and conversation state."""
    intent = get_intent(user_input)
    level = extract_program_level(user_input)
    
    if level: conversation_state["current_program_level"] = level
    current_level = conversation_state.get("current_program_level")

    if intent == "greet":
        response = f"Hello! 👋 Welcome to the Admission Office of {KNOWLEDGE_BASE['university_name']}. How can I help you today? Are you looking for information about BS, MS, or MPhil programs?"
        return response, {}

    elif intent == "goodbye":
        return "Goodbye! Best of luck with your application.", {}

    elif intent == "get_program_list":
        bs_list = '\n- BS Programs: ' + ", ".join(KNOWLEDGE_BASE['programs']['bs']['list'])
        ms_list = '\n- MS Programs: ' + ", ".join(KNOWLEDGE_BASE['programs']['ms']['list'])
        mphil_list = '\n- MPhil Programs: ' + ", ".join(KNOWLEDGE_BASE['programs']['mphil']['list'])
        response = f"We offer the following programs:{bs_list}{ms_list}{mphil_list}\n\nWould you like details on a specific program level?"
        return response, conversation_state

    elif intent in ["get_bs_details", "get_ms_details", "get_mphil_details"]:
        level_key = intent.split('_')[1] # e.g., 'get_bs_details' -> 'bs'
        return KNOWLEDGE_BASE['programs'][level_key]['description'], {"current_program_level": level_key}

    elif intent in ["get_merit_list_info", "get_deadline", "get_result_info"]:
        if current_level:
            info_type = intent.split('_')[1] + "_date" # e.g., 'get_deadline' -> 'deadline_date'
            date = KNOWLEDGE_BASE['programs'][current_level][info_type]
            response = f"The {info_type.replace('_', ' ')} for {current_level.upper()} programs is on {date}."
        else:
            response = f"For which program level (BS, MS, or MPhil) are you asking?"
        return response, conversation_state

    elif intent == "get_admission_procedure":
        procedure = "\n".join(KNOWLEDGE_BASE['admission_procedure'])
        response = f"{procedure}\nYou can start at our official website: {KNOWLEDGE_BASE['links']['website']}"
        return response, conversation_state
        
    elif intent == "get_schedule":
        response = f"Enrolled students can find their class schedules on the student portal or download it here: {KNOWLEDGE_BASE['links']['schedule']}"
        return response, conversation_state

    else: # Fallback for unknown intent
        return random.choice(KNOWLEDGE_BASE["fallbacks"]), conversation_state

# --- 8. MAIN CHAT LOOPS (TEXT & VOICE) ---
def run_text_chatbot():
    """Runs the chatbot using text input and output."""
    print(f"Chatbot: Hello! Welcome to {KNOWLEDGE_BASE['university_name']}. Type 'exit' to end.")
    conversation_state = {}
    while True:
        raw_input = input("You: ")
        if raw_input.lower() in ["exit", "bye", "goodbye"]:
            print("Chatbot: Goodbye! Have a great day.")
            break
        
        translated_input = translate_to_english(raw_input)
        response, updated_state = chatbot_response(translated_input, conversation_state)
        conversation_state = updated_state
        print(f"Chatbot: {response}")

def run_voice_chatbot():
    """Runs the chatbot using voice input and output."""
    recognizer = sr.Recognizer()
    tts_engine = pyttsx3.init()

    def speak(text):
        print(f"Chatbot: {text}")
        tts_engine.say(text)
        tts_engine.runAndWait()

    speak(f"Hello! Welcome to the Admission Office of {KNOWLEDGE_BASE['university_name']}. How can I help you today?")
    conversation_state = {}
    
    while True:
        with sr.Microphone() as source:
            print("\nListening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                speak("I didn't hear anything. Please try again.")
                continue

        try:
            print("Recognizing...")
            raw_input = recognizer.recognize_google(audio)
            print(f"You: {raw_input}")

            if raw_input.lower() in ["exit", "bye", "goodbye"]:
                speak("Goodbye! Best of luck.")
                break
            
            translated_input = translate_to_english(raw_input)
            response, updated_state = chatbot_response(translated_input, conversation_state)
            conversation_state = updated_state
            speak(response)

        except sr.RequestError:
            speak("Sorry, my speech service is down. Please try again later.")
        except sr.UnknownValueError:
            speak("Sorry, I didn't catch that. Could you please say it again?")
        except KeyboardInterrupt:
            print("\nExiting chatbot.")
            break

# --- 9. SCRIPT EXECUTION ---
if __name__ == "__main__":
    mode = input("Choose mode: 'text' or 'voice'? ").lower()
    if mode == 'voice':
        run_voice_chatbot()
    else:
        run_text_chatbot()