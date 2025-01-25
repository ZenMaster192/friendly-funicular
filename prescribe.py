import numpy as np
import wave
from pydub import AudioSegment
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def speechrecognition_speech_to_text():
    recogniser = sr.Recognizer()
    #if an audio is provided
    if audio_file:
        try:
            with sr.AudioFile(audio_file) as source:
                print("Processing audio file...")
                audio = recogniser.record(source)
                text = recogniser.recognize_google(audio)
                print(f"Speech-to-Text: {text}")
                return text
        except sr.UnknownValueError as e:
            print(f"Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
    else:
        #Try the microphone
        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = recogniser.listen(source)
                print("Processing audio...")
                text = recogniser.recognize_google(audio)
                print(f"Speech-to-Text: {text}")
        except sr.UnknownValueError as e:
            print(f"Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
    

def dummy_dataset():
    #Sample texts for prescriptions data
    texts = ["Take 500mg of Paracetamol once daily for 7 days.",
             "Take 200mg of Ibuprofen twice daily for 5 days.",
             "Take 250mg of Amoxicillin thrice daily for 3 days.",
             "Apply ointment in the injured area for 6 days.",
             "Use two drops of eye drops in each eye twice daily.",
             "Take 300mg of Azithromycin once daily for 3 days."]
    #Labels for the texts
    labels = ['Paracetamol', 'Ibuprofen', 'Amoxicillin', 'Ointment', 'Eye drops', 'Azithromycin']
    return texts, labels

def train_model (texts,labels):
    #Converting the texts into features
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    #Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    #Training the model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    #Predicting the labels for the test set
    y_pred = model.predict(X_test)
    #Calculating the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f} %")
    return model, vectorizer

def classify_text(text,model,vectorizer):
    #Converting the text into features
    X = vectorizer.transform([text])
    #Predicting the label for the text
    label = model.predict(X)[0]
    return label

if __name__ == "__main__":
    audio_file = "prescription.wav"
    #Simulating speech-to-text conversion
    input_text = speechrecognition_speech_to_text()
    #Creating a dummy dataset
    texts, labels = dummy_dataset()
    #Training the model
    model, vectorizer = train_model(texts, labels)
    #Classifying the text
    text = "Take 500mg of Paracetamol once daily for 7 days."
    label = classify_text(input_text, model, vectorizer)
    print(f"Prescription: {text}")
    print(f"Predicted Label: {label}")

    #Saving the model and vectorizer
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
    with open('vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)