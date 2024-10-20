import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle

nltk.download('punkt')
nltk.download('stopwords')

data = pd.DataFrame({
    'review': [
        "Awful experience. The food was terrible and the service was non-existent.",
        "I will never come back. The food was cold and the staff was rude.",
        "The worst meal I've ever had. Completely disappointing.",
        "Very disappointing. The food was bland and the service was slow.",
        "Terrible experience. The food was undercooked and the service was poor.",
        "Not a good experience. The restaurant was dirty and the food was bad.",
        "The food was okay, but the service could have been better.",
        "Average experience. The food was mediocre and the staff was indifferent.",
        "The restaurant was noisy and the food was only slightly better than fast food.",
        "It was a decent meal. Nothing extraordinary, but not bad.",
        "The food was okay, but the ambiance was nice. Service was decent.",
        "Good place for a casual meal, but it lacks in flavor.",
        "A solid restaurant. The food was good, and the service was satisfactory.",
        "Nice atmosphere and decent food, but nothing exceptional.",
        "Average dining experience. The food was good but not memorable.",
        "The meal was good. The service was pleasant and the atmosphere was nice.",
        "Above average experience. The food was flavorful and the service was friendly.",
        "Not bad. The restaurant was clean and the food was decent.",
        "Enjoyed the meal. The food was tasty and the service was good.",
        "Very good experience. The food was great, and the staff was attentive.",
        "Had a pleasant time. The food was flavorful and the service was prompt.",
        "Excellent meal. The food was fantastic and the service was top-notch.",
        "Great dining experience. The dishes were well-prepared and the staff was courteous.",
        "The food was wonderful. The atmosphere and service made it a memorable meal.",
        "One of the best meals I've had. Everything was perfect from start to finish.",
        "Outstanding experience. The food was exceptional and the service was impeccable.",
        "Fantastic restaurant. The food was delicious and the service was excellent.",
        "The ultimate dining experience. Everything was flawless, from the food to the service.",
        "Perfection in every aspect. The best restaurant experience I've ever had.",
        "Absolutely perfect. The food, service, and atmosphere were all outstanding."
    ],
    'rating': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10]
})

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

data['cleaned_review'] = data['review'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned_review'])
y = data['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Save the model and vectorizer to disk
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
    
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
