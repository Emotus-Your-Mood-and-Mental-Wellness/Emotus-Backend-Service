import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Configuration
MAX_LEN = 100
NUM_WORDS = 10000


class TextRequest(BaseModel):
    text: str


class MoodRecommendation(BaseModel):
    sympathy_message: str
    predicted_mood: str
    stress_level: str
    thoughtful_suggestions: list
    things_to_do: list


class MoodPredictionService:
    def __init__(self,
                 model_path: str,
                 recommendations_path: str,
                 dataset_path: str):
        # Load model and tokenizer
        self.model = load_model(model_path)

        # Load dataset and prepare tokenizer
        df = pd.read_csv(dataset_path)
        self.tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(df['refined_tweet'])

        # Prepare label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit_transform(df['label'])

        # Load recommendations
        with open(recommendations_path, 'r') as file:
            self.recommendations_data = json.load(file)

        # Mood to stress mapping
        self.mood_to_stress_mapping = {
            'anger': 'high',
            'fear': 'high',
            'happy': 'low',
            'love': 'low',
            'sadness': 'medium'
        }

    def get_intensity_from_text(self, text: str) -> str:
        high_intensity_keywords = ["keterlaluan", "parah", "benci", "kesal", "muak"]
        medium_intensity_keywords = ["sedikit", "lumayan", "agak", "cuma", "mungkin", "kurang"]
        low_intensity_keywords = ["tenang", "sedang", "biasa", "ringan", "normal", "senang", "cinta", "suka", "sayang"]
        negations = ["tidak", "gak", "nggak", "bukan"]

        if any(word in text.lower() for word in high_intensity_keywords):
            return 'high'
        elif any(word in text.lower() for word in medium_intensity_keywords):
            return 'medium'
        elif any(word in text.lower() for word in low_intensity_keywords):
            return 'low'

        if any(word in text.lower() for word in negations):
            return 'low'

        return 'low'  # default

    def predict_mood(self, text: str) -> str:
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        prediction = self.model.predict(padded)
        mood = self.label_encoder.inverse_transform([np.argmax(prediction)])[0]
        return mood

    def recommend_activity(self, mood: str, intensity: str) -> Dict[str, Any]:
        stress_level = self.mood_to_stress_mapping.get(mood, 'low')

        if intensity == 'high':
            stress_level = 'high'
        elif intensity == 'medium' and stress_level == 'low':
            stress_level = 'medium'

        current_hour = datetime.now().hour
        if current_hour < 18:
            time_of_day = 'morning'
        elif 18 <= current_hour < 22:
            time_of_day = 'afternoon'
        elif 22 <= current_hour < 4:
            time_of_day = 'evening'
        else:
            time_of_day = 'night'

        time_based_recommendations = self.recommendations_data.get(stress_level, {}).get(time_of_day, [])

        thoughtful_suggestions = []
        tips = []
        sympathy_message = ""

        if time_based_recommendations:
            thoughtful_suggestions_data = time_based_recommendations[0].get('thoughtful_suggestions', [])
            tips_data = time_based_recommendations[0].get('tips_untukmu_saat_ini', [])
            sympathy_messages = time_based_recommendations[0].get('sympathy', [])

            if isinstance(thoughtful_suggestions_data, list):
                thoughtful_suggestions = random.sample(
                    thoughtful_suggestions_data,
                    k=min(3, len(thoughtful_suggestions_data))
                )

            if isinstance(tips_data, list):
                tips = random.sample(
                    tips_data,
                    k=min(3, len(tips_data))
                )

            if isinstance(sympathy_messages, list):
                sympathy_message = random.choice(sympathy_messages)

        return {
            "sympathy_message": sympathy_message,
            "predicted_mood": mood,
            "stress_level": stress_level,
            "thoughtful_suggestions": thoughtful_suggestions,
            "things_to_do": tips
        }

    def get_recommendation_from_sentence(self, text: str) -> Dict[str, Any]:
        predicted_mood = self.predict_mood(text)
        intensity = self.get_intensity_from_text(text)
        return self.recommend_activity(predicted_mood, intensity)


# FastAPI App
app = FastAPI(title="Mood Prediction and Recommendation API")

# Initialize service (replace paths with your actual paths)
mood_service = MoodPredictionService(
    model_path='my_model.h5',
    recommendations_path='RecommendationTimeBased.json',
    dataset_path='preprocessed_dataset_adjustdikit.csv'
)


@app.post("/predict_mood", response_model=MoodRecommendation)
async def predict_mood_endpoint(request: TextRequest):
    try:
        recommendation = mood_service.get_recommendation_from_sentence(request.text)
        return recommendation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)