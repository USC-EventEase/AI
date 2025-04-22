from flask import Flask, request, jsonify
from flask_cors import CORS  # enable CORS for cross-origin requests
from pymongo import MongoClient
import math
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib

# Disable the progress bar for model download
os.environ["HUGGINGFACE_HUB_DISABLE_PROGRESS_BAR"] = "1"

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes; customize with resources if needed

print("⏳ Downloading Model for Embeddings")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model download completed")

SIMILARITY_THRESHOLD = 0.50

# Use environment variable for MongoDB URI or fallback to a default value
MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
print("✅ DB connection established")
db = client['recommendations']

MONGO_URI_TEST = os.environ.get("MONGO_URI_TEST")
trainer = MongoClient(MONGO_URI_TEST)
print("✅ DB connection established")
coll_trest = trainer["test"]['past_events']

##########################################
# Helper Functions
##########################################

def compute_embedding(event_data):
    # Encode event name and description to produce an embedding
    return bert_model.encode(
        event_data["event_name"] + " " + event_data["event_description"],
        convert_to_numpy=True
    ).tolist()


def prediction_model():
    # Get only past events with attendance data
    data = list(coll_trest.find({"date": {"$lt": datetime.now().strftime("%Y-%m-%d")},"attendance": {"$exists": True}}))

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Feature engineering
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Encode categorical variables
    label_encoders = {}
    for col in ['type','location', 'weather']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le    

    # Features and target
    features = ['type', 'capacity', 'ticket_price','location', 'weather', 'day_of_week', 'month', 'is_weekend']
    X = df[features]
    y = df['attendance']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    #print(f"Model MAE: {mae:.2f}")

    return model, label_encoders

def predict_crowd(event_details):
    """
    Predict crowd for a new event
    
    Parameters:
    event_details (dict): Dictionary containing event details with keys:
        - type: concert, conference, tourist_spot, etc.
        - date: YYYY-MM-DD
        - capacity: integer
        - ticket_price: float
        - location: string 
    """
    
    # Load model and encoders
    model,label_encoders = prediction_model()
    
    # Create DataFrame
    df = pd.DataFrame([event_details])
    
    # Feature engineering
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    #weather
    if df['month'][0] in [12,1]: #dec, jan
        df['weather'] = 'snowy'
    elif df['month'][0] == 2: #feb
        df['weather'] = 'rainy'
    elif df['month'][0] in [3,4,11]: #march apr nov
        df['weather'] = 'cloudy'
    else:
        df['weather'] = 'sunny'

    #location
    list_location = ["Los Angeles", "New York", "Seattle","San Francisco","Chicago"]
   
    for l in list_location:
        if l in df['location'][0]:
            df['location'] = l
            
    
    # Encode categorical variables
    for col in ['type', 'location', 'weather']:
        le = label_encoders[col]
        df[col] = le.transform(df[col])
    
    # Features
    features = ['type', 'capacity', 'ticket_price', 'location', 'weather', 'day_of_week', 'month', 'is_weekend']
    X = df[features]
    
    # Predict
    prediction = model.predict(X)
    
    return int(prediction[0])
##########################################
# Background Job Functions
##########################################

def update_similarity_for_new_event(event_id, new_embedding):
    # Find all other embeddings (using _id as the event identifier)
    other_embeddings = list(db.embeddings.find({ "_id": {"$ne": event_id} }))
    if not other_embeddings:
        db.similarities.update_one(
            {"_id": event_id},
            {"$set": {"list_of_recommendations": []}},
            upsert=True
        )
        print("No other embeddings available; skipping similarity update.")
        return
    existing_event_ids = [data["_id"] for data in other_embeddings]
    existing_embeddings = np.array([np.array(data["embedding"]) for data in other_embeddings])
    similarity = cosine_similarity([new_embedding], existing_embeddings)[0]

    # Compute sorted list of similar events with similarity above threshold
    similar_events = sorted(
        [
            {"event_id": existing_event_ids[i], "similarity": float(similarity[i])}
            for i in range(len(existing_event_ids))
            if similarity[i] >= SIMILARITY_THRESHOLD
        ],
        key=lambda x: x["similarity"],
        reverse=True
    )

    # Update the document for the new event with its recommendations
    db.similarities.update_one(
        {"_id": event_id},
        {"$push": {"list_of_recommendations": {"$each": similar_events}}},
        upsert=True
    )

    # Add reciprocal recommendations for each similar event
    for event in similar_events:
        db.similarities.update_one(
            {"_id": event['event_id']},
            {"$push": {"list_of_recommendations": {"event_id": event_id, "similarity": event['similarity']}}},
            upsert=True
        )

    print(f"Recommendation matching completed for event: {event_id}")

def cleanup_recommendations_for_deleted_event(event_id):
    """
    Remove the recommendation document for this event and
    remove any reference to event_id from other events' recommendation lists.
    """
    # Remove the event's own similarity document
    db.similarities.delete_one({"_id": event_id})
    # Remove references of event_id from every other event
    db.similarities.update_many(
        {},
        {"$pull": {"list_of_recommendations": {"event_id": event_id}}}
    )
    print(f"Cleaned up recommendations for deleted event: {event_id}")

##########################################
# APScheduler Setup
##########################################

scheduler = BackgroundScheduler()
scheduler.start()

##########################################
# API Endpoints
##########################################

@app.route('/api/add_recommendations', methods=['POST'])
def add_recommendation():
    """
    Expected JSON payload:
    {
       "eventId": "some_unique_identifier",
       "eventData": {
           "event_name": "Event Name",
           "event_description": "Event Description",
           ... // other event details if needed
       }
    }
    """
    try:
        data = request.json
        event_id = data.get('eventId')
        event_data = data.get('eventData')
        if not event_id or not event_data:
            return jsonify({"error": "Missing eventId or eventData"}), 400

        # Compute and store the embedding (using the event_id as the _id)
        embedding = compute_embedding(event_data)
        db.embeddings.insert_one({"_id": event_id, "embedding": embedding})
        print(f"Saved embedding for event: {event_id}")

        # Schedule background job to run recommendation matching
        scheduler.add_job(
            func=update_similarity_for_new_event,
            args=[event_id, embedding],
            trigger="date",
            run_date=datetime.now() + timedelta(seconds=2)
        )

        return jsonify({
            "message": "Embedding created. Recommendation matching scheduled.",
            "eventId": event_id,
            "embedding": embedding
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/delete_recommendations', methods=['POST'])
def delete_recommendation():
    """
    Expected JSON payload:
    {
      "eventId": "the_event_id"
    }
    """
    try:
        data = request.json
        event_id = data.get('eventId')
        if not event_id:
            return jsonify({"error": "Missing eventId"}), 400

        # Delete the event's embedding
        db.embeddings.delete_one({"_id": event_id})
        print(f"Deleted embedding for event: {event_id}")

        scheduler.add_job(
            func=cleanup_recommendations_for_deleted_event,
            args=[event_id],
            trigger="date",
            run_date=datetime.now() + timedelta(seconds=2)
        )

        return jsonify({
            "message": "Event deleted. Cleanup of recommendations scheduled.",
            "eventId": event_id
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_recommendations', methods=['GET'])
def get_recommendations():
    event_id = request.args.get('eventId')
    if not event_id:
        return jsonify({"error": "Missing eventId"}), 400

    recommendation_doc = db.similarities.find_one({"_id": event_id})
    if not recommendation_doc or not recommendation_doc.get('list_of_recommendations'):
        return jsonify({"error": "No recommendations found for this event"}), 404

    return jsonify(recommendation_doc['list_of_recommendations']), 200


@app.route('/api/get_crowd_predictions', methods=['POST'])
def get_crowd():
    try:
        data = request.json
        new_event = data.get('eventDetails')
        predicted_attendance = predict_crowd(new_event)
        return jsonify({
                    "message": "Prediction Successfull",
                    "eventId": predicted_attendance
                }), 200 
    except Exception as e:
        return jsonify({"error": str(e)}), 500




##########################################
# Run the Flask App
##########################################

if __name__ == '__main__':
    # Bind to 0.0.0.0 to allow external access (required for Docker)
    app.run(host='0.0.0.0', port=3002, debug=False, threaded=True)
