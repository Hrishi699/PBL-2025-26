
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from urllib.parse import urlparse
import re
import pickle
import os

app = Flask(__name__)
CORS(app)

# Feature extraction function
def extract_features(url):
    """Extract features from URL for phishing detection"""
    features = []
    
    try:
        parsed = urlparse(url if url.startswith('http') else 'http://' + url)
        
        # 1. URL Length
        features.append(len(url))
        
        # 2. Number of dots
        features.append(url.count('.'))
        
        # 3. Number of hyphens
        features.append(url.count('-'))
        
        # 4. Number of underscores
        features.append(url.count('_'))
        
        # 5. Number of slashes
        features.append(url.count('/'))
        
        # 6. Number of question marks
        features.append(url.count('?'))
        
        # 7. Number of equal signs
        features.append(url.count('='))
        
        # 8. Number of @ symbols
        features.append(url.count('@'))
        
        # 9. Number of ampersands
        features.append(url.count('&'))
        
        # 10. Number of exclamation marks
        features.append(url.count('!'))
        
        # 11. Number of spaces
        features.append(url.count(' '))
        
        # 12. Number of tildes
        features.append(url.count('~'))
        
        # 13. Number of commas
        features.append(url.count(','))
        
        # 14. Number of plus signs
        features.append(url.count('+'))
        
        # 15. Number of asterisks
        features.append(url.count('*'))
        
        # 16. Number of hash symbols
        features.append(url.count('#'))
        
        # 17. Number of dollar signs
        features.append(url.count('$'))
        
        # 18. Number of percent signs
        features.append(url.count('%'))
        
        # 19. Has IP address
        features.append(1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0)
        
        # 20. Has HTTPS
        features.append(1 if parsed.scheme == 'https' else 0)
        
        # 21. Domain length
        features.append(len(parsed.netloc))
        
        # 22. Number of subdomains
        features.append(parsed.netloc.count('.'))
        
        # 23. Has port
        features.append(1 if parsed.port else 0)
        
        return np.array(features).reshape(1, -1)
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Initialize and train model
def train_model():
    """Train Random Forest model with sample data"""
    # Sample training data (features extracted from URLs)
    # In production, use a real phishing dataset
    X_train = np.array([
        # Legitimate URLs (features)
        [30, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 15, 1, 0],
        [25, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 10, 0, 0],
        [35, 2, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 18, 2, 0],
        [28, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 12, 1, 0],
        # Phishing URLs (features)
        [120, 8, 5, 3, 8, 2, 3, 1, 2, 0, 0, 1, 0, 1, 0, 0, 1, 2, 1, 0, 45, 6, 1],
        [95, 6, 4, 2, 6, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 38, 5, 0],
        [110, 7, 6, 4, 7, 3, 4, 2, 3, 1, 0, 1, 1, 0, 0, 1, 0, 2, 1, 0, 42, 7, 1],
        [85, 5, 3, 1, 5, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 35, 4, 0],
    ])
    
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 0 = legitimate, 1 = phishing
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        min_samples_split=2
    )
    model.fit(X_train, y_train)
    
    return model

# Always train a fresh model (simpler approach)
print("Training Random Forest model...")
model = train_model()
print("Model ready!")
    
    # Train new model
    model = train_model()
    with open('phishing_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("New model trained and saved!")
    return model

model = load_or_train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if URL is phishing or legitimate"""
    try:
        data = request.get_json()
        url = data.get('url', '')
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Extract features
        features = extract_features(url)
        
        if features is None:
            return jsonify({'error': 'Invalid URL format'}), 400
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        result = {
            'url': url,
            'is_phishing': bool(prediction),
            'confidence': float(probability[1] * 100 if prediction else probability[0] * 100),
            'phishing_probability': float(probability[1] * 100),
            'legitimate_probability': float(probability[0] * 100)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain model with new data"""
    global model
    try:
        model = train_model()
        with open('phishing_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        return jsonify({'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
