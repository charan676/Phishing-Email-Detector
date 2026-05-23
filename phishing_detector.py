import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os
import argparse

class PhishingDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        
    def load_data(self, file_path):
        """Load the email dataset"""
        if not os.path.exists(file_path):
            print(f"Dataset file {file_path} not found.")
            print("Creating a sample dataset for demonstration...")
            self.create_sample_dataset(file_path)
        
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def create_sample_dataset(self, file_path):
        """Create a sample dataset for demonstration"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Sample phishing emails
        phishing_emails = [
            "URGENT: Your account has been suspended. Click here to verify your information immediately: http://secure-bank-login.com/verify",
            "Congratulations! You've won $1,000,000 in our lottery. Claim your prize now at http://winner-prize-claim.com",
            "Your PayPal account has been limited. Please update your payment details at http://paypal-security-update.com/login",
            "Suspicious activity detected on your account. Click here to secure your account: http://account-security-center.org",
            "Invoice #12345: Your payment of $500 is overdue. Please pay immediately at http://secure-payment-gateway.net",
            "Your Amazon order has been cancelled. Refund available at http://amazon-refund-portal.com",
            "Your Netflix subscription will expire today. Renew now at http://netflix-security-update.com",
            "Security Alert: Someone tried to access your account. Verify your identity at http://identity-verification-center.org",
            "You have a new message in your inbox. Click here to read: http://secure-message-center.com",
            "Your Microsoft account will be deleted. Save it now at http://microsoft-account-recovery.net"
        ]
        
        # Sample legitimate emails
        legitimate_emails = [
            "Your monthly bank statement is now available. Please log in to your online banking portal to view.",
            "Thank you for your recent purchase. Your order #12345 has been shipped and will arrive in 3-5 business days.",
            "Meeting reminder: Tomorrow at 2 PM in conference room B. Please bring your quarterly reports.",
            "Your prescription is ready for pickup at Main Street Pharmacy. We're open until 9 PM today.",
            "We've received your job application and will review it shortly. Thank you for your interest in the position.",
            "Your electricity bill for this month is $125. Payment is due by the end of the month.",
            "Happy birthday! As a valued customer, we're offering you a 20% discount on your next purchase.",
            "Your doctor's appointment has been scheduled for next Tuesday at 10:30 AM. Please arrive 15 minutes early.",
            "The neighborhood association meeting has been rescheduled to next Thursday at 7 PM.",
            "Your library book is due for renewal. Please visit our website or call to extend your borrowing period."
        ]
        
        # Create DataFrame
        phishing_df = pd.DataFrame({
            'text': phishing_emails,
            'label': ['phishing'] * len(phishing_emails)
        })
        
        legitimate_df = pd.DataFrame({
            'text': legitimate_emails,
            'label': ['safe'] * len(legitimate_emails)
        })
        
        # Combine and save
        df = pd.concat([phishing_df, legitimate_df], ignore_index=True)
        df.to_csv(file_path, index=False)
        print(f"Sample dataset created at {file_path}")
    
    def extract_features(self, text):
        """Extract features from email text"""
        features = {}
        
        # Count suspicious keywords
        suspicious_words = ['urgent', 'verify', 'account', 'suspended', 'click', 'link', 'password', 
                           'security', 'update', 'immediately', 'confirm', 'bank', 'paypal', 'amazon',
                           'prize', 'winner', 'congratulations', 'limited', 'expire', 'payment']
        
        text_lower = text.lower()
        features['suspicious_word_count'] = sum(1 for word in suspicious_words if word in text_lower)
        
        # Count URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[\$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        features['url_count'] = len(re.findall(url_pattern, text))
        
        # Check for suspicious URL patterns
        suspicious_domains = ['secure', 'verify', 'login', 'update', 'security', 'account', 'paypal', 
                             'amazon', 'netflix', 'microsoft', 'bank', 'refund', 'claim', 'prize']
        
        urls = re.findall(url_pattern, text)
        features['suspicious_url_count'] = sum(1 for url in urls 
                                             for domain in suspicious_domains 
                                             if domain in url.lower())
        
        # Count HTML tags
        html_tags = re.findall(r'<[^>]+>', text)
        features['html_tag_count'] = len(html_tags)
        
        # Count exclamation marks
        features['exclamation_count'] = text.count('!')
        
        # Count dollar signs (often used in scams)
        features['dollar_sign_count'] = text.count('$')
        
        # Text length
        features['text_length'] = len(text)
        
        return features
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Extract features
        feature_df = pd.DataFrame([self.extract_features(text) for text in df['text']])
        
        # Combine with text
        X_text = df['text']
        X_features = feature_df
        y = df['label']
        
        return X_text, X_features, y
    
    def train_model(self, X_text, X_features, y):
        """Train the phishing detection model"""
        # Split data
        X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
            X_text, X_features, y, test_size=0.2, random_state=42
        )
        
        # Create pipeline with TF-IDF and Random Forest
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=3000)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train model
        self.pipeline.fit(X_text_train, y_train)
        
        # Make predictions
        y_pred = self.pipeline.predict(X_text_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Print results
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Safe', 'Phishing'],
                   yticklabels=['Safe', 'Phishing'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Save the plot
        os.makedirs('output', exist_ok=True)
        plt.savefig('output/confusion_matrix.png')
        print("Confusion matrix saved to output/confusion_matrix.png")
        
        # Save the model
        joblib.dump(self.pipeline, 'phishing_model.pkl')
        print("Model saved to phishing_model.pkl")
        
        return accuracy
    
    def load_model(self, model_path='phishing_model.pkl'):
        """Load a trained model"""
        if os.path.exists(model_path):
            self.pipeline = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"No model found at {model_path}")
            return False
    
    def predict(self, email_text):
        """Predict if an email is phishing or safe"""
        if self.pipeline is None:
            print("No model loaded. Please train or load a model first.")
            return None
        
        prediction = self.pipeline.predict([email_text])[0]
        probability = self.pipeline.predict_proba([email_text])[0]
        
        result = {
            'prediction': prediction,
            'confidence': max(probability) * 100,
            'phishing_probability': probability[1] * 100 if 'phishing' in self.pipeline.classes_ else probability[0] * 100
        }
        
        return result

def main():
    parser = argparse.ArgumentParser(description='Phishing Email Detection Model')

    parser.add_argument(
        '--mode',
        choices=['train', 'predict'],
        default='train',
        help='Choose mode: train or predict'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='data/emails.csv',
        help='Path to dataset CSV file'
    )

    parser.add_argument(
        '--email',
        type=str,
        help='Email text for prediction'
    )

    args = parser.parse_args()

    detector = PhishingDetector()

    # TRAIN MODE
    if args.mode == 'train':
        print("Loading dataset...")

        df = detector.load_data(args.dataset)

        if df is not None:
            print("Preparing data...")

            X_text, X_features, y = detector.prepare_data(df)

            print("Training model...")

            accuracy = detector.train_model(X_text, X_features, y)

            print(f"\nTraining completed successfully!")
            print(f"Final Accuracy: {accuracy * 100:.2f}%")

    # PREDICT MODE
    elif args.mode == 'predict':

        if detector.load_model():

            if args.email:
                result = detector.predict(args.email)

                print("\n===== Prediction Result =====")
                print(f"Email Type : {result['prediction']}")
                print(f"Confidence : {result['confidence']:.2f}%")
                print(f"Phishing Probability : {result['phishing_probability']:.2f}%")

            else:
                print("Please provide email text using --email")


if __name__ == "__main__":
    main()