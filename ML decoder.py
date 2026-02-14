"""
Machine Learning Module for Neural Signal Decoding
Implements classification models for motor imagery, P300, and other BCI paradigms
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.python import layers
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from EEG signals"""
    
    @staticmethod
    def extract_band_power(data: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Extract power in different frequency bands
        
        Args:
            data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Feature vector with band powers
        """
        from scipy import signal
        
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        features = []
        
        for channel_data in data:
            # Compute power spectral density
            freqs, psd = signal.welch(channel_data, fs=sampling_rate, nperseg=256)
            
            # Extract power in each band
            for band_name, (low, high) in bands.items():
                idx_band = np.logical_and(freqs >= low, freqs <= high)
                band_power = np.trapz(psd[idx_band], freqs[idx_band])
                features.append(band_power)
        
        return np.array(features)
    
    @staticmethod
    def extract_statistical_features(data: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from signals
        
        Args:
            data: EEG data (channels x samples)
            
        Returns:
            Statistical features
        """
        features = []
        
        for channel_data in data:
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.var(channel_data),
                np.max(channel_data),
                np.min(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75)
            ])
        
        return np.array(features)
    
    @staticmethod
    def extract_time_domain_features(data: np.ndarray) -> np.ndarray:
        """
        Extract time-domain features
        
        Args:
            data: EEG data (channels x samples)
            
        Returns:
            Time-domain features
        """
        features = []
        
        for channel_data in data:
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
            
            # RMS
            rms = np.sqrt(np.mean(channel_data**2))
            
            # Peak-to-peak amplitude
            ptp = np.ptp(channel_data)
            
            features.extend([zero_crossings, rms, ptp])
        
        return np.array(features)


class NeuralDecoder:
    """Machine learning model for decoding neural signals"""
    
    def __init__(self, model_type: str = 'lda'):
        """
        Initialize decoder
        
        Args:
            model_type: Type of model ('lda', 'svm', 'rf', 'cnn')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        if model_type == 'lda':
            self.model = LinearDiscriminantAnalysis()
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10)
        elif model_type == 'cnn':
            self.model = None  # Will be built dynamically
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Initialized {model_type} decoder")
    
    def build_cnn_model(self, input_shape: tuple, num_classes: int):
        """
        Build a CNN model for EEG classification
        
        Args:
            input_shape: Shape of input data (channels, samples)
            num_classes: Number of output classes
        """
        model = keras.Sequential([
            layers.Input(shape=input_shape + (1,)),
            
            # Convolutional blocks
            layers.Conv2D(32, (1, 5), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((1, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (1, 5), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((1, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (1, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((1, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info(f"Built CNN model: {model.summary()}")
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """
        Train the decoder
        
        Args:
            X: Feature matrix (samples x features)
            y: Labels (samples,)
            validation_split: Validation set size
            
        Returns:
            Training history/results
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        if self.model_type == 'cnn':
            # Reshape for CNN (add channel dimension)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1, 1)
            
            # Build model if not already built
            if self.model is None:
                self.build_cnn_model((X_train.shape[1], 1), len(np.unique(y)))
            
            # Train
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                verbose=1
            )
            
            self.is_trained = True
            return history
        else:
            # Train sklearn model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            val_score = self.model.score(X_val, y_val)
            
            logger.info(f"Training accuracy: {train_score:.4f}")
            logger.info(f"Validation accuracy: {val_score:.4f}")
            
            self.is_trained = True
            
            return {
                'train_accuracy': train_score,
                'val_accuracy': val_score
            }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'cnn':
            X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1, 1)
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'cnn':
            X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1, 1)
            return self.model.predict(X_scaled)
        else:
            return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Performance metrics
        """
        predictions = self.predict(X)
        
        if self.model_type == 'cnn':
            predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(y, predictions)
        conf_matrix = confusion_matrix(y, predictions)
        report = classification_report(y, predictions, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': report
        }
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model_type == 'cnn':
            self.model.save(filepath)
            # Save scaler separately
            with open(filepath + '_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        if self.model_type == 'cnn':
            self.model = keras.models.load_model(filepath)
            with open(filepath + '_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    
    # Simulate EEG features (200 samples, 50 features)
    X = np.random.randn(200, 50)
    y = np.random.randint(0, 4, 200)  # 4 classes (left, right, forward, backward)
    
    # Initialize decoder
    decoder = NeuralDecoder(model_type='lda')
    
    # Train
    results = decoder.train(X, y)
    print("Training results:", results)
    
    # Evaluate
    metrics = decoder.evaluate(X[:50], y[:50])
    print("Evaluation metrics:", metrics)
    
    # Predict
    predictions = decoder.predict(X[:10])
    print("Predictions:", predictions)