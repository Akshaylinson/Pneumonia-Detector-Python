import random
import numpy as np
from datetime import datetime

class PneumoniaModelSimulator:
    def __init__(self):
        self.model_name = "ViT-PneumoniaDetector v2.1"
        self.confidence_threshold = 0.7
        
    def predict(self, filename):
        """Simulate AI model prediction based on filename patterns"""
        
        # Extract features from filename for more realistic simulation
        filename_lower = filename.lower()
        
        # Simulate different cases based on filename patterns
        if any(pattern in filename_lower for pattern in ['pneumonia', 'pn', 'abnormal', 'disease']):
            pneumonia_prob = random.uniform(0.75, 0.95)
        elif any(pattern in filename_lower for pattern in ['normal', 'healthy', 'clear']):
            pneumonia_prob = random.uniform(0.05, 0.25)
        else:
            # Random case for unknown filenames
            pneumonia_prob = random.uniform(0.1, 0.9)
        
        normal_prob = 1 - pneumonia_prob
        
        # Add some randomness to make it more realistic
        pneumonia_prob = self._add_prediction_variance(pneumonia_prob)
        normal_prob = 1 - pneumonia_prob
        
        # Determine primary label
        if pneumonia_prob > normal_prob:
            top_label = "Pneumonia Detected"
            confidence_level = self._get_confidence_level(pneumonia_prob)
        else:
            top_label = "Normal Chest X-Ray"
            confidence_level = self._get_confidence_level(normal_prob)
        
        return {
            'top_label': top_label,
            'top_prob': max(pneumonia_prob, normal_prob),
            'probs': {
                'Pneumonia': round(pneumonia_prob, 3),
                'Normal': round(normal_prob, 3)
            },
            'confidence': confidence_level,
            'model_used': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'findings': self._generate_findings(pneumonia_prob),
            'recommendations': self._generate_recommendations(pneumonia_prob)
        }
    
    def _add_prediction_variance(self, prob):
        """Add realistic variance to predictions"""
        variance = random.gauss(0, 0.05)  # Small random noise
        return max(0.01, min(0.99, prob + variance))
    
    def _get_confidence_level(self, probability):
        """Determine confidence level based on probability"""
        if probability >= 0.9:
            return "very high"
        elif probability >= 0.8:
            return "high"
        elif probability >= 0.7:
            return "moderate"
        else:
            return "low"
    
    def _generate_findings(self, pneumonia_prob):
        """Generate realistic medical findings based on probability"""
        if pneumonia_prob > 0.7:
            findings = [
                "Consolidation observed in lower lung fields",
                "Increased opacity in affected areas",
                "Possible air bronchograms present",
                "Border effacement suggesting pleural involvement"
            ]
        elif pneumonia_prob > 0.4:
            findings = [
                "Mild opacity changes in peripheral regions",
                "Inconclusive interstitial patterns",
                "Recommend clinical correlation",
                "Follow-up imaging suggested"
            ]
        else:
            findings = [
                "Clear lung fields bilaterally",
                "Normal cardiomediastinal silhouette",
                "No focal consolidation observed",
                "No pleural effusion or pneumothorax"
            ]
        
        return random.sample(findings, min(3, len(findings)))
    
    def _generate_recommendations(self, pneumonia_prob):
        """Generate medical recommendations based on prediction"""
        if pneumonia_prob > 0.7:
            return [
                "Urgent clinical evaluation recommended",
                "Consider antibiotic therapy based on clinical findings",
                "Follow-up chest X-ray in 48-72 hours",
                "Monitor oxygen saturation levels"
            ]
        elif pneumonia_prob > 0.4:
            return [
                "Clinical correlation required",
                "Consider laboratory tests (CBC, CRP)",
                "Repeat imaging if symptoms persist",
                "Close monitoring advised"
            ]
        else:
            return [
                "No immediate intervention required",
                "Routine follow-up as per standard care",
                "Continue monitoring for symptom development"
            ]
