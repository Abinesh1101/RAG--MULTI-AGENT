import pickle
import os

class IntentAgent:
    """
    Agent responsible for classifying user query intent.
    Uses trained ML model to classify into: fact, analysis, summary, visual
    """
    
    def __init__(self, model_path='models/intent_model.pkl', vectorizer_path='models/vectorizer.pkl'):
        """Initialize the Intent Agent with trained model and vectorizer"""
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.load_model()
    
    def load_model(self):
        """Load the trained intent classification model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("✅ Intent Agent: Model loaded successfully")
        except FileNotFoundError:
            print("❌ Error: Model files not found. Please train the model first.")
            raise
    
    def classify(self, query):
        """
        Classify the intent of a user query
        
        Args:
            query (str): User's question
            
        Returns:
            str: Intent category (fact, analysis, summary, visual)
        """
        if not query or not isinstance(query, str):
            return "fact"  # Default to fact if invalid query
        
        # Vectorize the query
        query_vec = self.vectorizer.transform([query])
        
        # Predict intent
        intent = self.model.predict(query_vec)[0]
        
        # Get confidence scores
        probabilities = self.model.predict_proba(query_vec)[0]
        confidence = max(probabilities) * 100
        
        print(f"🎯 Intent Agent: Classified as '{intent}' (Confidence: {confidence:.1f}%)")
        
        return intent
    
    def classify_with_confidence(self, query):
        """
        Classify intent and return confidence scores
        
        Args:
            query (str): User's question
            
        Returns:
            tuple: (intent, confidence_dict)
        """
        query_vec = self.vectorizer.transform([query])
        intent = self.model.predict(query_vec)[0]
        probabilities = self.model.predict_proba(query_vec)[0]
        
        # Create confidence dictionary
        confidence_dict = dict(zip(self.model.classes_, probabilities))
        
        return intent, confidence_dict


# Test the agent if run directly
if __name__ == "__main__":
    print("🧪 Testing Intent Agent...\n")
    
    agent = IntentAgent()
    
    test_queries = [
        "What is the total emissions in 2023?",
        "Compare Tesla and Google's sustainability efforts",
        "Summarize the environmental report",
        "What does the emissions chart show?",
        "Give me an overview of Amazon’s progress",
        "How many solar panels are installed?",
        "Analyze Amazon’s renewable energy growth",
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        intent = agent.classify(query)
        print()