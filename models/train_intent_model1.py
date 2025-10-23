import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os


training_data = {
    'query': [
        
        "What is the total revenue in 2023?",
        "How many employees does the company have?",
        "What is the carbon emission target?",
        "When was the sustainability report published?",
        "What is the renewable energy percentage?",
        "How much water was consumed?",
        "What are the company's main goals?",
        "Where are the manufacturing facilities located?",
        "What is the waste reduction target?",
        "How many electric vehicles were produced?",
        "What is the greenhouse gas emission?",
        "What certifications does the company have?",
        "What is the energy consumption per unit?",
        "How many solar panels were installed?",
        "What is the recycling rate?",
        "What is the total carbon footprint?",
        "How much renewable energy is used?",
        "What are the emission reduction goals?",
        "How many tons of waste were recycled?",
        "What is the water conservation target?",
        "What percentage of energy is clean?",
        "How many sustainability projects exist?",
        "What is the environmental compliance rate?",
        "How many green buildings are there?",
        "What is the biodiversity impact score?",
        "How many electric vehicles did Amazon deploy in 2023?",
        "What percentage of Amazon’s electricity came from renewable energy?",
        "How much total carbon did Amazon emit in 2023?",
        "How many Climate Pledge signatories were there by 2023?",
        "How many boxes were saved by the Amazon Day program?",
        "Compare emissions between 2022 and 2023",
        "Analyze the trend in renewable energy usage",
        "How does Tesla compare to Google in sustainability?",
        "What is the difference between Q1 and Q4 performance?",
        "Compare water usage across different facilities",
        "Analyze the relationship between production and emissions",
        "How have carbon emissions changed over time?",
        "Compare the environmental impact of different products",
        "What are the trends in waste management?",
        "Analyze energy efficiency improvements",
        "Compare sustainability metrics year over year",
        "How do different regions perform on emissions?",
        "What patterns exist in resource consumption?",
        "Analyze the cost-benefit of renewable investments",
        "Compare recycling rates across quarters",
        "How do solar and wind energy compare?",
        "Analyze the correlation between growth and emissions",
        "Compare environmental performance across divisions",
        "What differences exist between facilities?",
        "Analyze trends in water conservation efforts",
        "Compare energy sources over time",
        "How does current performance compare to targets?",
        "Analyze the impact of sustainability initiatives",
        "Compare waste reduction across different sites",
        "What trends emerge from the environmental data?",
        "Compare Amazon’s carbon intensity in 2022 and 2023",
        "Analyze Amazon’s renewable energy adoption progress",
        "How has Amazon’s EV fleet grown over time?",
        "Compare packaging waste reduction year over year for Amazon",
        "Analyze the correlation between Amazon’s growth and emissions decline",
        "Summarize the sustainability report",
        "Give me an overview of environmental initiatives",
        "Summarize key achievements in 2023",
        "What are the main highlights of the report?",
        "Provide a summary of climate action goals",
        "Summarize the environmental impact section",
        "Give an overview of renewable energy projects",
        "Summarize the waste management strategy",
        "What are the key takeaways from the report?",
        "Summarize corporate social responsibility efforts",
        "Give me a brief overview of emissions data",
        "Summarize water conservation initiatives",
        "What are the main sustainability commitments?",
        "Summarize progress toward environmental goals",
        "Give an overview of green technology adoption",
        "Provide a summary of the executive summary",
        "Summarize the carbon neutrality roadmap",
        "Give an overview of the entire document",
        "Summarize environmental achievements and challenges",
        "What are the key points from the report?",
        "Summarize renewable energy investments",
        "Give an overview of sustainability metrics",
        "Summarize the climate strategy",
        "Provide a summary of environmental commitments",
        "Summarize the main findings of the report",
        "Summarize Amazon’s 2023 Sustainability Report",
        "Give an overview of Amazon’s climate goals and progress",
        "Summarize Amazon’s renewable energy achievements",
        "What are the main highlights of Amazon’s 2023 report?",
        "Provide a summary of Amazon’s decarbonization strategy",
        "What does the emissions chart show?",
        "Explain the graph on page 5",
        "Describe the renewable energy diagram",
        "What information is in the pie chart?",
        "Show me the trend in the line graph",
        "What does the bar chart indicate?",
        "Explain the sustainability metrics visualization",
        "Describe the image showing solar panels",
        "What is shown in the carbon footprint figure?",
        "Explain the infographic about water usage",
        "What does the table on emissions display?",
        "Describe the chart comparing different years",
        "What is depicted in the facility image?",
        "Explain the diagram of energy flow",
        "What does the visualization on recycling show?",
        "Describe the chart showing emission trends",
        "What information is in the graph?",
        "Explain the image of the manufacturing plant",
        "What does the figure illustrate?",
        "Describe the visualization on page 10",
        "What is shown in the sustainability dashboard?",
        "Explain the chart with the colored bars",
        "What does the picture of solar arrays show?",
        "Describe the diagram in the report?",
        "What information does the visual present?",
        "What does the Amazon carbon footprint chart illustrate?",
        "Explain the renewable energy capacity graph in Amazon’s report",
        "Describe the emissions reduction figure from 2022–2023",
        "What does the packaging waste chart show?",
        "Explain the infographic showing Amazon’s EV deployment progress",
    ],

    'intent': (
        ['fact'] * 30 +
        ['analysis'] * 30 +
        ['summary'] * 30 +
        ['visual'] * 30
    )
}

# Create DataFrame
df = pd.DataFrame(training_data)

print("=" * 60)
print("🤖 INTENT CLASSIFICATION MODEL TRAINING")
print("=" * 60)
print(f"\n📊 Training Data Summary:")
print(f"   Total samples: {len(df)}")
print(f"\n   Intent Distribution:")
print(df['intent'].value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['query'],
    df['intent'],
    test_size=0.2,
    random_state=42,
    stratify=df['intent']
)

print(f"\n📦 Data Split:")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Vectorize text
print("\n🔄 Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=300,
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.9
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
print("🎯 Training Logistic Regression model...")
model = LogisticRegression(
    max_iter=2000,
    random_state=42,
    C=10.0,
    solver='lbfgs'
)
model.fit(X_train_vec, y_train)

# Evaluate
print("\n✅ Model Training Complete!")
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📈 Model Performance:")
print(f"   Accuracy: {accuracy * 100:.2f}%")
print(f"\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save model and vectorizer
os.makedirs("models", exist_ok=True)

with open("models/intent_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n💾 Model Saved Successfully!")
print("   📁 models/intent_model.pkl")
print("   📁 models/vectorizer.pkl")

# Test with examples
print("\n" + "=" * 60)
print("🧪 TESTING WITH SAMPLE QUERIES")
print("=" * 60)

test_queries = [
    "What does the emissions chart show?",
    "Compare 2022 and 2023 data for Amazon",
    "What is Amazon’s total emission?",
    "Summarize Amazon’s sustainability report",
    "Analyze Amazon’s renewable energy growth",
    "Describe the Amazon EV infographic",
    "How many solar panels are installed?",
    "Give me an overview of Amazon’s progress"
]

for query in test_queries:
    query_vec = vectorizer.transform([query])
    prediction = model.predict(query_vec)[0]
    probabilities = model.predict_proba(query_vec)[0]
    confidence = max(probabilities) * 100
    print(f"\n❓ Query: '{query}'")
    print(f"   ✅ Predicted Intent: {prediction} (Confidence: {confidence:.1f}%)")

print("\n" + "=" * 60)
print("🎉 MODEL TRAINING COMPLETE WITH AMAZON PDF QUERIES!")
print("=" * 60)
