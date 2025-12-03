"""
CRIME TYPE PREDICTION - LOGISTIC REGRESSION WITH PIE CHART
===========================================================
Predicts the likelihood/percentage of each crime type.

INPUTS:  Gender, Race, Location (CATEGORICAL ONLY)
OUTPUT:  Probability for each crime type + PIE CHART
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better looking charts
sns.set_style('whitegrid')

# ====================
# 1. LOAD DATA
# ====================

# >>> PUT YOUR TRAIN CSV FILENAME HERE <
train_df = pd.read_csv('YOUR_TRAIN_FILE.csv')

# >>> PUT YOUR TEST CSV FILENAME HERE <
test_df = pd.read_csv('YOUR_TEST_FILE.csv')

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")


# ====================
# 2. DEFINE FEATURES (CATEGORICAL ONLY)
# ====================

# >>> PUT YOUR CATEGORICAL COLUMN NAMES HERE <
categorical_features = [
    'gender',      # REPLACE ME
    'race',        # REPLACE ME
    'location',    # REPLACE ME
]

# >>> PUT YOUR CRIME TYPE COLUMN NAME HERE <
target_column = 'crime_type'  # REPLACE ME - must be TEXT (Theft, Assault, etc.)

print(f"\nCrime types in dataset: {sorted(train_df[target_column].unique())}")


# ====================
# 3. PREPARE DATA (Already split!)
# ====================

X_train = train_df[categorical_features]
y_train = train_df[target_column]

X_test = test_df[categorical_features]
y_test = test_df[target_column]


# ====================
# 4. BUILD PIPELINE
# ====================

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
     categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42))
])


# ====================
# 5. TRAIN MODEL
# ====================

print("\nTraining model...")
pipeline.fit(X_train, y_train)
print("✓ Model trained successfully!")


# ====================
# 6. EVALUATE MODEL
# ====================

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print("\n" + "="*60)
print("DETAILED PERFORMANCE BY CRIME TYPE")
print("="*60)
print(classification_report(y_test, y_test_pred))


# ====================
# 7. CHART: CONFUSION MATRIX
# ====================

print("\n📊 Creating confusion matrix chart...")
cm = confusion_matrix(y_test, y_test_pred)
crime_types = sorted(train_df[target_column].unique())

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=crime_types, yticklabels=crime_types,
            cbar_kws={'label': 'Number of Predictions'})
plt.xlabel('Predicted Crime Type', fontsize=12, fontweight='bold')
plt.ylabel('Actual Crime Type', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix - Logistic Regression Model', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")
plt.close()


# ====================
# 8. PREDICT FOR NEW PERSON
# ====================

# >>> PUT YOUR PERSON PROFILE HERE <
new_person = pd.DataFrame({
    'gender': ['Female'],      # REPLACE ME
    'race': ['White'],         # REPLACE ME
    'location': ['Downtown'],  # REPLACE ME
})

# Get crime type prediction
predicted_crime = pipeline.predict(new_person)

# Get probabilities for ALL crime types
crime_probabilities = pipeline.predict_proba(new_person)[0]
crime_types_list = pipeline.classes_

print("\n" + "="*60)
print("PREDICTION FOR NEW PERSON")
print("="*60)
print(f"\nProfile:")
print(f"  Gender:   {new_person['gender'].values[0]}")
print(f"  Race:     {new_person['race'].values[0]}")
print(f"  Location: {new_person['location'].values[0]}")

print(f"\n🎯 Most Likely Crime Type: {predicted_crime[0]}")
print(f"   Confidence: {max(crime_probabilities)*100:.2f}%")

print(f"\nIf you experience a crime, the likelihood of it being:")
# Sort by probability (highest first)
crime_prob_pairs = sorted(zip(crime_types_list, crime_probabilities), 
                          key=lambda x: x[1], reverse=True)

for i, (crime, prob) in enumerate(crime_prob_pairs, 1):
    print(f"  {i}. {crime:15s} {prob*100:5.2f}%")


# ====================
# 9. PIE CHART: CRIME TYPE PROBABILITIES
# ====================

print("\n📊 Creating pie chart for crime type probabilities...")

# Prepare data for pie chart
sorted_crimes = [x[0] for x in crime_prob_pairs]
sorted_probs = [x[1] * 100 for x in crime_prob_pairs]

# Create color palette
colors = plt.cm.Set3(range(len(sorted_crimes)))

# Create pie chart
plt.figure(figsize=(12, 8))
wedges, texts, autotexts = plt.pie(
    sorted_probs, 
    labels=sorted_crimes, 
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    explode=[0.1 if i == 0 else 0 for i in range(len(sorted_crimes))],  # Explode the largest slice
    shadow=True,
    textprops={'fontsize': 11, 'weight': 'bold'}
)

# Make percentage text more visible
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_weight('bold')

# Title with person's profile
profile_text = f"{new_person['gender'].values[0]}, {new_person['race'].values[0]}, {new_person['location'].values[0]}"
plt.title(f'Crime Type Probability Distribution\n{profile_text}', 
          fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('crime_probability_pie_chart.png', dpi=300, bbox_inches='tight')
print("✓ Saved: crime_probability_pie_chart.png")
plt.close()


# ====================
# 10. COMPARE MULTIPLE SCENARIOS
# ====================

# >>> PUT YOUR SCENARIOS HERE <
scenarios = pd.DataFrame({
    'gender': ['Female', 'Male', 'Female', 'Male'],
    'race': ['White', 'Black', 'Hispanic', 'Asian'],
    'location': ['Downtown', 'Suburb', 'Downtown', 'Park'],
})

# Predict crime types
scenarios['predicted_crime'] = pipeline.predict(scenarios)

# Get confidence (max probability)
scenario_probas = pipeline.predict_proba(scenarios)
scenarios['confidence'] = np.max(scenario_probas, axis=1) * 100

print("\n" + "="*60)
print("CRIME TYPE PREDICTIONS FOR MULTIPLE SCENARIOS")
print("="*60)
print(scenarios)


# ====================
# 11. PIE CHARTS FOR MULTIPLE SCENARIOS (2x2 GRID)
# ====================

print("\n📊 Creating pie charts for multiple scenarios...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx in range(min(4, len(scenarios))):
    probas = scenario_probas[idx] * 100
    
    # Sort by probability
    sorted_indices = np.argsort(probas)[::-1]
    sorted_crime_types = crime_types_list[sorted_indices]
    sorted_probabilities = probas[sorted_indices]
    
    # Create pie chart
    colors = plt.cm.Set3(range(len(sorted_crime_types)))
    wedges, texts, autotexts = axes[idx].pie(
        sorted_probabilities,
        labels=sorted_crime_types,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=[0.1 if i == 0 else 0 for i in range(len(sorted_crime_types))],
        textprops={'fontsize': 9, 'weight': 'bold'}
    )
    
    # Make percentage text white and bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_weight('bold')
    
    # Title for each subplot
    profile = f"{scenarios.iloc[idx]['gender']}, {scenarios.iloc[idx]['race']}, {scenarios.iloc[idx]['location']}"
    axes[idx].set_title(f'Scenario {idx+1}: {profile}', fontsize=11, fontweight='bold', pad=10)

plt.suptitle('Crime Type Probability Distribution - Multiple Scenarios', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('multiple_scenarios_pie_charts.png', dpi=300, bbox_inches='tight')
print("✓ Saved: multiple_scenarios_pie_charts.png")
plt.close()


# ====================
# 12. DETAILED BREAKDOWN FOR EACH SCENARIO
# ====================

print("\n" + "="*60)
print("DETAILED BREAKDOWN FOR EACH SCENARIO")
print("="*60)

for idx in range(len(scenarios)):
    print(f"\nScenario {idx+1}: {scenarios.iloc[idx]['gender']}, "
          f"{scenarios.iloc[idx]['race']}, "
          f"{scenarios.iloc[idx]['location']}")
    
    probas = scenario_probas[idx]
    crime_prob_pairs = sorted(zip(crime_types_list, probas), 
                              key=lambda x: x[1], reverse=True)
    
    print(f"  Most Likely: {crime_prob_pairs[0][0]} ({crime_prob_pairs[0][1]*100:.1f}%)")
    print(f"  Full breakdown:")
    for crime, prob in crime_prob_pairs[:5]:  # Top 5
        print(f"    - {crime:15s} {prob*100:5.1f}%")


# ====================
# 13. CRIME DISTRIBUTION IN DATA
# ====================

print("\n" + "="*60)
print("ACTUAL CRIME DISTRIBUTION IN YOUR DATA")
print("="*60)
crime_counts = train_df[target_column].value_counts()
crime_percentages = (crime_counts / len(train_df)) * 100

for crime, count in crime_counts.items():
    percentage = crime_percentages[crime]
    print(f"  {crime:15s} {count:5d} incidents ({percentage:5.1f}%)")


print("\n" + "="*60)
print("✓ ANALYSIS COMPLETE")
print("="*60)
print("\n📊 Charts created:")
print("  1. confusion_matrix.png")
print("  2. crime_probability_pie_chart.png (single person)")
print("  3. multiple_scenarios_pie_charts.png (2x2 grid)")