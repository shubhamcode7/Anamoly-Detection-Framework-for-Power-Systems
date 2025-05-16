import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load and preprocess the dataset
data = pd.read_csv(r"Datasets/triple/data2.csv")
X = data.drop(columns=['marker'])
y = data['marker']

# Handle infinities and missing values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Apply PCA for dimensionality reduction (keeping 95% variance)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Plot the explained variance ratio by PCA components
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()

# Define BPSWO class for feature selection
class BPSWO:
    def __init__(self, n_particles, n_features, max_iter, inertia, cognitive, social):
        self.n_particles = n_particles
        self.n_features = n_features
        self.max_iter = max_iter
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

    def initialize_particles(self):
        particles = [random.choices([0, 1], k=self.n_features) for _ in range(self.n_particles)]
        velocities = [[random.uniform(-1, 1) for _ in range(self.n_features)] for _ in range(self.n_particles)]
        return particles, velocities

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update_velocity_position(self, velocities, particles, personal_best, global_best):
        for i in range(self.n_particles):
            for j in range(self.n_features):
                r1, r2 = random.random(), random.random()
                velocities[i][j] = (
                    self.inertia * velocities[i][j]
                    + self.cognitive * r1 * (personal_best[i][j] - particles[i][j])
                    + self.social * r2 * (global_best[j] - particles[i][j])
                )
                particles[i][j] = 1 if random.random() < self.sigmoid(velocities[i][j]) else 0

    def train(self, X_train, y_train):
        particles, velocities = self.initialize_particles()
        personal_best = particles[:]
        global_best = max(personal_best, key=lambda p: self.evaluate_fitness(p, X_train, y_train))
        best_fitness = -1

        for _ in range(self.max_iter):
            self.update_velocity_position(velocities, particles, personal_best, global_best)

            for i in range(self.n_particles):
                fitness = self.evaluate_fitness(particles[i], X_train, y_train)
                if fitness > self.evaluate_fitness(personal_best[i], X_train, y_train):
                    personal_best[i] = particles[i][:]
                if fitness > best_fitness:
                    global_best = particles[i][:]
                    best_fitness = fitness

        return global_best

    def evaluate_fitness(self, particle, X_train, y_train):
        selected_features = [i for i in range(len(particle)) if particle[i] == 1]
        if not selected_features:
            return 0

        X_train_selected = X_train[:, selected_features]
        classifier = RandomForestClassifier()
        classifier.fit(X_train_selected, y_train)
        predictions = classifier.predict(X_train_selected)
        return accuracy_score(y_train, predictions)

# Run BPSWO on PCA-reduced dataset
bpswo = BPSWO(n_particles=8, n_features=X_train_pca.shape[1], max_iter=5, inertia=0.5, cognitive=1.0, social=1.0)
best_features = bpswo.train(X_train_pca, y_train)
selected_features = [i for i, selected in enumerate(best_features) if selected == 1]

# Visualize selected features
plt.figure(figsize=(10, 5))
sns.barplot(x=list(range(len(best_features))), y=best_features, color='skyblue')
plt.xlabel('Feature Index')
plt.ylabel('Selected (1) vs Not Selected (0)')
plt.title('BPSWO Selected Features Visualization')
plt.show()

# Select the best features for final training and testing
X_train_selected = X_train_pca[:, selected_features]
X_test_selected = X_test_pca[:, selected_features]

# Train the final Random Forest classifier on selected features
final_classifier = RandomForestClassifier()
final_classifier.fit(X_train_selected, y_train)

# Make predictions and evaluate performance
y_pred = final_classifier.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Display the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Display the confusion matrix as a heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# Testing of the train model using Single Sample prediction
# ----- SINGLE SAMPLE PREDICTION -----

# 1. No event

sample_input = [[
    82.12777035, 131860.3269, -37.84386237, 131835.2537, -157.8556021, 131935.5467, 80.53494768,
    346.62723, -39.41949631, 348.82455, -159.2192417, 347.54278, 82.14495909, 131885.4002,
    0, 0, 0, 0, 80.63235051, 347.72589, 0, 0, 0, 0, 60, 0, 10.90036337, 0.030688873, 0,
    76.91254043, 130854.3906, -43.02520915, 130704.6406, -163.078307, 130934.2188, -109.5419319,
    352.4551392, 130.5313081, 352.4894714, 10.56060798, 352.8270874, 76.93451315, 130830.1563,
    0, 0, 0, 0, -109.481512, 352.586731, 0, 0, 0, 0, 59.99900055, 0, 10.56193604, -3.028617977, 0,
    76.91385442, 130381.004, -43.04631915, 130355.9307, -163.0465998, 130406.0773, -109.5667192,
    350.28943, 130.4796787, 350.10632, 10.47939807, 349.7401, 76.94250231, 130381.004,
    0, 0, 0, 0, -109.5323417, 349.92321, 0, 0, 0, 0, 59.999, 0, 10.5726427, -3.0266769, 0,
    82.19079571, 131910.4735, -37.79229617, 131283.6417, -157.7925768, 131985.6933, 80.7641308,
    345.71168, -39.25906812, 345.34546, -159.2536192, 345.34546, 82.20798445, 131734.9606,
    0, 0, 0, 0, 80.75267165, 345.34546, 0, 0, 0, 0, 60, 0, 10.83249353, 0.024871573, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]]

# Preprocess the sample
sample_scaled = scaler.transform(sample_input)
sample_pca = pca.transform(sample_scaled)
sample_selected = sample_pca[:, selected_features]

# Predict class
predicted_encoded = final_classifier.predict(sample_selected)
predicted_class = label_encoder.inverse_transform(predicted_encoded)

print("\n Prediction for given sample:")
print(f" Predicted Class: {predicted_class[0]}")

true_label = 'NoEvents'
true_encoded = label_encoder.transform([true_label])[0]

# Compare
is_correct = predicted_encoded[0] == true_encoded
print(f" Correct Prediction? {is_correct}")


# 2 Attack


sample_input = [[
     -51.58911987, 130757.1031, -171.5836709, 130130.2713, 68.42834947, 130807.2496,
    -56.80876539, 517.10264, -176.9522854, 518.38441, 63.17432649, 516.73642,
    -51.5833903, 130556.5169, 0, 0, 0, 0, -56.86033159, 517.46886, 0, 0, 0, 0,
    60, 0, 7.206279779, 0.091472567, 0,
    -59.54750365, 128124.4097, -179.5076772, 128099.3364, 60.51007274, 128149.483,
    116.3849169, 522.77905, -3.672659467, 523.87771, -123.6271035, 522.77905,
    -59.51312618, 128124.4097, 0, 0, 0, 0, 116.356269, 523.14527, 0, 0, 0, 0,
    60, 0, 6.977699027, -3.070043387, 0,
    -59.57615154, 128099.3364, -179.5420547, 128074.2632, 60.45850654, 128149.483,
    116.3448099, 519.84929, -3.930490475, 520.0324, -123.8104499, 520.58173,
    -59.55323323, 128099.3364, 0, 0, 0, 0, 116.2015704, 520.21551, 0, 0, 0, 0,
    60, 0, 7.009322602, -3.075297971, 0,
    -51.7208869, 131154.25, -171.653145, 131004.6875, 68.29925525, 131234.4531,
    -56.82403884, 514.4119263, -176.9046043, 515.9625854, 63.22357092, 515.3446045,
    -51.69342271, 131134.2188, 0, 0, 0, 0, -56.8350252, 515.2244873, 0, 0, 0, 0,
    59.99900055, 0, 7.274633082, 0.088453404, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0
]]

# Preprocess the sample
sample_scaled = scaler.transform(sample_input)
sample_pca = pca.transform(sample_scaled)
sample_selected = sample_pca[:, selected_features]

# Predict class
predicted_encoded = final_classifier.predict(sample_selected)
predicted_class = label_encoder.inverse_transform(predicted_encoded)

print("\n Prediction for given sample:")
print(f" Predicted Class: {predicted_class[0]}")

true_label = 'Attack'
true_encoded = label_encoder.transform([true_label])[0]

# Compare
is_correct = predicted_encoded[0] == true_encoded
print(f"Correct Prediction? {is_correct}")

# 3 Natural


sample_input = [[
     -74.4501359, 130807.2496, 165.5618845, 130155.3446, 45.57306302, 130857.3961,
    -79.34319547, 510.69379, 160.6344474, 511.24312, 40.29039215, 511.97556,
    -74.43867674, 130606.6634, 0, 0, 0, 0, -79.47497576, 511.24312, 0, 0, 0, 0,
    60, 0, 7.247031373, 0.08235373, 0,
    -82.31684643, 128199.6295, 157.7295514, 128199.6295, 37.72927081, 128249.7761,
    93.93070093, 516.55331, -26.22427828, 517.28575, -146.5167674, 517.83508,
    -82.28819854, 128224.7028, 0, 0, 0, 0, 93.72443613, 517.28575, 0, 0, 0, 0,
    60, 0, 7.02451515, -3.080765013, 0,
    -82.34549432, 128174.5562, 157.695174, 128149.483, 37.67770461, 128249.7761,
    93.79892064, 511.42623, -26.53367549, 514.5391, -146.5053082, 514.35599,
    -82.32830558, 128199.6295, 0, 0, 0, 0, 93.58692626, 513.44044, 0, 0, 0, 0,
    60, 0, 7.150497115, -3.080807586, 0,
    -74.57794079, 131212.0781, 165.4843231, 131057.9688, 45.43670647, 131302.4844,
    -79.31854287, 508.3865967, 160.7272329, 508.5239258, 40.3500392, 510.0917969,
    -74.55596807, 131187.6406, 0, 0, 0, 0, -79.41742352, 508.9988708, 0, 0, 0, 0,
    60, 0, 7.298229825, 0.079368709, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]]

# Preprocess the sample
sample_scaled = scaler.transform(sample_input)
sample_pca = pca.transform(sample_scaled)
sample_selected = sample_pca[:, selected_features]

# Predict class
predicted_encoded = final_classifier.predict(sample_selected)
predicted_class = label_encoder.inverse_transform(predicted_encoded)

print("\n Prediction for given sample:")
print(f" Predicted Class: {predicted_class[0]}")

true_label = 'Natural'
true_encoded = label_encoder.transform([true_label])[0]

# Compare
is_correct = predicted_encoded[0] == true_encoded
print(f"Correct Prediction? {is_correct}")


# Now do the confusion matrix
true_labels = ['NoEvents', 'Attack', 'Natural']
true_encoded = label_encoder.transform(true_labels)
predicted_encoded = label_encoder.transform(predicted_labels)

# Plot the confusion matrix
cm = confusion_matrix(true_encoded, predicted_encoded)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()