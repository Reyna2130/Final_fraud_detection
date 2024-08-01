import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib.colors import ListedColormap

# Load dataset
df = pd.read_csv('pages/creditcard.csv', low_memory=False)

# Set the style of the plots
plt.style.use('dark_background')  # Use dark background style

# Plot 1: Transaction Amount Over Time
st.markdown("<h2 style='color: white;'>Transaction Amount Over Time</h2>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Time'], df['Amount'], linestyle='-', marker='', color='#02CCFE')
ax.set_title('Transaction Amount Over Time', color='white')
ax.set_xlabel('Time', color='white')
ax.set_ylabel('Amount', color='white')
fig.patch.set_facecolor('black')  # Set the figure background color
ax.set_facecolor('black')  # Set the axes background color
st.pyplot(fig)

# Plot 2: Confusion Matrix for Logistic Regression
st.markdown("<h2 style='color: white;'>Confusion Matrix</h2>", unsafe_allow_html=True)
X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
colors = ["#B6D0E2", "#87CEEB", "#1E3F66", "#4682B4"]
cmap = ListedColormap(colors)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=cmap, ax=ax, values_format='.2f')
ax.set_title('Confusion Matrix', color='white')
fig.patch.set_facecolor('black')  # Set the figure background color
ax.set_facecolor('black')  # Set the axes background color
st.pyplot(fig)

# Plot 3: Mean Transaction Amount by Class
st.markdown("<h2 style='color: white;'>Mean Transaction Amount by Class</h2>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(8, 6))
df.groupby('Class')['Amount'].mean().plot(kind='bar', color=['#87CEEB', '#4682B4'], ax=ax)
ax.set_title('Mean Transaction Amount by Class', color='white')
ax.set_xlabel('Class (0: Non-Fraudulent, 1: Fraudulent)', color='white')
ax.set_ylabel('Mean Transaction Amount', color='white')
fig.patch.set_facecolor('black')  # Set the figure background color
ax.set_facecolor('black')  # Set the axes background color
st.pyplot(fig)

# Plot 4: Features Over Time
st.markdown("<h2 style='color: white;'>Features Over Time</h2>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Time'], df['V1'], color='#B6D0E2', label='V1')
ax.plot(df['Time'], df['V2'], color='#87CEEB', label='V2')
ax.plot(df['Time'], df['V3'], color='#4682B4', label='V3')
ax.plot(df['Time'], df['V4'], color='#4682B4', label='V4')
ax.set_title('Features Over Time', color='white')
ax.set_xlabel('Time', color='white')
ax.set_ylabel('Feature Value', color='white')
ax.legend()
fig.patch.set_facecolor('black')  # Set the figure background color
ax.set_facecolor('black')  # Set the axes background color
st.pyplot(fig)



# Separate features and target variable
X = df.iloc[:, :-1]
y = df['Class']

# Scale the features
X_scaled = scale(X)

# Apply PCA to reduce to 10 components
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X_scaled)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.33, random_state=500)

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Build the neural network
model = Sequential()
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(27, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
model.fit(X_train_sm, y_train_sm, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
st.write(f"Model Accuracy: {accuracy:.2f}")

# Make predictions on original test set
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Generate synthetic test data
# Creating 10 synthetic samples
synthetic_data = np.random.randn(10, X.shape[1])

# Manually set some samples to be fraudulent
synthetic_data[-2:] = X[df['Class'] == 1].sample(2).values

# Scale the synthetic data
synthetic_data_scaled = scale(synthetic_data)

# Apply PCA to the synthetic data
synthetic_data_reduced = pca.transform(synthetic_data_scaled)

# Make predictions on synthetic data
synthetic_predictions = (model.predict(synthetic_data_reduced) > 0.5).astype("int32")

# Display the synthetic data and their predictions
synthetic_df = pd.DataFrame({
    'ID': range(1, len(synthetic_data) + 1),
    'Prediction': synthetic_predictions.flatten()
})

# Function to color the rows based on the prediction
def highlight_predictions(val):
    color = 'background-color: green' if val == 0 else 'background-color: red'
    return color

# Style the DataFrame
styled_df = synthetic_df.style.applymap(highlight_predictions, subset=['Prediction'])

# Display the styled DataFrame
st.markdown("<h2 style='color: white;'>Synthetic Data Predictions</h2>", unsafe_allow_html=True)
st.dataframe(styled_df)

# Print a statement about fraudulent transactions
fraudulent_transactions = synthetic_df[synthetic_df['Prediction'] == 1]
if not fraudulent_transactions.empty:
    for idx in fraudulent_transactions['ID']:
        st.write(f"Transaction with ID {idx} was found to be fraudulent.")
else:
    st.write("No fraudulent transactions were detected in the synthetic data.")
