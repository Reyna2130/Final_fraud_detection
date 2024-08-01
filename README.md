# Final_fraud_detection

This project aims to detect fraudulent credit card transactions using machine learning algorithms. The system incorporates various features such as biometric authentication, data visualizations, synthetic data testing, location-based fraud detection, and secure databases to ensure robust fraud detection capabilities.

## Features

1. **Biometric Authentication**
   - Ensures secure access to the fraud detection page.

2. **Data Visualizations**
   - Confusion Matrix: Displays true positives, false positives, true negatives, and false negatives.
   - Bar Plots: Shows the distribution of fraudulent and non-fraudulent transactions.
   - Line Charts: Tracks transaction trends and anomalies over time.

3. **Model Accuracy**
   - The machine learning model achieves an accuracy of 0.99, indicating high reliability in detecting fraudulent transactions.

4. **Synthetic Data Testing**
   - Synthetic data with 10 transactions is generated for testing. Normal transactions are highlighted in green, and fraudulent transactions in red. Transaction IDs detected as fraud are also displayed.

5. **Location-Based Fraud Detection**
   - Detects fraud based on the user's current location. Any transaction initiated from a location other than Kolkata is flagged as potentially fraudulent.

6. **Company-Based Fraud Detection**
   - Uses a dataset of company names commonly reported for involvement in scams. Transactions associated with companies from this dataset are considered fraudulent.

7. **Real-Time Alerts**
   - If fraud is detected, an alert message is sent to the user's registered phone number with relevant transaction details.

8. **Multiple Transactions Detection**
   - Detects fraud when multiple transactions of the same amount (less than the amount required for OTP) are made back-to-back.

9. **Secure Database**
   - Ensures that all transaction data is stored securely.
