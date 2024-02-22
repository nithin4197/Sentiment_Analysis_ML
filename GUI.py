import tkinter as tk
from tkinter import messagebox
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Load the pickled model
model = joblib.load('svm_model.pkl')  
vectorizer = joblib.load('vectorizer.pkl') 

def predict():
    input_text = entry.get()
    if not input_text:
        messagebox.showwarning("Warning", "Please enter text for prediction.")
        return

    # Preprocess the input text (similar to how you preprocessed your training data)
    input_vectorized = vectorizer.transform([input_text])

    # Make prediction
    prediction = model.predict(input_vectorized)
    messagebox.showinfo("Prediction", f"The predicted sentiment of the feedback is: {prediction[0]}")

def show_metrics():
    # Make sure the test.csv file is in the same folder as GUI.py
    test = pd.read_csv("./test.csv",
                       encoding='latin1')
    test.dropna(subset=['text'], inplace=True)
    drop_col = ['textID', 'Time of Tweet', 'Population -2020', 'Land Area (Km²)', 'Density (P/Km²)', 'Country',
                'Age of User']
    Test = test.drop(drop_col, axis=1)

    X_test = vectorizer.transform(Test['text'])
    y_test_true = Test['sentiment']

    # Make predictions on the test set
    y_test_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test_true, y_test_pred)
    report = classification_report(y_test_true, y_test_pred)

    messagebox.showinfo("Model Evaluation", f"Accuracy: {accuracy:.2f}\n\nClassification Report:\n{report}")

# Create the main window
root = tk.Tk()
root.title("ML Model Prediction GUI")

# Set the background color to black
root.configure(bg='#000000')

# Entry widget for input text
entry = tk.Entry(root, width=40, font=('Arial', 14))  # Increase font size
entry.pack(pady=10)  # Center the entry widget

# Button for prediction
predict_button = tk.Button(root, text="Predict", command=predict, bg='#4CAF50', fg='white', relief=tk.GROOVE, height=2, width=15)
predict_button.pack(pady=5)  # Center the button

# Button for showing accuracy and classification report
show_metrics_button = tk.Button(root, text="Model Metrics", command=show_metrics, bg='#008CBA', fg='white', relief=tk.GROOVE, height=2, width=15)
show_metrics_button.pack(pady=5)  # Center the button

# Run the main loop
root.mainloop()
