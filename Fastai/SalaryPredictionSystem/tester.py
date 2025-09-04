from fastai.tabular.all import load_learner
import pandas as pd

# Load the previously exported learner
learn = load_learner('adult_model.pkl')

# Create a sample input record (as a dictionary)
sample_data = {
    'age': 37,
    'workclass': 'Private',
    'fnlwgt': 34146,
    'education': 'Bachelors',
    'marital-status': 'Married-civ-spouse',
    'occupation': 'Exec-managerial',
    'relationship': 'Husband',
    'race': 'White',
    'education-num': 13
}

# Convert the dictionary to a Pandas Series (which supports to_frame())
sample_series = pd.Series(sample_data)

# Get the prediction
pred_row, pred_idx, probs = learn.predict(sample_series)

# For classification, the second element is the predicted class index.
# Convert it to a standard Python integer:
pred_index = pred_idx.item()

# Define your class names.
# IMPORTANT: Ensure that the order here exactly matches the encoding order used during training.
# For example, if your original salary labels were ["<=50K", ">50K"],
# then class 0 corresponds to '<=50K' and class 1 to '>50K'
class_names = ['<=50K', '>50K']

# Translate the predicted index into the human-readable label:
predicted_label = class_names[pred_index]

# Convert the probabilities tensor to a dictionary mapping label to probability
probs_dict = {label: float(prob) for label, prob in zip(class_names, probs)}

print("Predicted Label:", predicted_label)
print("Prediction Probabilities:")
for label, prob in probs_dict.items():
    print(f"  {label}: {prob*100:.2f}%")