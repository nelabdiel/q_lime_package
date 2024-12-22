# Q-LIME Pi: Quantum LIME Explanation Library

## Overview
Q-LIME (Quantum Local Interpretable Model-agnostic Explanations) is a Python package designed to generate interpretable explanations for machine learning models using a quantum-inspired approach. It builds on the principles of LIME (Local Interpretable Model-agnostic Explanations) but incorporates quantum mechanisms for flipping features and evaluating their contributions to model predictions.

This package assumes that you have a pre-trained model, vectorizer, and vectorized data. Q-LIME provides tools to:

1. Explain model predictions with feature contributions.
2. Visualize feature contributions as bar graphs.
3. Highlight contributing words in text with HTML formatting.

---

## Installation

### Requirements
Ensure you have the following Python dependencies:

- `numpy`
- `pennylane`
- `matplotlib`
- `scikit-learn`
- `IPython`

### Install the Package
You can install the package from source:

```bash
pip install .
```

Or, if hosted on a repository like PyPI or GitHub, you can install directly:

```bash
pip install qlime
```

---

## Usage

### 1. Importing Q-LIME Functions
```python
from qlime import explain, visualize_q_lime, highlight_text_with_contributions
```

### 2. Explaining a Prediction
Assume you have:
- A **trained classifier** (e.g., logistic regression).
- A **vectorized input** (e.g., a bag-of-words vector).
- The **vectorizer** used for feature mapping.

Here's how to compute feature contributions:

```python
# Example input: pre-trained classifier and vectorized input
vectorized_input = X_test_vectorized[0]  # A single vectorized instance
classifier_weights = classifier.coef_[0]  # Logistic regression weights

# Generate feature contributions
contributions = explain(vector=vectorized_input, weights=classifier_weights)
```

### 3. Visualizing Feature Contributions
To create a bar graph of feature contributions:

```python
visualize_q_lime(vector=vectorized_input, contributions=contributions, vectorizer=vectorizer)
```

This will produce a horizontal bar chart showing the contribution of each feature to the prediction. Positive contributions are shown in green, and negative contributions in red.

### 4. Highlighting Text
To highlight top-contributing words in the original text:

```python
highlighted_html = highlight_text_with_contributions(
    text=original_text, contributions=contributions, vectorizer=vectorizer
)

display(HTML(highlighted_html))  # Requires an IPython environment
```

---

## Example Workflow

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from qlime import explain, visualize_q_lime, highlight_text_with_contributions
from IPython.display import HTML, display

# Step 1: Prepare data and train a classifier
vectorizer = CountVectorizer(binary=True, stop_words='english', max_features=10)
X_train_vectorized = vectorizer.fit_transform(X_train)
classifier = LogisticRegression()
classifier.fit(X_train_vectorized, y_train)

# Step 2: Pick a test instance
idx = 0
original_text = X_test[idx]
vectorized_input = X_test_vectorized[idx]
classifier_weights = classifier.coef_[0]

# Step 3: Explain prediction
contributions = explain(vector=vectorized_input, weights=classifier_weights)

# Step 4: Visualize contributions
visualize_q_lime(vector=vectorized_input, contributions=contributions, vectorizer=vectorizer)

# Step 5: Highlight text
highlighted_html = highlight_text_with_contributions(
    text=original_text, contributions=contributions, vectorizer=vectorizer
)
display(HTML(highlighted_html))
```

---

## API Reference

### `explain(vector, weights)`
Computes feature contributions for a given vector using the Q-LIME method.

**Args:**
- `vector (np.array)`: Binary feature vector (1-D array).
- `weights (np.array)`: Logistic regression weights (1-D array).

**Returns:**
- `np.array`: Feature contributions (1-D array).

---

### `visualize_q_lime(vector, contributions, vectorizer)`
Generates a bar graph of feature contributions.

**Args:**
- `vector (np.array)`: Binary feature vector.
- `contributions (np.array)`: Feature contributions.
- `vectorizer (CountVectorizer)`: Vectorizer to map indices to feature names.

**Returns:**
- Displays a bar graph.

---

### `highlight_text_with_contributions(text, contributions, vectorizer, top_n=5)`
Highlights the top-N contributing words in a text using HTML.

**Args:**
- `text (str)`: Original text.
- `contributions (np.array)`: Feature contributions.
- `vectorizer (CountVectorizer)`: Vectorizer to map indices to feature names.
- `top_n (int)`: Number of words to highlight (default: 5).

**Returns:**
- `str`: HTML-formatted string with highlights.

---

## Contributing
If you'd like to contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with detailed explanations.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- LIME: Ribeiro et al. (2016)
- PennyLane: Quantum machine learning framework
- Scikit-learn: Machine learning toolkit

