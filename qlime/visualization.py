import matplotlib.pyplot as plt
import re
from IPython.display import HTML

def visualize_q_lime(vector, contributions, vectorizer):
    """
    Visualize Q-LIME feature contributions as a bar graph, sorted by magnitude.
    
    Args:
        vector (np.array): Binary feature vector.
        contributions (np.array): Feature contributions.
        vectorizer (CountVectorizer): Vectorizer to map indices to words.
    """
    feature_names = vectorizer.get_feature_names_out()
    non_zero_indices = [i for i, c in enumerate(contributions) if abs(c) > 1e-7]
    
    # Extract feature names and contributions for non-zero indices
    words = [feature_names[i] for i in non_zero_indices]
    values = [contributions[i] for i in non_zero_indices]
    
    # Sort by absolute contribution values
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    sorted_words = [words[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    # Assign colors based on the contribution sign
    colors = ['blue' if v > 0 else 'red' for v in sorted_values]
    
    # Plot the bar graph
    plt.barh(sorted_words, sorted_values, color=colors, alpha=0.5, height=0.5)
    plt.xlabel("Contribution")
    plt.title("Q-LIME Feature Contributions (Sorted by Magnitude)")
    plt.gca().invert_yaxis()  # Invert y-axis for better readability
    plt.show()


def highlight_text_with_contributions(text, contributions, vectorizer, top_n=5):
    """
    Highlight top-N words in the text based on Q-LIME contributions.
    
    Args:
        text (str): Original text.
        contributions (np.array): Feature contributions.
        vectorizer (CountVectorizer): Vectorizer to map indices to words.
        top_n (int): Number of words to highlight.
    
    Returns:
        str: HTML-formatted text with highlights.
    """
    feature_names = vectorizer.get_feature_names_out()
    non_zero_indices = [i for i, c in enumerate(contributions) if abs(c) > 1e-7]

    # Get top-N contributing features
    top_contributors = sorted(
        non_zero_indices, key=lambda i: abs(contributions[i]), reverse=True
    )[:top_n]

    word_colors = {}
    for idx in top_contributors:
        word = feature_names[idx]
        word_colors[word] = "blue" if contributions[idx] > 0 else "red"

    highlighted_text = []
    for word in text.split():
        clean_word = re.sub(r'[^\w]', '', word.lower())  # Remove punctuation
        if clean_word in word_colors:
            color = word_colors[clean_word]
            highlighted_text.append(f'<span style="color:{color}; background-color:yellow; font-weight:bold; ">{word}</span>')
        else:
            highlighted_text.append(word)

    return " ".join(highlighted_text)
