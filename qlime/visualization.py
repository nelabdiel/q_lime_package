import matplotlib.pyplot as plt
import re
from IPython.display import HTML

def visualize_q_lime(vector, contributions, vectorizer):
    """
    Visualize Q-LIME feature contributions as a bar graph.
    
    Args:
        vector (np.array): Binary feature vector.
        contributions (np.array): Feature contributions.
        vectorizer (CountVectorizer): Vectorizer to map indices to words.
    """
    feature_names = vectorizer.get_feature_names_out()
    non_zero_indices = [i for i, c in enumerate(contributions) if abs(c) > 1e-7]
    
    words = [feature_names[i] for i in non_zero_indices]
    values = [contributions[i] for i in non_zero_indices]

    colors = ['green' if v > 0 else 'red' for v in values]
    plt.barh(words, values, color=colors)
    plt.xlabel("Contribution")
    plt.title("Q-LIME Feature Contributions")
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
        word_colors[word] = "green" if contributions[idx] > 0 else "red"

    highlighted_text = []
    for word in text.split():
        clean_word = re.sub(r'[^\w]', '', word.lower())  # Remove punctuation
        if clean_word in word_colors:
            color = word_colors[clean_word]
            highlighted_text.append(f'<span style="color:{color}; font-weight:bold;">{word}</span>')
        else:
            highlighted_text.append(word)

    return " ".join(highlighted_text)
