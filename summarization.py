import matplotlib.pyplot as plt

def summarize_content(text_content):
    stop_words = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "have", "but", "not",
    "they", "his", "her", "she", "him", "you", "your", "yours", "me",
    "my", "i", "we", "our", "ours", "had", "been", "do", "does", "did",
    "doing", "am", "all", "any", "more", "most", "other", "some", "such",
    "no", "nor", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "don", "should", "now", "linkedin", "instagram",
    "facebook", "join", "us"
}

    words = text_content.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    summarized =  " ".join(filtered_words)
    return summarized

text = summarize_content(text_SVC)
text_summ_SVC = text
text_summ_BERT = summarize_content(text_BERT)


before_count = len(text_BERT.split())
after_count = len(text.split())

labels = ['Before Summarization', 'After Summarization']
counts = [before_count, after_count]

plt.bar(labels, counts)
plt.xlabel('Text')
plt.ylabel('Number of Words')
plt.title('Word Count Before and After Summarization')
plt.show()