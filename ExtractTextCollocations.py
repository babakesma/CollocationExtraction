# Install NLTK if you haven't already
# !pip install nltk

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

# Sample text data (synthetic example)
text = """
Estimates show that the Iranian stock market is highly progressive and only just bullish.
The point here is that if the fever of risks increases and their effect on the economy increases, there is even a possibility that the stock market will not grow.
I guess correctly many assumptions in journals which say about Iranian stock market is not true.
Iranian stock market it is not well-designed.
"""


# Tokenize the text into words
tokens = nltk.word_tokenize(text)

# Perform POS tagging on the tokens
tagged_tokens = nltk.pos_tag(tokens)

# Create a BigramCollocationFinder from the tagged tokens
finder = BigramCollocationFinder.from_words(tagged_tokens)

# Define the desired collocation criteria (noun followed by an adjective)
def is_valid_collocation(word1, word2):
    condition=(word1[1].startswith("N") and word2[1].startswith("V"))
    condition=(word1[1].startswith("N") and word2[1].startswith("N")) or condition
    condition=(word1[1].startswith("N") and word2[1].startswith("J")) or condition
    condition=(word1[1].startswith("V") and word2[1].startswith("N")) or condition 
    condition=(word1[1].startswith("V") and word2[1].startswith("I")) or condition
    condition=(word1[1].startswith("V") and word2[1].startswith("R")) or condition
    condition=(word1[1].startswith("R") and word2[1].startswith("J")) or condition 
    condition=(word1[1].startswith("R") and word2[1].startswith("N")) or condition
    condition=(word1[1].startswith("R") and word2[1].startswith("R")) or condition
    condition=(word1[1].startswith("R") and word2[1].startswith("V")) or condition
    condition=(word1[1].startswith("J") and word2[1].startswith("N")) or condition
    condition=(word1[1].startswith("I") and word2[1].startswith("N")) or condition
    return condition


# Filter and score the collocations based on the criteria
scored_collocations = finder.score_ngrams(BigramAssocMeasures.pmi)
filtered_collocations = [col for col in scored_collocations if is_valid_collocation(col[0][0], col[0][1])]

# Sort the collocations by score
#sorted_collocations = sorted(filtered_collocations, key=lambda x: x[1], reverse=True)

# Print the top collocations (noun-adjective pairs)

#
#for collocation, score in sorted_collocations[:20]:  # Change the number to get more or fewer collocations
#    print(collocation, score)

collocation_counts = {}
for collocation, _ in filtered_collocations:
    collocation_str = f"{collocation[0][0]} {collocation[0][1]}"
    count= text.count(collocation_str)
    collocation_counts[collocation_str] = count
    print(f"{collocation}: {count}")
# Print the collocation counts
