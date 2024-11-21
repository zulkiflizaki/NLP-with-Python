"""This is an example of NLTK application"""

# import libraries
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string


# 1. Dataset 
# from voice to text
# elevenlabs-python


# sample text to be preprocessed
#text = 'SEGi College Penang is a very famous edutech in the IT industry!!'
text = 'A key concept in probability theory, the Bayes theorem, provides the foundation for the probabilistic classifier known as Naive Bayes. It is a simple yet powerful algorithm that has risen in popularity because of its understanding, simplicity, and ease of implementation. Naive Bayes Algorithm is a popular method for classification applications, especially spam filtering and text classification. In this article, we will learn about Naive Bayes Classifier From Scratch in Python.'


# 2. Data preprocessing
# tokenize
tokens = word_tokenize(text)

# remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

# perform stemming and lemmatization
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
lemmatize_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

cleaned_tokens = [token for token in lemmatize_tokens if not token.isdigit() and not token in string.punctuation]

pos_tags = pos_tag(cleaned_tokens)

named_entities = ne_chunk(pos_tags)

# 3. Bag-of-Words (BoW) Representation
count_vector = CountVectorizer();
BOW = count_vector.fit_transform(tokens)

print(f'Our vocabulary: {count_vector.vocabulary_}')

print(f'BoW representation for {text} {BOW}')
#print(f'Original text: {text}')
#print(f'Extracted tokens: {tokens}')
#print(f'Preprocessed tokens: {filtered_tokens}')
#print(f'Stemmed tokens: {stemmed_tokens}')
#print(f'Lemmatized tokens: {lemmatize_tokens}')
#print(f'Cleaned tokens: {cleaned_tokens}')
#print(f'Part-of-Speech (POS): {pos_tags}')
#print(f'Named Entity: {named_entities}')

wordcloud = WordCloud().generate(text)
plt.figure(figsize = (30,30))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()



# 4. Building a Naïve Bayes Classifier



# 5. Prediction on Unseen Reviews