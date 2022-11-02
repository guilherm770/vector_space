import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import getcwd

from utils import get_vectors, cosine_similarity, euclidean, get_country, get_accuracy, compute_pca


filePath = f"{getcwd()}"

data = pd.read_csv(filePath+'/vector_space/data/capitals.txt', delimiter=' ')
data.columns = ['city1', 'country1', 'city2', 'country2']

# print first five elements in the DataFrame
print(data.head(5))


import nltk
from gensim.models import KeyedVectors

nltk.download('punkt')

embeddings = KeyedVectors.load_word2vec_format(filePath+'/vector_space/data/GoogleNews-vectors-negative300.bin', binary = True)
f = open(filePath+'/vector_space/data/capitals.txt', 'r').read()
set_words = set(nltk.word_tokenize(f))
select_words = words = ['king', 'queen', 'oil', 'gas', 'happy', 'sad', 'city', 'town', 'village', 'country', 'continent', 'petroleum', 'joyful']
for w in select_words:
    set_words.add(w)

def get_word_embeddings(embeddings):

    word_embeddings = {}
    for word in embeddings.key_to_index:
        if word in set_words:
            word_embeddings[word] = embeddings[word]
    return word_embeddings


# Testing get_word_embeddings function
word_embeddings = get_word_embeddings(embeddings)
print(len(word_embeddings))
pickle.dump( word_embeddings, open( filePath+"/vector_space/data/word_embeddings_subset.p", "wb" ) )

print("dimension: {}".format(word_embeddings['Spain'].shape[0]))

# feel free to try different words
king = word_embeddings['king']
queen = word_embeddings['queen']

print(cosine_similarity(king, queen))

# Test euclidean function
print(euclidean(king, queen))

# Testing get_country function, note to make it more robust you can return the 5 most similar words.
print(get_country('Athens', 'Greece', 'Cairo', word_embeddings))

accuracy = get_accuracy(word_embeddings, data)
print(f"Accuracy is {accuracy:.2f}")

# Testing compute_pca function
np.random.seed(1)
X = np.random.rand(3, 10)
X_reduced = compute_pca(X, n_components=2)
print("Your original matrix was " + str(X.shape) + " and it became:")
print(X_reduced)

words = ['oil', 'gas', 'happy', 'sad', 'city', 'town',
         'village', 'country', 'continent', 'petroleum', 'joyful']

# given a list of words and the embeddings, it returns a matrix with all the embeddings
X = get_vectors(word_embeddings, words)

print('You have 11 words each of 300 dimensions thus X.shape is:', X.shape)

# We have done the plotting for you. Just run this cell.
result = compute_pca(X, 2)
plt.figure(figsize=(10,10)) 
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0] - 0.0005, result[i, 1] + 0.001))

plt.savefig(filePath+'/vector_space/data/pca_pic.png')