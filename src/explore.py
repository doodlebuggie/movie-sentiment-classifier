import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import movie_reviews

data_reviews = []

for category in movie_reviews.categories(): #goes through pos then neg
    for fileid in movie_reviews.fileids(category): # goes through every file in that cat
        text = movie_reviews.raw(fileid)
        data_reviews.append({"label": category, "text": text})

df = pd.DataFrame(data_reviews)
print(df.head())
print(df.tail(5))

#print the total number of rows, the count of pos vs neg, a few example texts, length/completeness checks

print(len(df))
print(df['label'].value_counts()) # check for balance between pos and neg
print("\ncheck for null")
print( df['text'].isnull())
print("\ncheck character counts")
print(df['text'].str.len())

#summary stats on lengths
print("Average ", df['text'].str.len().mean())
print("Median ", df['text'].str.len().median())
print("Minimum ", df['text'].str.len().min())
print("Maximum ", df['text'].str.len().max())

# from ai
df['text'].str.len().hist(bins=35)
plt.title("Distribution of Review Lengths (characters)")
plt.xlabel("Number of characters")
plt.ylabel("Number of reviews")
plt.savefig("outputs/review_length_distribution.png")
plt.show()