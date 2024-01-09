import nltk
import pandas
import numpy
import matplotlib.pyplot as matplot
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

tweets = pandas.read_csv("2022.csv")        # Create a data frame with the Musks's tweet objects.

vader_compound = []
vader_positive = []
vader_negative = []
vader_neutral  = []

for tweet in tweets["tweet"]:                                     # For every Elon Musk tweet, 
    positive_score = analyzer.polarity_scores(tweet)["pos"]       # Perform sentiment analysis,
    vader_positive.append(positive_score)                         # Add to a list.

    negative_score = analyzer.polarity_scores(tweet)["neg"]
    vader_negative.append(negative_score)

    neutral_score = analyzer.polarity_scores(tweet)["neu"]
    vader_neutral.append(neutral_score)

    compound_score = analyzer.polarity_scores(tweet)["compound"]                                               
    vader_compound.append(compound_score) 

# Add each of the lists as a column to the data frame. 

tweets["vader_positive"] = vader_positive
tweets["vader_negative"] = vader_negative
tweets["vader_neutral"]  = vader_neutral
tweets["vader_compound"] = vader_compound

print()

# Overall Averages, All Tweets

print("Positive Average: " + str(tweets["vader_positive"].mean()))  # Positive Average
print("Negative Average: " + str(tweets["vader_negative"].mean()))  # Negative Average
print("Neutral Average: " + str(tweets["vader_neutral"].mean()))    # Neutral Average
print("Compound Average: " + str(tweets["vader_compound"].mean()))  # Compound Average

print()


# Line Plot, January Tweets, Compound Score

matplot.style.use("_mpl-gallery")

compound_january = []

tweet_number = 0

for date in tweets["date"]:
    if date[0:7] == "2022-01":
        compound_january.append(tweets["vader_compound"][tweet_number])
    tweet_number += 1

matplot.plot(compound_january)

matplot.show()

