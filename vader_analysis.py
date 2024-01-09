import nltk
import pandas
import numpy
import statistics
import matplotlib.pyplot as matplot
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import linregress

analyzer = SentimentIntensityAnalyzer()

data = pandas.read_csv("data.csv")

data['replies_vader_compound_score'] = 0        # Create a column to store the compound scores of each tweet's replies



replies = []

for x in range(len(data["reply_text"])):

    replies_list = []
    reply_text = ""

    string_reply_text = str(data["reply_text"][x])
    
    for i in range(len(string_reply_text )):

        if (string_reply_text[i] == ","):
            if (string_reply_text[i + 1] == " ") and (string_reply_text[i + 2] == "'" or '"') and (string_reply_text[i + 2].isalpha() == False) and (string_reply_text[i + 2].isnumeric() == False):
                replies_list.append(reply_text)
                reply_text = ""
            else:
                reply_text += string_reply_text[i]
        else:
            if string_reply_text[i] != "[":
                reply_text += string_reply_text[i]

    replies.append(replies_list)



replies_vader_compound = []

for set_of_replies in replies:

    set_of_replies_vader_compound = []

    for reply in set_of_replies:
        compound_score = analyzer.polarity_scores(reply)["compound"]
        set_of_replies_vader_compound.append(compound_score)
    
    replies_vader_compound.append(set_of_replies_vader_compound)

data["replies_vader_compound"] = replies_vader_compound



replies_vader_compound_average = []

for scores in replies_vader_compound:
    if len(scores) > 0:
        replies_vader_compound_average.append(sum(scores) / len(scores))
    elif len(scores) == 0:
        replies_vader_compound_average.append(-2)

data["replies_vader_compound_average"] = replies_vader_compound_average


replies_vader_compound_average_for_graph = []

for reply in replies_vader_compound_average:
    if reply != -2 and reply != []:
        replies_vader_compound_average_for_graph.append(reply)

dates = []
average_compound = []

for i in range(len(data["replies_vader_compound_average"])):
    if data["replies_vader_compound_average"][i] != -2:
        dates.append(data["timestamp"][i][5:10])
        average_compound.append(data["replies_vader_compound_average"][i])

dates_ints = []

for i in range(len(dates)):
    dates_ints.append(float(dates[i][0:2] + "." + dates[i][4:]))

# OUTCOME 1: Average Sentiment Score per Reply Over Time.

#matplot.plot(dates_ints, average_compound)    
#matplot.show()

data_raw = {}

for i in range(len(dates_ints)):
    if dates_ints[i] not in data_raw.keys():
        data_raw[dates_ints[i]] = []
        data_raw[dates_ints[i]].append(average_compound[i])
    elif dates_ints[i] in data_raw.keys():
        data_raw[dates_ints[i]].append(average_compound[i])


average_per_day_list = []

for values_list in data_raw.values():
    average_per_day = sum(values_list) / len(values_list)
    average_per_day_list.append(average_per_day)

data_polished = {}
keys_list = list(data_raw.keys())

for i in range(len(data_raw.keys())):
    data_polished[keys_list[i]] = average_per_day_list[i]

values_list = list(data_polished.values())

print(keys_list)
print()
print(values_list)

index = range(10)

new_x = [4*i for i in keys_list]

matplot.bar(keys_list, data_polished.values(), width=0.3)
matplot.xlabel("Time (Month)")
matplot.ylabel("Average VADER Compound Sentiment Score per Day")
matplot.title("Magnitude Change in VADER Compound Sentiment Score")
matplot.show()

after_takeover = [] # Takeover Date: 27th of October
before_takeover = []

for i in range(579):
    after_takeover.append(data["replies_vader_compound_average"][i])
    before_takeover.append(data["replies_vader_compound_average"][i + 789])

# OUTCOME 2: Vader Before / After Takeover

print()
print("Average Compound Before Takeover: " + str(sum(before_takeover) / len(before_takeover)))
print("Standard Deviation Before Takeover:" + str(statistics.stdev(before_takeover)))

print()

print("Average Compound After Takeover: " + str(sum(after_takeover) / len(after_takeover)))
print("Standard Deviation After Takeover:" + str(statistics.stdev(after_takeover)))

print()

# OUTCOME 3: Regression

print(linregress(keys_list, values_list))
print()
print(linregress(dates_ints,average_compound ))

print()

# OUTCOME 4: Scatter

matplot.scatter(keys_list, values_list)
matplot.xlabel("Time (Month)")
matplot.ylabel("Average VADER Compound Sentiment Score per Day")
matplot.title("Time Relationship VADER Compound Sentiment Score")
matplot.show()
