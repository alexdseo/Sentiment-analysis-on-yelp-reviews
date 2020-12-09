import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.neural_network import MLPClassifier
train = pd.read_csv("train_Madison.csv")
#train = train[:20]
test = pd.read_csv("test_Madison.csv")
names = train["name"]
stars = train["star"]

    
ratingTotal = {}
ratingCount = {}
for i in range(0, len(names)):
    name = names[i]
    star = stars[i]
    if name not in ratingTotal:
        ratingTotal[name] = 0
        ratingCount[name] = 0
    ratingTotal[name] += star
    ratingCount[name]+=1
print(len(ratingTotal))
print(len(ratingCount))
ratingAverage = dict((k, float(ratingTotal[k])/ratingCount[k]) for k in ratingCount)


positiveCount = []
negativeCount = []
positiveOverall = []
ratingArray = []
nwords = []
posRatio = []
negRatio = []
numReviews = []
positiveWords = ["gem", "incredible", "amazing", "favorites", "wonderful", "perfect", "fantastic", "notch", "favorite", "awesome", "outstanding", "yum", "delicious", "excellent", "perfectly", "loved", "savory", "cozy", "unique", "yummy", "glad", "homemade", "best", "love", "lovely", "friendly", "reasonable", "beautiful", "classic", "generous", "comfortable", "authentic", "flavorful", "enjoyed", "happy", "tasty", "enjoy", "truly", "fancy", "wow", "must", "ambience", "nicely", ]
negativeWords = ["disappointing", "dirty", "mediocre", "poor", "terrible", "awful", "rude", "horrible", "worst", "overpriced", "sorry", "waited", "bland", "unfortunately", "sad", "bad", "loud", "overly", "greasy", "frozen", "dry", "empty"]
for i in range(0, len(names)):
    print(i)
    print(len(names))
    posC = 0
    negC = 0
    for word in positiveWords:
        wordCount = train[word][i]
        posC+=wordCount
    for negWord in negativeWords:
        negWordCount = train[negWord][i]
        negC+=negWordCount
    nword = np.array(train["nword"])[i]
    nwords.append(nword)
    posRatio.append(posC/nword)
    negRatio.append(negC/nword)
    positiveCount.append(posC)
    negativeCount.append(negC)
    if posC>negC:
        positiveOverall.append(1)
    elif posC<negC:
        positiveOverall.append(-1)
    else:
        positiveOverall.append(0)
    ratingArray.append(ratingAverage[names[i]])
    #numReviews.append(ratingCount[names[i]])
#predictors = np.array([positiveCount, negativeCount, ratingArray])

#predictors = np.array([nwords, positiveOverall, ratingArray, positiveCount, negativeCount, posRatio])
posRatio = np.array(posRatio)
print(posRatio)
predictors = np.array([nwords, positiveOverall, ratingArray, positiveCount, negativeCount, posRatio, negRatio])
predictors = predictors.T
reg = linear_model.LinearRegression()
reg.fit(predictors, train["star"])

# smallRatingAverage = dict((k, float(ratingTotal[k])/ratingCount[k]) for k in ratingCount if ratingCount[k]<3)
# smallTotal = 0
# for k in smallRatingAverage:
#     smallTotal+=float(smallRatingAverage[k])
# print("Small Total: ", len(smallRatingAverage))
# print("Small Average: ", smallTotal/len(smallRatingAverage))

# largeRatingAverage = dict((k, float(ratingTotal[k])/ratingCount[k]) for k in ratingCount if ratingCount[k]>=50)
# largeTotal = 0
# for k in largeRatingAverage:
#     largeTotal+=float(largeRatingAverage[k])
# print("Large Total: ", len(largeRatingAverage))    
# print("Large Average: ", largeTotal/len(largeRatingAverage))

testNames = test["name"]
testIDs = test["Id"]
testExpected = []
for i in range (0, len(testNames)):
    name = testNames[i]
    posC = 0
    negC = 0
    posOverall = 0
    if name in ratingAverage:
        rating = ratingAverage[name]
    else:
        rating = 3.758
        
    for word in positiveWords:
        wordCount = test[word][i]
        posC+=wordCount
    for word in negativeWords:
        wordCount = test[word][i]
        negC+=wordCount 
    if posC > negC:
        posOverall = 1
    elif posC < negC:
        posOverall = -1
    else:
        posOverall = 0
    
    nword = np.array(test["nword"])[i]
    #numReviews = ratingCount[name]
    posRatio = posC/nword
    negRatio = negC/nword
    #predictors = np.array([[posC, negC, rating]])
    predictors = np.array([[nword, posOverall, rating, posC, negC, posRatio, negRatio]])
    prediction = reg.predict(predictors)
    testExpected.append(prediction[0])
    print(prediction)
submitDF = pd.DataFrame({'Id':testIDs, 'Expected':testExpected})
submitDF.to_csv("submission.csv", sep=',', index=False)
    