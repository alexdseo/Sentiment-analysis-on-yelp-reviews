import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.neural_network import MLPClassifier
train = pd.read_csv("train_Madison.csv")
#train= train[:20]
test = pd.read_csv("test_Madison.csv")
names = train["name"]
stars = train["star"]
diffScores = []
#Goes through the training data and calculates the average rating for each restaurant    
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

#Turns the predictor from rating to difference above or below mean rating for the restaurant
for i in range(0, len(names)):
    name = names[i]
    diffScore = stars[i]-ratingAverage[name]
    print(diffScore)
    diffScores.append(diffScore)
    
positiveCount = []
negativeCount = []
positiveOverall = []
ratingArray = []
nwords = []
posRatio = []
negRatio = []
numReviews = []
avgChars = []
nchars = []
positiveWords = ["gem", "incredible", "amazing", "favorites", "wonderful", "perfect", "fantastic", "notch", "favorite", "awesome", "outstanding", "yum", "delicious", "excellent", "perfectly", "loved", "savory", "cozy", "unique", "yummy", "glad", "homemade", "best", "love", "lovely", "friendly", "reasonable", "beautiful", "classic", "generous", "comfortable", "authentic", "flavorful", "enjoyed", "happy", "tasty", "enjoy", "truly", "fancy", "wow", "must", "nicely"]
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
    nchar = np.array(train["nchar"])[i]
    avgChars.append(nchar/nword)
    nchars.append(nchar)
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

posRatio = np.array(posRatio)
print(posRatio)
predictors = np.array([nwords, positiveOverall, ratingArray, positiveCount, negativeCount, posRatio, negRatio, nchars, avgChars])
predictors = predictors.T
reg = linear_model.LinearRegression()
reg.fit(predictors, diffScores)


#WE CALCULATED THE AVERAGE RATING FOR RESTAURANTS WITH FEW RATINGS AND THE AVERAGE FOR RESTAURANTS WITH MANY AND FOUND LITTLE DIFFERENCE
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
    nchar = np.array(test["nchar"])[i]
    avgChar = nchar/nword
    #numReviews = ratingCount[name]
    posRatio = posC/nword
    negRatio = negC/nword
    #predictors = np.array([[posC, negC, rating]])
    predictors = np.array([[nword, posOverall, rating, posC, negC, posRatio, negRatio, nchar, avgChar]])
    prediction = reg.predict(predictors)
    prediction = prediction+rating
    testExpected.append(prediction[0])
    print(prediction)
submitDF = pd.DataFrame({'Id':testIDs, 'Expected':testExpected})
submitDF.to_csv("submission.csv", sep=',', index=False)
print(reg.coef_)
    