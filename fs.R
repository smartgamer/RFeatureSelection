#FEATURE SELECTION TECHNIQUES WITH R

#http://dataaspirant.com/2018/01/15/feature-selection-techniques-r/

# Once we have enough data, We won’t feed entire data into the model and expect great results. We need to pre-process the data.

# In fact, the challenging and the key part of machine learning processes is data preprocessing.
# 
# Below are the key things we indented to do in data preprocessing stage.
# 
# Feature transformation
# Feature selection

# Feature transformation is to transform the already existed features into other forms. Suppose using the logarithmic function to convert normal features to logarithmic features.
# 
# Feature selection is to select the best features out of already existed features. 
# ##
# Most models have a method to generate variable importance which indicates what features are used in the model and how important they are. Variable importance also has a use in the feature selection process.
##

## content ##
# Why modeling is not the final step
# The role of correlation
# Calculating feature importance with  regression methods
# Using caret package to calculate feature importance
# Random forest for calculating feature importance
# Conclusion
##

## 1. Why Modeling is Not The Final Step ##
# Like a coin, every project has two sides.
# 
# Business side
# Technical side
# The technical side deals with data collection, processing and then implementing it to get results. The business side is what envelops the technical side.

#Most models have a method to generate variable importance which indicates what features are used in the model and how important they are. Variable importance also has a use in the feature selection process.

# As the Occam’s Razor principle states.
# The simplest models are the best.

# Finding the best features to use in the model based on decreasing variable importance helps one to identify and select the features which produce 80% of the results and discard the rest of the variables which account for rest 20% of the accuracy.

##


## 2. Role of Correlation ##

##If you are working with a model which assumes the linear relationship between the dependent variables, correlation can help you come up with an initial list of importance. It also works as a rough list for nonlinear models.###
# Use the library cluster generation to make a positive definite matrix of 15 features
# install.packages("clusterGeneration")
library(clusterGeneration)

S = genPositiveDefMat("unifcorrmat",dim=15)

# create 15 features using multivariate normal distribution for 5000 datapoints
# install.packages("mnormt")
library(mnormt)

n = 5000

X = rmnorm(n,varcov=S$Sigma)

# Create a two class dependent variable using binomial distribution

Y = rbinom(n,size=1,prob=0.3)
data = data.frame(Y,X)

# Create a correlation table for Y versus all features
cor(data,data$Y)

#As expected, since we are using a randomly generated dataset, there is little correlation of Y with all other features. These numbers may be different for different runs.
# In this case, the correlation for X11 seems to be the highest. Had we to necessarily use this data for modeling, X11 will be expected to have the maximum impact on predicting Y. In this way, the list of correlations with the dependent variable will be useful to get an idea of the features that impact the outcome.
# While plotting correlations, we always assume that the features and dependent variable are numeric. If we are looking at Y as a class, we can also see the distribution of different features for every class of Y.
##

## 3. Using Regression to Calculate Variable Importance  ##
# The summary function in regression also describes features and how they affect the dependent feature through significance. It works on variance and marks all features which are significantly important.
# 
# Such features usually have a p-value less than 0.05 which indicates that confidence in their significance is more than 95%.
# Using the mlbench library to load diabetes data

library(mlbench)
data(PimaIndiansDiabetes)
data_lm = as.data.frame(PimaIndiansDiabetes)

# Fit a logistic regression model
fit_glm = glm(diabetes~.,data_lm,family = "binomial")

# generate summary

summary(fit_glm)

#The output by logistic model gives us the estimates and probability values for each of the features. It also marks the important features with stars based on p-values.

# For features whose class is a factor, the features are broken on the basis of each unique factor level. We see that the most important variables include glucose, mass and pregnant features for diabetes prediction. In this manner, regression models provide us with a list of important features.

## 4.Using The Caret Package to perform  variable importance ##
# R has a caret package which includes the varImp() function to calculate important features of almost all models.
# Let’s compare our previous model summary with the output of the varImp() function.

# Using variable importance functionR

# Using varImp() function
library(caret)
varImp(fit_glm)
# The varImp output ranks glucose to be the most important feature followed by mass and pregnant. This is exactly similar to the p-values of the logistic regression model.
# However, varImp() function also works with other models such as random forests and can also give an idea of the relative importance using the importance score it generates.

## 5. Variable Importance Through Random Forest  ##
# Random forests are based on decision trees and use bagging to come up with a model over the data. Random forests also have a feature importance methodology which uses ‘gini index’ to assign a score and rank the features.
# 
# Let us see an example and compare it with varImp() function.
# 
# Using Random forest for feature importanceR

# Import the random forest library and fit a model

library(randomForest)
fit_rf = randomForest(diabetes~., data=data_lm)

# Create an importance based on mean decreasing gini
importance(fit_rf)
varImp(fit_rf)
#We see that the importance scores by varImp() function and the importance() function of random forest are exactly the same. If the model being used is random forest, we also have a function known as varImpPlot() to plot this data

# Create a plot of importance scores by random forest
varImpPlot(fit_rf)
#These scores which are denoted as ‘Mean Decrease Gini’ by the importance measure represents how much each feature contributes to the homogeneity in the data.
##


##Conclusion
# Variable importance is usually followed by variable selection. Whether feature importance is generated before fitting the model (by methods such as correlation scores) or after fitting the model (by methods such as varImp() or Gini Importance), the important features not only give an insight on the features with high weightage and used frequently by the model but also the features which are slowing down our model.
# 
# This is why feature selection is used as it can improve the performance of the model. This is by removing predictors with chance or negative influence and provide faster and more cost-effective implementations by the decrease in the number of features going into the model.
# 
# To decide on the number of features to choose, one should come up with a number such that neither too few nor too many features are being used in the model.
# 
# For a methodology such as using correlation, features whose correlation is not significant and just by chance (say within the range of +/- 0.1 for a particular problem) can be removed.
# 
# For other methods such as scores by the varImp() function or importance() function of random forests, one should choose the features until which there is a sharp decline in importance scores.
# 
# In case of a large number of features (say hundreds or thousands), a more simplistic approach can be a cutoff score such as only the top 20 or top 25 features or the features such as the combined importance score crosses a threshold of 80% or 90% of the total importance score.
# 
# In the end, variable selection is a trade-off between the loss in complexity against the gain in execution speed that the project owners are comfortable with.
# 
# The methods mentioned in this article are meant to provide an overview of the ways in which variable importance can be calculated for a data. There can be other similar variable importance methods with their uses and implementations as per the situation.
##

###
# Complete Code ##
# Use the library cluster generation to make a positive definite matrix of 15 features

library(clusterGeneration)
S = genPositiveDefMat("unifcorrmat",dim=15)
#create 15 features using multivariate normal distribution for 5000 datapoints
library(mnormt)

n = 5000
X = rmnorm(n,varcov=S$Sigma)

# Create a two class dependent variable using binomial distribution
Y = rbinom(n,size=1,prob=0.3)

data = data.frame(Y,X)
# Create a correlation table for Y versus all features

cor(data,data$Y)
# Using the mlbench library to load diabetes data
library(mlbench)
data(PimaIndiansDiabetes)
data_lm=as.data.frame(PimaIndiansDiabetes)
# Fit a logistic regression model
fit_glm=glm(diabetes~.,data_lm,family = "binomial")

# generate summary
summary(fit_glm)
# Using varImp() function
library(caret)
varImp(fit_glm)

#Import the random forest library and fit a model
library(randomForest)
fit_rf=randomForest(diabetes~., data=data_lm)
# Create an importance based on mean decreasing gini
importance(fit_rf)

# compare the feature importance with varImp() function
varImp(fit_rf)

# Create a plot of importance scores by random forest
varImpPlot(fit_rf)

##


##


##


##


##


##


##


##


##








































