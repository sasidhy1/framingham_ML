{
library(MASS)
library(leaps)
library(glmnet)
library(car)
library(dplyr)
library(tidyverse)
library(broom)
library(caret)
library(fastDummies)
} 

#!# enter in your own directory & file name
setwd("C:/Users/Matt/Desktop/bootcamp final project")
data = read.csv("reduced.csv", head = T)
#View(data) 

{
  sex = data$SEX
  age = data$AGE
  cigs = data$CIGPDAY
  edu = data$EDUC
  chol = data$TOTCHOL
  bmi = data$BMI
  glu = data$GLUCOSE
  hr = data$HEARTRTE
  bp = data$SYSBP
  meds = data$BPMEDS

  dia = data$DIABETES
  stroke = data$STROKE
  ang = data$ANGINA
  cvd = data$CVD
}

is.factor(sex)
sex = as.factor(sex)
#is.factor(edu)
#edu = as.factor(edu)
#is.factor(meds)
#meds = as.factor(meds)
#is.factor(meds)
#meds = as.factor(meds)

continuous_vars = data.frame(age,cigs,chol,bmi,glu,hr,bp)
cor(continuous_vars, method = c("pearson"))

accuracy_checker = function(MODEL,DATA,RESPONSE) {
  predictedval = predict(MODEL,newdata=DATA,type='response')
  fitted.results.cat = ifelse(predictedval > 0.5,"1","0")
  fitted.results.cat = as.numeric(fitted.results.cat)
  compare = data.frame("actual"=RESPONSE,"predicted"=fitted.results.cat)
  count=0
  for (i in 1:length(compare$actual)){
    if (compare$actual[i] == compare$predicted[i]) count=count+1
  }
  print(count/nrow(compare))
}

############
# diabetes #
############

df = dummy_cols(data.frame(sex,age,cigs,chol,bmi,glu,hr,bp,dia), remove_first_dummy = TRUE)
df = df[,-c(1)] 

Train <- createDataPartition(df$dia, p=0.7, list=FALSE)
training <- df[ Train, ]
testing <- df[ -Train, ]

ocvl = cv.glmnet(data.matrix(training[,-c(8)]),training[,8],alpha=1,nfolds = 300)
plot(ocvl)
min = log(ocvl$lambda.min)
elastic = glmnet(data.matrix(training[,-c(8)]),training[,8],lambda=c(exp(c(min))),alpha=1)
betas = elastic$beta;betas

predictors = training[,-c(8)]
data_reduced = predictors[,betas[,1] != 0]
model = glm(training[,8] ~ .,data = data_reduced, family = binomial(link = "logit"))
summary(model)

step = stepAIC(model,trace=FALSE);step
summary(step)
step_model = step$coefficients; step_model

vif(step)
#plot(step, which = 4, id.n = 3)

accuracy_checker(step,training,training$dia)
accuracy_checker(step,testing,testing$dia)

##########
# stroke #
##########

df = dummy_cols(data.frame(sex,age,cigs,chol,bmi,glu,hr,bp,stroke), remove_first_dummy = TRUE)
df = df[,-c(1)] 

Train <- createDataPartition(df$stroke, p=0.7, list=FALSE)
training <- df[ Train, ]
testing <- df[ -Train, ]

ocvl = cv.glmnet(data.matrix(training[,-c(8)]),training[,8],alpha=1,nfolds = 300)
plot(ocvl)
min = log(ocvl$lambda.min)
elastic = glmnet(data.matrix(training[,-c(8)]),training[,8],lambda=c(exp(c(min))),alpha=1)
betas = elastic$beta;betas

predictors = training[,-c(8)]
data_reduced = predictors[,betas[,1] != 0]
model = glm(training[,8] ~ .,data = data_reduced, family = binomial(link = "logit"))
summary(model)

step = stepAIC(model,trace=FALSE);step
summary(step)
step_model = step$coefficients; step_model

vif(step)
#plot(step, which = 4, id.n = 3)

accuracy_checker(step,training,training$stroke)
accuracy_checker(step,testing,testing$stroke)

##########
# angina #
##########

df = dummy_cols(data.frame(sex,age,cigs,chol,bmi,glu,hr,bp,ang), remove_first_dummy = TRUE)
df = df[,-c(1)] 

Train <- createDataPartition(df$ang, p=0.7, list=FALSE)
training <- df[ Train, ]
testing <- df[ -Train, ]

ocvl = cv.glmnet(data.matrix(training[,-c(8)]),training[,8],alpha=1,nfolds = 300)
plot(ocvl)
min = log(ocvl$lambda.min)
elastic = glmnet(data.matrix(training[,-c(8)]),training[,8],lambda=c(exp(c(min))),alpha=1)
betas = elastic$beta;betas

predictors = training[,-c(8)]
data_reduced = predictors[,betas[,1] != 0]
model = glm(training[,8] ~ .,data = data_reduced, family = binomial(link = "logit"))
summary(model)

step = stepAIC(model,trace=FALSE);step
summary(step)
step_model = step$coefficients; step_model

vif(step)
#plot(step, which = 4, id.n = 3)

accuracy_checker(step,training,training$ang)
accuracy_checker(step,testing,testing$ang)

#######
# cvd #
#######

df = dummy_cols(data.frame(sex,age,cigs,chol,bmi,glu,hr,bp,cvd), remove_first_dummy = TRUE)
df = df[,-c(1)] 

Train <- createDataPartition(df$cvd, p=0.7, list=FALSE)
training <- df[ Train, ]
testing <- df[ -Train, ]

ocvl = cv.glmnet(data.matrix(training[,-c(8)]),training[,8],alpha=1,nfolds = 200)
plot(ocvl)
min = log(ocvl$lambda.min)
elastic = glmnet(data.matrix(training[,-c(8)]),training[,8],lambda=c(exp(c(min))),alpha=1)
betas = elastic$beta;betas

predictors = training[,-c(8)]
data_reduced = predictors[,betas[,1] != 0]
model = glm(training[,8] ~ .,data = data_reduced, family = binomial(link = "logit"))
summary(model)

step = stepAIC(model,trace=FALSE);step
summary(step)
step_model = step$coefficients; step_model

vif(step)
#plot(step, which = 4, id.n = 3)

accuracy_checker(step,training,training$cvd)
accuracy_checker(step,testing,testing$cvd)
