##########################################################
# HarvardX (PH125.9x) - Capstone Project - Konstantinos Liakopoulos
##########################################################


if (!require("rpart")) install.packages("pacman")
if (!require("corrplot")) install.packages("corrplot")
if (!require("dplyr")) install.packages("dplyr")
if (!require("caret")) install.packages("caret", dependencies = TRUE)
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("rpart.plot")) install.packages("rpart.plot")
if (!require("corrplot")) install.packages("corrplot")
if (!require("pROC")) install.packages("pROC")
if (!require("tidyr")) install.packages("tidyr")
if (!require("e1071")) install.packages('e1071', dependencies=TRUE)
if (!require("randomForest")) install.packages("randomForest")


library(dplyr)
library(rpart)
library(caret)
library(rpart.plot)
library(ggplot2)
library(corrplot)
library(pROC)
library(tidyr)
library(e1071)
library(randomForest)

# Import libraries and seed

set.seed(1, sample.kind="Rounding")


# Read  data. check if data are missing (NA). Examine data.

wine<-read.csv("winequality-red.csv")


missing<- wine %>% 
  is.na() %>% 
  colSums()
data.frame(missing)


str(wine)


summary(wine$quality)
summary(wine)

table(wine$quality)
prop.table(table(wine$quality))

hist(wine$quality)

# ggplot(data.frame(wine), aes(x=quality)) +geom_bar()+
#   scale_x_continuous(limits = c(2, max(wine$quality)+1), breaks = round(seq(3,8)))+
#   xlab("Quality") +
#   ylab("Count") +
#   ggtitle("Wine quality")

mean(wine$quality)
sd(wine$quality)

quantile(as.integer(wine$quality), .95)


corrplot(cor(wine), method="number")

wine%>%ggplot(aes(factor(quality), alcohol, group=quality))+geom_boxplot()+ggtitle("Alcohol & Quality")


# A factor is created named status with two values good or bad based on wine quality.
# This is the output value that will be predicted.


wine$status = factor(ifelse(wine$quality >=  7, "good", "bad"))
wine<-mutate(wine%>%dplyr::select(-quality))

table(wine$status)
prop.table(table(wine$status))


# Create train andtest dataset..

test_index <- createDataPartition(wine$status, times = 1, p = 0.75, list = FALSE)

wine_train<-wine[test_index ,]

wine_test<-wine[-test_index ,]

wine_train%>%
  gather(-status, key = "var", value = "value") %>% 
  ggplot(aes(x = status, y = value, color = status)) +
  geom_boxplot() +
  facet_wrap(~ var, scales = "free", ncol = 3)+
  theme(legend.position="none")


# Regression Tree Model

rpart_model<-train(status~.,data=wine_train, method = "rpart")
rpart_model

varImp(rpart_model)

plot(varImp(rpart_model))

rpart_predictor<-predict(rpart_model, wine_test)

plot(rpart_model$finalModel, margin=0.1)
text(rpart_model$finalModel, cex=0.75)
confusionMatrix(rpart_predictor,wine_test$status)

ggplot(wine, aes(alcohol, sulphates, color = status)) + geom_point()

# Tune regression tree

tunegrider = data.frame(cp=seq(0,0.04, len=200))
train_control <- trainControl(method="repeatedcv", number=10, repeats=3)
rpart_model<-train(status~.,data=wine_train, method = "rpart",
                   tuneGrid=tunegrider,trControl=train_control)

ggplot(rpart_model,highlight = TRUE)


plot(rpart_model$finalModel, margin=0.05)
text(rpart_model$finalModel, cex=0.55)

rpart_predictor<-predict(rpart_model, wine_test)

rpart_model$bestTune

rpart_model$finalModel

confusionMatrix(rpart_predictor,wine_test$status)

# K-Nearest Neighbors Model

train_control <- trainControl(method="repeatedcv", number=10, repeats=3)
knn_model<-train(status~.,data=wine_train, method = "knn",trControl=train_control,
                 tuneGrid=data.frame(k=seq(1,10,0.5)))
knn_predictor<-predict(knn_model, wine_test)

knn_model$bestTune

knn_model$finalMode

ggplot(knn_model, highlight = TRUE)


confusionMatrix(knn_predictor,wine_test$status)

# Nornalize data for K-NN model

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

wine_n <- as.data.frame(lapply(wine[1:11], normalize))


wine_n$status<-wine$status


summary(wine_n)

# Tune K-nn model

test_index_n <- createDataPartition(wine$status, times = 1, p = 0.75, list = FALSE)

wine_train_n <-wine[test_index_n  ,]

wine_test_n <-wine[-test_index_n  ,]


train_control <- trainControl(method="repeatedcv", number=10, repeats=3)
knn_model<-train(status~.,data=wine_train_n, method = "knn",
                 trControl=train_control, tuneGrid=data.frame(k=seq(1,10,0.5)))
knn_predictor<-predict(knn_model, wine_test_n)

knn_model$bestTune

knn_model$finalMode

ggplot(knn_model, highlight = TRUE)

confusionMatrix(knn_predictor,wine_test_n$status)

# Random Forest Model


rfTunegrider = data.frame(mtry=seq(2,6, 0.5))
train_control <- trainControl(method="repeatedcv", number=10, repeats=3)
random_model<-train(status~.,data=wine_train, method = "rf",
                    metric="Kappa", tuneGrid=rfTunegrider,trControl=train_control,ntree = 1000)

random_predictor<-predict(random_model, wine_test)
result<-confusionMatrix(random_predictor,wine_test$status)

result

random_model$bestTune

random_model$finalModel

ggplot(random_model,highlight = TRUE)

# Evaluate Model, F-FSCORE, ROC, AUC

result$byClass["F1"]

evalResult.rf <- predict(random_model, wine_test, type = "prob")


random_roc<-roc(wine_test$status,evalResult.rf[,2])

plot(random_roc)

auc(random_roc)