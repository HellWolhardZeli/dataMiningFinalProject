
### LIBRARY CALL ###
library("rpart")
library("rpart.plot")	
rm(corpus)
#######################################################################################################

### READING DATA. YOU HAVE TO CHANGE THIS TO WHERE YOUR DATA IS ###

#call the visualizatoin libraries
library(ggplot2)
library(GGally)
library(corrplot)
#this library is for CrossTable Function
library(gmodels)
library(lattice)
library(caret)
library(tidyverse)
library(pROC)
library(formattable)
library(kableExtra)

loans_EDA <- read.csv("kiva_loans.csv")
  x<-table(loans_EDA$activity)
  x<-data.frame(head(sort(x, decreasing = TRUE),10))
  names(x)[1]<-"Activity"
  names(x)[2]<-"Loans"
  
  x %>%
    mutate(
      Loans = color_bar("lightgreen")(Loans)
    ) %>%
    kable("html", escape = F) %>%
    kable_styling("hover", full_width = F, font_size = 16)%>%
    column_spec(2,width = "50cm")
  
  #sector
  x<-table(loans_EDA$sector)
  x<-data.frame(head(sort(x, decreasing = TRUE),10))
  names(x)[1]<-"Sector"
  names(x)[2]<-"Loans"
  
  x %>%
    mutate(
      Loans = color_bar("lightgreen")(Loans)
    ) %>%
    kable("html", escape = F) %>%
    kable_styling("hover", full_width = F, font_size = 16)%>%
    column_spec(2,width = "50cm")
  
  #Country
  x<-table(loans_EDA$country)
  x<-data.frame(head(sort(x, decreasing = TRUE),10))
  x<-data.frame(x)
  names(x)[1]<-"Countries"
  names(x)[2]<-"Loans"
  x %>%
    mutate(
      Loans = color_bar("lightgreen")(Loans)
    ) %>%
    kable("html", escape = F) %>%
    kable_styling("hover", full_width = F, font_size = 16)%>%
    column_spec(2,width = "50cm")
 
  #######################################################################################################
  
  #######################################################################################################
  
  loan <- read.csv("Kiva.Final.DB_BEFORE.SMOTE.csv")
  loan <- Kiva.Final.DB_after.SMOTE_latest
  ###################################### Spliting data in training and testing set############################
  n <- nrow(loan)
  shuffled_df <- loan[sample(n), ]
  train_indices <- 1:round(0.8 * n)
  train_dataset <- shuffled_df[train_indices, ]
  test_indices <- (round(0.8 * n) + 1):n
  test_dataset <- shuffled_df[test_indices, ]
  train_dataset=train_dataset[complete.cases(train_dataset),]
  dim(train_dataset)
  
 
  ########################### BEFORE SMOTE ############################################################ 
  ###################################### Decision tree ############################
  
  set.seed(100)
  
  cvtree <- train(funded_bracket~., data=loan[,], 
                  method = 'ctree', 
                  trControl=trainControl(method = 'cv', number=5),
                  tuneGrid=expand.grid(mincriterion=0.95))
  
  cvtree
  
  plot(cvtree$finalModel, type="simple") # Easier to see
  plot(cvtree$finalModel) #with labels on terminal node
  
  
  varImp(cvtree)
  
  pred <-predict(cvtree, newdata=test_dataset)
  pred.int <- round(pred)
  metrics <- confusionMatrix(as.factor(pred.int), as.factor(test_dataset$funded_bracket))
  
  print(metrics)
  ########### ROC #############
  
  pred <-predict(cvtree, newdata=test_dataset)
  pred.int <- round(pred)
  
  roc(pred.int, test_dataset$funded_bracket , plot=TRUE)
  
  
  ###################################### N. Bayes ############################
  
  naive.bayes.model <- train(as.factor(funded_bracket)~., 
                             data = train_dataset, 
                             trControl = trainControl(method = "cv", number = 5),
                             method = "nb")
  
  
  # Important features
  naive.bayes.imp <- varImp(naive.bayes.model)
  print(naive.bayes.imp)
  plot(naive.bayes.imp, top = 20, main = "Naive Bayes Important features")
  
  
  # prediction on test data 
  naive.bayes.pred <- predict(naive.bayes.model, test_dataset,)
  
  
  # Error metrics
  naive.bayes.metrics <- confusionMatrix(as.factor(naive.bayes.pred), as.factor(test_dataset$funded_bracket))
  print(naive.bayes.metrics)
  
  pred = naive.bayes.pred;
  # ROC 
  roc(pred.int, as.integer(test_dataset$funded_bracket) , plot=TRUE,levels=base::levels(as.factor(pred)))
  
  #End
  
  ###################################### SVM #############################
  
  svm.model <- train(funded_bracket ~., 
                     data = train_dataset, 
                     method = "svmRadial",
                     preProc = c("center", "scale"),
                     trControl = trainControl(method = "cv", number = 3),
                     tuneLength = 8)
  
  # Important variables according to SVM model
  svm.imp <- varImp(svm.model)
  print(svm.imp)
  plot(svm.imp, top = 20, main = "SVM Important features")
  
  # prediction on test data 
  svm.pred <- predict(svm.model, test_dataset)
  
  
  # error metrics
  svm.metrics <- confusionMatrix(as.factor(as.integer(svm.pred)), as.factor(test_dataset$funded_bracket))
  print(svm.metrics)
  
  
  #ROC
  roc(as.integer(svm.pred), as.integer(test_dataset$funded_bracket) , plot=TRUE)
  
  #End
  
  
  ################################################# Logistic ##########################################################3  
  set.seed(100)
  glm.model <- train(as.factor(funded_bracket) ~., 
                     data = train_dataset, 
                     method = "glm", family = binomial(link = "logit"),
                     trControl = trainControl(method = "cv", number = 5))
  
  #Important Features#
  glm.imp <- varImp(glm.model)
  print(glm.imp)
  plot(glm.imp, top = 20, main = "Logistic Regression Important features")
  
  #Prediction on test data#
  glm.pred <- predict(glm.model, test_dataset, type = "raw")
  
  #Error metrics#
  test_dataset$funded_bracket=as.factor(test_dataset$funded_bracket)
  glm.metrics <- confusionMatrix(glm.pred, test_dataset$funded_bracket)
  print(glm.metrics)
  
  #ROC#
  roc(as.numeric(glm.pred), as.numeric(test_dataset$funded_bracket , plot=TRUE))
  
  
  ######################################### AFTER SMOTE ###########################################################

  loan <- read.csv("Kiva.Final.DB_After.SMOTE.csv")
  loan <- Kiva.Final.DB_after.SMOTE_latest
  ###################################### Spliting data in training and testing set############################
  n <- nrow(loan)
  shuffled_df <- loan[sample(n), ]
  train_indices <- 1:round(0.8 * n)
  train_dataset <- shuffled_df[train_indices, ]
  test_indices <- (round(0.8 * n) + 1):n
  test_dataset <- shuffled_df[test_indices, ]
  train_dataset=train_dataset[complete.cases(train_dataset),]
  dim(train_dataset)
  
  
  ###################################### Decision tree ############################
  
  set.seed(100)
  
  cvtree <- train(funded_bracket~., data=loan[,], 
                  method = 'ctree', 
                  trControl=trainControl(method = 'cv', number=5),
                  tuneGrid=expand.grid(mincriterion=0.95))
  
  cvtree
  
  plot(cvtree$finalModel, type="simple") # Easier to see
  plot(cvtree$finalModel) #with labels on terminal node
  
  
  varImp(cvtree)
  
  pred <-predict(cvtree, newdata=test_dataset)
  pred.int <- round(pred)
  metrics <- confusionMatrix(as.factor(pred.int), as.factor(test_dataset$funded_bracket))
  
  print(metrics)
  ########### ROC #############
  
  pred <-predict(cvtree, newdata=test_dataset)
  pred.int <- round(pred)

  roc(pred.int, test_dataset$funded_bracket , plot=TRUE)

  
   ###################################### N. Bayes ############################
  
  naive.bayes.model <- train(as.factor(funded_bracket)~., 
                             data = train_dataset, 
                             trControl = trainControl(method = "cv", number = 5),
                             method = "nb")
  
  
  # Important features
  naive.bayes.imp <- varImp(naive.bayes.model)
  print(naive.bayes.imp)
  plot(naive.bayes.imp, top = 20, main = "Naive Bayes Important features")
  
  
  # prediction on test data 
  naive.bayes.pred <- predict(naive.bayes.model, test_dataset,)
  
  
  # Error metrics
  naive.bayes.metrics <- confusionMatrix(as.factor(naive.bayes.pred), as.factor(test_dataset$funded_bracket))
  print(naive.bayes.metrics)
  
  pred = naive.bayes.pred;
  # ROC 
  roc(pred.int, as.integer(test_dataset$funded_bracket) , plot=TRUE,levels=base::levels(as.factor(pred)))
  
  #End
  
  ###################################### SVM #############################

  svm.model <- train(funded_bracket ~., 
                     data = train_dataset, 
                     method = "svmRadial",
                     preProc = c("center", "scale"),
                     trControl = trainControl(method = "cv", number = 3),
                     tuneLength = 8)
  
  # Important variables according to SVM model
  svm.imp <- varImp(svm.model)
  print(svm.imp)
  plot(svm.imp, top = 20, main = "SVM Important features")
  
  # prediction on test data 
  svm.pred <- predict(svm.model, test_dataset)
  
  
  # error metrics
  svm.metrics <- confusionMatrix(as.factor(as.integer(svm.pred)), as.factor(test_dataset$funded_bracket))
  print(svm.metrics)
  
  
  #ROC
  roc(as.integer(svm.pred), as.integer(test_dataset$funded_bracket) , plot=TRUE)
  
  #End
  

  ################################################# Logistic ##########################################################3  
  set.seed(100)
  glm.model <- train(as.factor(funded_bracket) ~., 
                     data = train_dataset, 
                     method = "glm", family = binomial(link = "logit"),
                     trControl = trainControl(method = "cv", number = 5))
  
  #Important Features#
  glm.imp <- varImp(glm.model)
  print(glm.imp)
  plot(glm.imp, top = 20, main = "Logistic Regression Important features")
  
  #Prediction on test data#
  glm.pred <- predict(glm.model, test_dataset, type = "raw")
  
  #Error metrics#
  test_dataset$funded_bracket=as.factor(test_dataset$funded_bracket)
  glm.metrics <- confusionMatrix(glm.pred, test_dataset$funded_bracket)
  print(glm.metrics)
  
  #ROC#
  roc(as.numeric(glm.pred), as.numeric(test_dataset$funded_bracket , plot=TRUE))
