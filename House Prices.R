##################### H2O and data.table ##############################
#Clear R environment
rm(list = ls())
#Setting working directory
setwd("D:/Data Science/R CODE/House Prices Advanced Regression Techniques")

#install and load the package
x <- c("xlsx", "ggplot2", "plyr", "data.table",
       "DT", "gmodels", "dummies", "h2o", "DMwR", "caret")
install.packages(x) #Installs all the packages
sapply(x, library, character.only = TRUE)

#load data using fread
train <- fread("train.csv", stringsAsFactors = TRUE)
test <- fread("test.csv", stringsAsFactors = TRUE)

#combine data set
test[,SalePrice := mean(train$SalePrice)]
combin <- rbindlist(list(train,test))

#Analysing dataset
dim(combin) #We have 2919 obs and 81 col.
lapply(combin,class)
sapply(combin,summary)

#missing values
combin[,colSums(is.na(combin))]
colSums(is.na(test))

#Dropping variables with too many NAs
combin <- combin[,c("FireplaceQu", "PoolQC", "Fence", "MiscFeature", "Alley") := NULL]

#Imputing column values with less NAs(Using KNN imputation)
combin <- knnImputation(combin, k = 3)

#Analysing categorical variables
tokeep <- which(sapply(combin,is.factor))
combin.factor <- combin[, tokeep, with = FALSE]
summary(combin.factor)

#one hot coding for categorical variables 
dummy.var <- dummyVars(~., data = combin)
combin.dummy <- data.table(predict(dummy.var, newdata = combin))

#Divide into train and test
c.train <- combin.dummy[1:nrow(train),]
c.test <- combin.dummy[-(1:nrow(train)),]

#launch the H2O cluster
localh2o <- h2o.init(nthreads = -1)

#data to h2o cluster
train.h2o <- as.h2o(c.train)
test.h2o <- as.h2o(c.test)

#check column index number
colnames(train.h2o)

#dependent variable (Purchase)
y.dep <- 272

#independent variables (dropping ID variables)
x.indep <- c(1:271)

#Multiple Regression in H2O
regression.model <- h2o.glm( y = y.dep, x = x.indep,
                             training_frame = train.h2o,
                             family = "gaussian")

h2o.performance(regression.model)
h2o.r2(regression.model)

#make predictions
predict.reg <- as.data.frame(h2o.predict(regression.model, test.h2o))
sub_reg <- data.frame(Id = test$Id, SalePrice = predict.reg$predict)
fwrite(sub_reg, file = "sub_reg.csv", row.names = F)

#Random Forest
system.time(
  rforest.model <- h2o.randomForest(y=y.dep, x=x.indep, 
                                    training_frame = train.h2o,
                                    ntrees = 1500, mtries = 20,
                                    max_depth = 10, seed = 1122))

h2o.performance(rforest.model)
h2o.r2(rforest.model)

#check variable importance
h2o.varimp(rforest.model)

#making predictions on unseen data
system.time(
  predict.rforest <- as.data.frame(h2o.predict(rforest.model, test.h2o))
)


#writing submission file
sub_rf <- data.frame(Id = test$Id, SalePrice = predict.rforest$predict)
fwrite(sub_rf, file = "sub_rf.csv", row.names = F)

#GBM
system.time(
  gbm.model <- h2o.gbm(y=y.dep, x=x.indep,
                       training_frame = train.h2o,
                       ntrees = 1000, max_depth = 4,
                       learn_rate = 0.01, seed = 1122)
)

h2o.performance(gbm.model)
h2o.r2(gbm.model)

#making prediction and writing submission file
predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))
sub_gbm <- data.frame(Id = test$Id, SalePrice = predict.gbm$predict)
fwrite(sub_gbm, file = "sub_gbm.csv", row.names = F)

#deep learning models
system.time(
  dlearning.model <- h2o.deeplearning(y = y.dep,
                                      x = x.indep,
                                      training_frame = train.h2o,
                                      epoch = 60,
                                      hidden = c(100,100),
                                      activation = "Rectifier",
                                      seed = 1122
  )
)

h2o.performance(dlearning.model)
h2o.r2(dlearning.model)

#making predictions
predict.dl2 <- as.data.frame(h2o.predict(dlearning.model, test.h2o))

#create a data frame and writing submission file
sub_dlearning <- data.frame(Id = test$Id, SalePrice = predict.dl2$predict)
write.csv(sub_dlearning, file = "sub_dlearning_new.csv", row.names = F)

#Shutting down H2O
h2o.shutdown()