#[Coursera]PracticalMachineLearning

## Practical Machine Learning Assignment Writeup

Our goal in this assignment is to build a predictive model to predict the correctness 
of a participant's excercise form using data from accelerometers on the belt, forearm, arm, and dumbell.

##Data

The data for this assignment is provided by Groupware@LES  
Read more on Human Activity Recognition: http://groupware.les.inf.puc-rio.br/har

Training data can be downloaded from the below link:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

Test data can be downloaded from the below link:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

##Analysis in R

Libraries used for this assignment:

```r
library(caret)
library(randomForest)
library(doParallel)
library("foreach")
library("doSNOW")
```

Setting seed for reproducibility

```r
set.seed(1234)
```

We first load the csv files obtained from the links given in the data section and process them.
The below functions are used for loading and processing the data.

The ```ct``` function is used to get the count of NA values of a column and the columns above a threshold value are deleted
(in this case it is 20 - this suits both the training data and the testing data so no need to make 2 seperate functions
or modifiy the existing one to process test data)

```r
ct <- function(x){
	return(length(which(is.na(x))))
}
```

The given data as it is has a lot of redundant columns for our analysis and also contains a lot of NA values.
So, we use the ```process_data``` function to load the data and remove unwanted columns and NA values.

```r
process_data <- function(dir){
	processed_data <- read.csv(dir)
	
	#deleting unwanted columns
	#removing index,user_name,raw_time data fields,cvtd_time,new_window,num_window

	processed_data[1:7] = list(NULL)

	#cleaning data
	#removing cols. with >= 20 NA's
	
	# removing the last column in the names list best it needs no conversion
	# it is 'classe' column in training data and 'problem_id' in testing data (the last column in testing data is removed        #later)
	
	names_list <- names(processed_data)[-length(names(processed_data))]

	for(n in names_list){
		processed_data[[n]] <- as.numeric(as.character(processed_data[[n]]))
		if(ct(processed_data[[n]])>=20){
			processed_data[[n]] <- NULL
		}
	}
	return(processed_data)
}
```
Before processing the dimensions of the training data are:

```r
train_data <- read.csv("pml-training.csv")
dim(train_data)

#[1] 19622 160
```

After processing data:
```r
train_data <- process_data("pml-training.csv")
dim(train_data)

#[1] 19622 52
```

We have reduced the column count to 1/3rd of original data.

Now, We apply the same function on the given training data and split it into testing(25%) and training data(75%)
to check for our model acuracy.

```r
train_data <- process_data("pml-training.csv")
inTrain <- createDataPartition(y = train_data$classe,p = 0.75,list = F)

training <- train_data[inTrain,]
testing <- train_data[-inTrain,]

dim(testing)
#[1] 4904   53

dim(training)
#[1] 14718    53
```

We train using the random forest model.

```r
registerDoSNOW(makeCluster(4, type="SOCK"))

modelFit <- foreach(ntree=rep(150, 4), .combine = combine, .packages = "randomForest") %dopar% randomForest(training, training$classe, ntree=ntree, keep.forest=TRUE)
```
Using the above trained model to predict the values of testing data.

```r
predicted_in_train <- predict(modelFit,newdata = testing)
confusionMatrix(predicted_in_train, testing$classe)
```

##Results:

```r
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1395    0    0    0    0
         B    0  949    0    0    0
         C    0    0  855    0    0
         D    0    0    0  804    0
         E    0    0    0    0  901

Overall Statistics
                                     
               Accuracy : 1          
                 95% CI : (0.9992, 1)
    No Information Rate : 0.2845     
    P-Value [Acc > NIR] : < 2.2e-16  
                                     
                  Kappa : 1          
 Mcnemar's Test P-Value : NA         

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
Detection Rate         0.2845   0.1935   0.1743   0.1639   0.1837
Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

We can conclude from the above confusion matrix that out model has obtained 100% accuracy on classifying 25% of given test data and since it is also efficient - takes about ~2 minutes to train ~20k rows (full training data) and also gives high accuracy, further model tuning seems unnessary.
So, we can expect the out of sample error to be very low or even non-existant which largely depends on the size of the training data.

##Predicting given test data for submission

We use the same ```data_process``` fuction to process the testing data.

```r
train_data <- process_data("pml-training.csv")
test_data <- process_data("pml-testing.csv")
y <- train_data$classe
train_data$classe <- NULL

registerDoSNOW(makeCluster(4, type="SOCK"))
modelFit <- foreach(ntree=rep(200, 4), .combine = combine, .packages = "randomForest") %dopar% randomForest(train_data, y, ntree=ntree, keep.forest=TRUE)

test_data$problem_id <- NULL
```

After processing the testing and training data, we predict the values using our trained model and write them into text files.

```r
predicted_in_test <- predict(modelFit,newdata = test_data)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predicted_in_test)
```

##Conclusion

The predicted data scored 100% (as expected from training result) on the course project submission, so we can conclude that using random forest as ouu training mode and cleaning and processing the data carefully we have a highly accuracte prediction model.
