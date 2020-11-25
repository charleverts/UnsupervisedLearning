#### DIMENSIONALITY REDUCTION TECHNIQUES

# Principal Component Analysis (PCA)
# Metric Multi Dimensional Scaling (MDS)
# IsoMaps

## Two DATASETS to be examined: MNIST & WDBC

# First dataset - minmnist - 10000 examples of 5 and 8 handwritten digits (5000 + 5000).
# Every row is a grey scale image of a figure 5 or 8, represented by a one dimension vector of size 785 columns.
# First number of each row is the label of the image.
# Following 784 numbers in each row are the pixels of the 28 x 28 image whose values range from 0 - 255.

# Second dataset - WDBC - Wisconsin Diagnostic Breast Cancer.
# 569 data points classified as either malignant or benign. 
# Data computed from a digitsied image of a fine needle aspirate (FNA) of a breast mass.
# Each instance contains 30 features describing different characteristics.

## OBJECTIVE 
# Apply different dimensional reduction mehtods and determine which work best on different types of data.
# To evaluate the performance of the reduction method used, classify the data using the KNN algorithm.
# Apply the algorithm for the first time in the original dimension.
# Apply the algorithm for the scond time after dimensionality reduction.
# The difference in results will give an indication of the increase in accuracy due to reduction method used.

## 3 TYPES OF DIMENSIONALITY REDUCTION METHODS TO TEST: PCA, METRIC MDS, ISOMAP ## 

# 1.	Once you have cleaned and pre-processed your data use the KNN algorithm to classify the handwritten digits 
#     into either digit 5 or 8. Similarly do the same with the breast cancer data, classifying the observations 
#     into benign or malignant. In both cases determine the classification accuracy. 
#     Try different values of K in the KNN algorithm until you find the value of K that results 
#     in the 'best' accuracy for each type of dataset.

# 2.	Perform dimension reduction on both datasets using PCA, metric MDS and IsoMap and then 
#     attempt to classify the data with the reduced dimensions using KNN as in (1). 
#     Consider cases where the reduced number of dimensions is: 2, 3, 4, 5 and 6.  
#     Briefly, investigate and discuss the results relative to the results obtained without dimension reduction.

# 3.	Which dimension reduction method would you recommend if your goal was to classify the two handwritten digits using the KNN?

#### Load Data ####

# clear environment and set working directory
rm(list=ls())
setwd("C:/Users/user/Desktop/FinTech 2020/Unsupervised Learning/Assignment 2")

library(dplyr)      
library(ggplot2)    
library(rsample)    
library(recipes)   
library(caret)
library(gmodels)
library(tibble)
library(purrr)
library(stringr)
library(cluster)
library(NbClust)
library(fpc)
library(mclust)
library(tidyverse)
library(gridExtra)
library(tidyverse)
library(magrittr)
library(cluster.datasets)
library(cowplot)
library(NbClust)
library(clValid)
library(ggfortify)
library(clustree)
library(dendextend)
library(factoextra)
library(FactoMineR)
library(corrplot)
library(GGally)
library(ggiraphExtra)
library(knitr)
library(kableExtra)

# mnist dataset 10000 x 785
mnist <- read.csv("min_mnist.csv", header = TRUE) 
head(mnist)
str(mnist)
View(mnist)

# Change outcome from int to factor with 2 levels 5 & 8
mnist$X5 <- as.factor(mnist$X5)
str(mnist)

# 80/20 train/test split
set.seed(23)
sample_mnist <- createDataPartition(mnist[,"X5"],1,.8)[[1]]
View(sample_mnist)

#80% train split
mnist_train = mnist[sample_mnist,]
mnist_train$X5 <- as.factor(mnist_train$X5)
head(mnist_train)
View(mnist_train)

#20% test split
mnist_test = mnist[-sample_mnist,]
mnist_test$X5 <- as.factor(mnist_test$X5)
head(mnist_test)
View(mnist_test)

## KNN for mnist dataset (Benchmark Accuracy)

View(mnist_train)
mnist_train_x <- mnist_train[,2:785]
mnist_train_y <- mnist_train[,1]

# Rename features
colnames(mnist_train_x) <- paste0("V", 1:ncol(mnist_train_x))

# do this as part of dimentsionality reduction
mnist_train_x %>%
   as.data.frame() %>%
   map_df(sd) %>%
   gather(feature, sd) %>%
   ggplot(aes(sd)) +
   geom_histogram(binwidth = 1, col = "black", fill = "red") + 
   labs(title = "Standard Deviation of MNIST Features",
       x = "Standard Deviation",
       y = "Count") +
   theme(plot.title = element_text(hjust = 0.5))

# Remove near zero variance features manually
nzv <- nearZeroVar(mnist_train_x)
index <- setdiff(1:ncol(mnist_train_x), nzv)
mnist_train_x <- mnist_train_x[, index]

# Use train/validate resampling method
cv <- trainControl(
  method = "LGOCV", 
  p = 0.7,
  number = 1,
  savePredictions = TRUE)

# Create a hyperparameter grid search
hyper_grid <- expand.grid(k = seq(3, 25, by = 2))

# Execute grid search
knn_mnist <- train(
  mnist_train_x,
  mnist_train_y,
  method = "knn",
  tuneGrid = hyper_grid,
  preProc = c("center", "scale"),
  trControl = cv
)

#establish the best value for # neighbours
ggplot(knn_mnist) + labs(title = "Best Value for # Neighbours",
                       x ="# Neighbours", y = "Classification Accuracy") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_point(color = '#FF00CC') +
  geom_line(color = '#FF00CC') +
  geom_vline(xintercept = 3, linetype = 'dashed')

mnist_train_x <- mnist_train[,2:785]

mnist_train_y <- mnist_train[,1]

mnist_test_x <- mnist_test[,2:785]

mnist_test_y <- mnist_test[,1]


library(class)
mnist_test_pred <- knn(train = mnist_train_x, test = mnist_test_x,
                      cl = mnist_train_y, k = 3)

View(mnist_test_pred)

table <- CrossTable(x = mnist_test_y, y = mnist_test_pred, prop.chisq = FALSE)

cm <- confusionMatrix(mnist_test_pred, mnist_test_y)
cm$overall[1]*100 # accuracy
cm$byClass[c(1:2, 11)]*100 # sensitivity specificity accuracy

## KNN for wdbc dataset (Benchmark Accuracy)

# wdbc dataset 569 x 31
wdbc <- read.csv("WDBC.csv", header = TRUE) 
head(wdbc)
str(wdbc)
sum(is.na(wdbc)) #0 missing values

# remove id column
library(dplyr)
wdbc <- wdbc %>% select(-id) 
str(wdbc)

# normalise the data for wdbc only
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

wdbc_n <- as.data.frame(lapply(wdbc[2:31], normalize))
View(wdbc_n)
wdbc <- cbind(wdbc$diagnosis, wdbc_n)
View(wdbc)
names(wdbc)[names(wdbc)=="wdbc$diagnosis"] <- "diagnosis"
View(wdbc)

# 80/20 train/test split
set.seed(23)
sample_wdbc <- createDataPartition(wdbc[,"diagnosis"],1,.8)[[1]]

#80% train split
wdbc_train = wdbc[sample_wdbc,]
wdbc_train$diagnosis <- as.factor(wdbc_train$diagnosis)
head(wdbc_train)
View(wdbc_train)

#20% test split
wdbc_test = wdbc[-sample_wdbc,]
wdbc_test$diagnosis <- as.factor(wdbc_test$diagnosis)
head(wdbc_test)
View(wdbc_test)

## KNN for wdbc dataset

View(wdbc_train)
wdbc_train_x <- wdbc_train[,2:31]
wdbc_train_y <- wdbc_train[,1]

# Use train/validate resampling method
cv <- trainControl(
  method = "LGOCV", 
  p = 0.7,
  number = 1,
  savePredictions = TRUE)

# Create a hyperparameter grid search
hyper_grid <- expand.grid(k = seq(3, 25, by = 2))

# Execute grid search
knn_wdbc <- train(
  wdbc_train_x,
  wdbc_train_y,
  method = "knn",
  tuneGrid = hyper_grid,
  preProc = c("center", "scale"),
  trControl = cv
)

#establish the best value for # neighbours
ggplot(knn_wdbc) + labs(title = "Best Value for # Neighbours",
                         x ="# Neighbours", y = "Classification Accuracy") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_point(color = '#00FF00') +
  geom_line(color = '#00FF00') +
  geom_vline(xintercept = 3, linetype = 'dashed')

wdbc_train_x <- wdbc_train[,2:31]
wdbc_train_y <- wdbc_train[,1]

wdbc_test_x <- wdbc_test[,2:31]
wdbc_test_y <- wdbc_test[,1]

library(class)
wdbc_test_pred <- knn(train = wdbc_train_x, test = wdbc_test_x,
                       cl = wdbc_train_y, k = 3)

CrossTable(x = wdbc_test_y, y = wdbc_test_pred, prop.chisq = FALSE)

cm <- confusionMatrix(wdbc_test_pred, wdbc_test_y)
cm$overall[1]*100 # accuracy
cm$byClass[c(1:2, 11)]*100 # sensitivity specificity accuracy

# Top 20 most important features
vi <- varImp(knn_wdbc)
ggplot(vi, aes(col = importance))

#### Dimensionality Reduction ####

#### Principal Component Analysis ####

# work on wdbc first since dataset smaller and results are quicker
#apply to mnist once the code is finalised for wdbc

#Each of these explains a percentage of the total variation in the dataset.
wdbc.pr <- prcomp(wdbc[c(2:31)], center = TRUE, scale = TRUE)
summary(wdbc.pr)

screeplot(wdbc.pr, type = "l", npcs = 30, main = "Screeplot for all PCs")
abline(h = 1, col = "red", lty = 5)
legend("topright", legend = c("Eigenvalue = 1"),
       col = c("red"), lty = 5, cex = 1)
cumpro <- cumsum(wdbc.pr$sdev^2 / sum(wdbc.pr$sdev^2))
plot(cumpro[0:15], xlab = "PC #", ylab = "Amount of explained variance", main = "Cumulative variance plot")
abline(v = 6, col = "blue", lty = 5)
abline(h = 0.88759, col = "blue", lty = 5)
legend("topleft", legend = c("Cut-off @ PC6"),
       col = c("blue"), lty = 5, cex = 1)

library("factoextra")
fviz_pca_ind(wdbc.pr, geom.ind = "point", pointshape = 21, 
             pointsize = 2, 
             fill.ind = wdbc$diagnosis, 
             col.ind = "black", 
             palette = "jco", 
             addEllipses = TRUE,
             label = "var",
             col.var = "black",
             repel = TRUE,
             legend.title = "Diagnosis") +
  ggtitle("2D PCA-plot from 30 feature dataset") +
  theme(plot.title = element_text(hjust = 0.5))

wdbc_pca <- as.data.frame(wdbc.pr$x[,1:6]) 
wdbc_pca <- cbind(wdbc$diagnosis, wdbc_pca)
names(wdbc_pca)[names(wdbc_pca)=="wdbc$diagnosis"] <- "diagnosis"
wdbc_pca # use this in the rerunning of KNN

## Principal Component Analysis Visualisation

# in this section, variable importance is evaluated and 
# dimensionality reduction conducted.

wdbc_no_y <- wdbc[,2:31]
View(wdbc_no_y)
corrplot(cor(wdbc_no_y), type = "upper", method = "color", 
         tl.cex = 0.9, tl.col = "#0033CC")

wdbc.pca <- PCA(wdbc_no_y,  graph = FALSE)

# Visualize eigenvalues/variances

plot1 <- fviz_screeplot(wdbc.pca, addlabels = TRUE) + ggtitle("Scree Plot") +
  theme(plot.title = element_text(hjust = 0.5))
plot1

# Extract the results for variables

var <- get_pca_var(wdbc.pca)
dim1 <- var$contrib[,1]
dim2 <- var$contrib[,2]

# Contributions of variables to PC1

plot2 <- fviz_contrib(wdbc.pca, choice = "var", axes = 1, top = 20) + 
  ggtitle("Contribution of variables to Dim-1") +
  theme(plot.title = element_text(hjust = 0.5))
plot2

# Contributions of variables to PC2

plot3 <- fviz_contrib(wdbc.pca, choice = "var", axes = 2, top = 20) + 
  ggtitle("Contribution of variables to Dim-2") +
  theme(plot.title = element_text(hjust = 0.5))
plot3

# Control variable colors using their contributions to the principle axis

plot3 <- fviz_pca_var(wdbc.pca, col.var="contrib",
                       gradient.cols = c("#FFFF00", "#00FF00"),
                       repel = TRUE) + # Avoid text overlapping 
  theme_dark() + ggtitle("Variables - PCA") +
  theme(plot.title = element_text(hjust = 0.5))
plot3

grid.arrange(plot1, plot2, nrow = 1)
plot3

# PCA mnist

mnist.pr <- prcomp(mnist[c(2:785)], center = TRUE, scale = FALSE)
summary(mnist.pr)
mnist.pr

screeplot(mnist.pr, type = "l", npcs = 50, main = "Screeplot for 30 PCs")

cumpro <- cumsum(mnist.pr$sdev^2 / sum(mnist.pr$sdev^2))

plot(cumpro[0:50], xlab = "PC #", ylab = "Amount of explained variance", main = "Cumulative variance plot")
abline(h = 0.8, v = 42, col = "blue", lty = 5)
abline(h = 0.88759, col = "blue", lty = 5)
legend("topleft", legend = c("Cut-off @ PC #42"),
       col = c("blue"), lty = 5, cex = 1)

plot(mnist.pr$x[,1],mnist.pr$x[,2], xlab = "PC1 (11.7%)", ylab = "PC2 (8.1%)", 
     main = "PC1 / PC2 - plot")

library("factoextra")
fviz_pca_ind(mnist.pr, geom.ind = "point", pointshape = 21, 
             pointsize = 2, 
             fill.ind = mnist$X5, 
             col.ind = "black", 
             palette = "jco", 
             addEllipses = TRUE,
             label = "var",
             col.var = "black",
             repel = TRUE,
             legend.title = "Number 5 or 8") +
  ggtitle("2D PCA-plot from 784 feature dataset") +
  theme(plot.title = element_text(hjust = 0.5))

mnist_pca <- as.data.frame(mnist.pr$x[,1:42]) # final dataset to be used when rerunning KNN
mnist_pca <- cbind(mnist$X5, mnist_pca)
names(mnist_pca)[names(mnist_pca)=="mnist$X5"] <- "X5"
mnist_pca # final dataset to be used when rerunning KNN

# Attempting to visualise the mnist dataset in R is essentially impossible

# Column bind the target variable to the PC 1:6 & 1:42 dataframes and rerun KNN

#### Metric MDS ####

## start with wdbc dataset

#install.packages(c("vegan", "ecodist", "labdsv", "ape", "ade4", "smacof"))

# load packages
library(vegan)
library(ecodist)
library(labdsv)
library(ape)
library(ade4)
library(smacof)

d <- dist(wdbc[,2:31])  # compute distance matrix
scaled_6_wdbc <- cmdscale(d, k = 6)  # perform MDS. k = 6, same as PCA number of components
head(scaled_6_wdbc)

scaled_6_wdbc <- as.data.frame(scaled_6_wdbc) #scaled dataset
wdbc_mds <- cbind(wdbc$diagnosis, scaled_6_wdbc)
names(wdbc_mds)[names(wdbc_mds)=="wdbc$diagnosis"] <- "diagnosis"
wdbc_mds # final dataset to be used when rerunning KNN

ggplot() + geom_point(data = as.data.frame(scaled_6_wdbc), 
                      mapping = aes(x = V1,y = V2),
                      alpha = 0.5, 
                      color = "#00FF00", size = 2 ) + geom_text(data = as.data.frame(scaled_6_wdbc), 
                                                              mapping = aes(x = V1,y = V2), 
                                                              label = rownames(wdbc) ) + 
  labs(title = "MDS representation of pair-wise distances of WDBC") +
  theme(plot.title = element_text(hjust = 0.5))

## Metric MDS for mnist

d <- dist(mnist[,2:785])  # compute distance matrix
scaled_6_mnist <- cmdscale(d, k = 6)  # perform MDS. k = 6, same as PCA number of components
head(scaled_6_mnist)

scaled_6_mnist <- as.data.frame(scaled_6_mnist) #scaled dataset
mnist_mds <- cbind(mnist$X5, scaled_6_mnist)
names(mnist_mds)[names(mnist_mds)=="mnist$X5"] <- "X5"
mnist_mds # final dataset to be used when rerunning KNN

ggplot() + geom_point(data = as.data.frame(scaled_6_mnist), 
                      mapping = aes(x = V1,y = V2),
                      alpha = 0.5, 
                      color = "#FF00CC", size = 1 ) + geom_text(data = as.data.frame(scaled_6_mnist), 
                                                                mapping = aes(x = V1,y = V2), 
                                                                label = rownames(mnist) ) + 
  labs(title = "MDS representation of pair-wise distances of MNIST") +
  theme(plot.title = element_text(hjust = 0.5))

#### isoMAPS ####

# WDBC dataset

library(RDRToolbox)

wdbc_iso_data <- as.matrix(wdbc[,2:31]) 
wdbc_iso = Isomap(data = wdbc_iso_data, dims = 2:6, k = 5, plotResiduals = TRUE) #6dims has best result
wdbc_iso = Isomap(data = wdbc_iso_data, dims = 6, k = 5)
head(wdbc_iso)
wdbc_iso_final <- as.data.frame(wdbc_iso)

colnames(wdbc_iso_final) <- paste0("V", 1:ncol(wdbc_iso_final))
wdbc_isomaps <- cbind(wdbc$diagnosis, wdbc_iso_final)
names(wdbc_isomaps)[names(wdbc_isomaps)=="wdbc$diagnosis"] <- "diagnosis"
wdbc_isomaps # final dataset to be used when rerunning KNN

#### FAILED TO COMPILE DUE TO RUN TIME ERROR ####
# # MNIST dataset
# 
# mnist_iso_data <- as.matrix(mnist[,2:785]) 
# mnist_iso = Isomap(data = mnist_iso_data, dims = 2, k = 3) #6dims has best result
# 
# head(mnist_iso)
# mnist_iso_final <- as.data.frame(mnist_iso)
# mnist_iso_final #final dataset to be check 
# 
# colnames(mnist_iso_final) <- paste0("V", 1:ncol(mnist_iso_final))
# wdbc_isomaps <- cbind(wdbc$diagnosis, wdbc_iso_final)
# names(wdbc_isomaps)[names(wdbc_isomaps)=="wdbc$diagnosis"] <- "diagnosis"
# wdbc_isomaps # final dataset to be used when rerunning KNN


#### Rerun the KNN algorithm on reduced datasets ####

# use the same partitioning as in first kNN so as to preserve the same row entries in train and test:
# sample_wdbc was already created in first section, will re use here.

#### WDBC dataset after PCA ####

#80% train split
wdbc_train = wdbc_pca[sample_wdbc,]
wdbc_train$diagnosis <- as.factor(wdbc_train$diagnosis)
head(wdbc_train)
View(wdbc_train)

#20% test split
wdbc_test = wdbc_pca[-sample_wdbc,]
wdbc_test$diagnosis <- as.factor(wdbc_test$diagnosis)
head(wdbc_test)
View(wdbc_test)

## KNN for wdbc_pca dataset

View(wdbc_train)
wdbc_train_x <- wdbc_train[,2:7]
wdbc_train_y <- wdbc_train[,1]

# Use train/validate resampling method
cv <- trainControl(
  method = "LGOCV", 
  p = 0.7,
  number = 1,
  savePredictions = TRUE)

# Create a hyperparameter grid search
hyper_grid <- expand.grid(k = seq(3, 25, by = 2))

# Execute grid search
knn_wdbc <- train(
  wdbc_train_x,
  wdbc_train_y,
  method = "knn",
  tuneGrid = hyper_grid,
  preProc = c("center", "scale"),
  trControl = cv
)

#establish the best value for # neighbours
ggplot(knn_wdbc) + labs(title = "Best Value for # Neighbours",
                         x ="# Neighbours", y = "Classification Accuracy") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_point(color = '#00FF00') +
  geom_line(color = '#00FF00') +
  geom_vline(xintercept = 5, linetype = 'dashed')

str(knn_wdbc)

wdbc_train_x <- wdbc_train[,2:7]
wdbc_train_y <- wdbc_train[,1]

wdbc_test_x <- wdbc_test[,2:7]
wdbc_test_y <- wdbc_test[,1]

library(class)
wdbc_test_pred <- knn(train = wdbc_train_x, test = wdbc_test_x,
                      cl = wdbc_train_y, k = 5)

CrossTable(x = wdbc_test_y, y = wdbc_test_pred, prop.chisq = FALSE)

cm <- confusionMatrix(wdbc_test_pred, wdbc_test_y)
cm$overall[1]*100 # accuracy
cm$byClass[c(1:2)]*100 # sensitivity specificity accuracy

# Top 20 most important features
vi <- varImp(knn_wdbc)
ggplot(vi)

#### WDBC dataset after MDS ####

#80% train split
wdbc_train = wdbc_mds[sample_wdbc,]
wdbc_train$diagnosis <- as.factor(wdbc_train$diagnosis)
head(wdbc_train)
View(wdbc_train)

#20% test split
wdbc_test = wdbc_mds[-sample_wdbc,]
wdbc_test$diagnosis <- as.factor(wdbc_test$diagnosis)
head(wdbc_test)
View(wdbc_test)

## KNN for wdbc_mds dataset

View(wdbc_train)
wdbc_train_x <- wdbc_train[,2:7]
wdbc_train_y <- wdbc_train[,1]

# Use train/validate resampling method
cv <- trainControl(
  method = "LGOCV", 
  p = 0.7,
  number = 1,
  savePredictions = TRUE)

# Create a hyperparameter grid search
hyper_grid <- expand.grid(k = seq(3, 25, by = 2))

# Execute grid search
knn_wdbc <- train(
  wdbc_train_x,
  wdbc_train_y,
  method = "knn",
  tuneGrid = hyper_grid,
  preProc = c("center", "scale"),
  trControl = cv
)

#establish the best value for # neighbours
ggplot(knn_wdbc) + labs(title = "Best Value for # Neighbours",
                        x ="# Neighbours", y = "Classification Accuracy") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_point(color = '#00FF00') +
  geom_line(color = '#00FF00') +
  geom_vline(xintercept = 3, linetype = 'dashed')

str(knn_wdbc)

wdbc_train_x <- wdbc_train[,2:7]
wdbc_train_y <- wdbc_train[,1]

wdbc_test_x <- wdbc_test[,2:7]
wdbc_test_y <- wdbc_test[,1]

library(class)
wdbc_test_pred <- knn(train = wdbc_train_x, test = wdbc_test_x,
                      cl = wdbc_train_y, k = 3)

CrossTable(x = wdbc_test_y, y = wdbc_test_pred, prop.chisq = FALSE)

cm <- confusionMatrix(wdbc_test_pred, wdbc_test_y)
cm$overall[1]*100 # accuracy
cm$byClass[c(1:2)]*100 # sensitivity specificity accuracy

# Top 20 most important features
vi <- varImp(knn_wdbc)
ggplot(vi)

#### WDBC dataset after ISOmap ####

#80% train split
wdbc_train = wdbc_isomaps[sample_wdbc,]
wdbc_train$diagnosis <- as.factor(wdbc_train$diagnosis)
head(wdbc_train)
View(wdbc_train)

#20% test split
wdbc_test = wdbc_isomaps[-sample_wdbc,]
wdbc_test$diagnosis <- as.factor(wdbc_test$diagnosis)
head(wdbc_test)
View(wdbc_test)

## KNN for wdbc_isomaps dataset

View(wdbc_train)
wdbc_train_x <- wdbc_train[,2:7]
wdbc_train_y <- wdbc_train[,1]

# Use train/validate resampling method
cv <- trainControl(
  method = "LGOCV", 
  p = 0.7,
  number = 1,
  savePredictions = TRUE)

# Create a hyperparameter grid search
hyper_grid <- expand.grid(k = seq(3, 25, by = 2))

# Execute grid search
knn_wdbc <- train(
  wdbc_train_x,
  wdbc_train_y,
  method = "knn",
  tuneGrid = hyper_grid,
  preProc = c("center", "scale"),
  trControl = cv
)

#establish the best value for # neighbours
ggplot(knn_wdbc) + labs(title = "Best Value for # Neighbours",
                        x ="# Neighbours", y = "Classification Accuracy") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_point(color = '#00FF00') +
  geom_line(color = '#00FF00') +
  geom_vline(xintercept = 5, linetype = 'dashed')

str(knn_wdbc)

wdbc_train_x <- wdbc_train[,2:7]
wdbc_train_y <- wdbc_train[,1]

wdbc_test_x <- wdbc_test[,2:7]
wdbc_test_y <- wdbc_test[,1]

library(class)
wdbc_test_pred <- knn(train = wdbc_train_x, test = wdbc_test_x,
                      cl = wdbc_train_y, k = 5)

CrossTable(x = wdbc_test_y, y = wdbc_test_pred, prop.chisq = FALSE)

cm <- confusionMatrix(wdbc_test_pred, wdbc_test_y)
cm$overall[1]*100 # accuracy
cm$byClass[c(1:2)]*100 # sensitivity specificity accuracy

# Top 20 most important features
vi <- varImp(knn_wdbc)
ggplot(vi)

#### MNIST dataset after PCA ####

# use the same partitioning as in first kNN so as to preserve the same row entries in train and test:
# sample_wdbc was already created in first section, will re use here.

#80% train split
mnist_train = mnist_pca[sample_mnist,]
mnist_train$X5 <- as.factor(mnist_train$X5)
head(mnist_train)
View(mnist_train)

#20% test split
mnist_test = mnist_pca[-sample_mnist,]
mnist_test$X5 <- as.factor(mnist_test$X5)
head(mnist_test)
View(mnist_test)

View(mnist_train)
mnist_train_x <- mnist_train[,2:43]
mnist_train_y <- mnist_train[,1]

# Rename features
colnames(mnist_train_x) <- paste0("V", 1:ncol(mnist_train_x))

# Use train/validate resampling method
cv <- trainControl(
  method = "LGOCV", 
  p = 0.7,
  number = 1,
  savePredictions = TRUE)

# Create a hyperparameter grid search
hyper_grid <- expand.grid(k = seq(3, 25, by = 2))

# Execute grid search
knn_mnist <- train(
  mnist_train_x,
  mnist_train_y,
  method = "knn",
  tuneGrid = hyper_grid,
  preProc = c("center", "scale"),
  trControl = cv
)

#establish the best value for # neighbours
ggplot(knn_mnist) + labs(title = "Best Value for # Neighbours",
                        x ="# Neighbours", y = "Classification Accuracy") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_point(color = '#FF00CC') +
  geom_line(color = '#FF00CC') +
  geom_vline(xintercept = 5, linetype = 'dashed')

mnist_train_x <- mnist_train[,2:43]
View(mnist_train_x)
mnist_train_y <- mnist_train[,1]
View(mnist_train_y)

mnist_test_x <- mnist_test[,2:43]
View(mnist_test_x)
mnist_test_y <- mnist_test[,1]
View(mnist_test_y)

library(class)
mnist_test_pred <- knn(train = mnist_train_x, test = mnist_test_x,
                       cl = mnist_train_y, k = 5)

table <- CrossTable(x = mnist_test_y, y = mnist_test_pred, prop.chisq = FALSE)
cm <- confusionMatrix(mnist_test_pred, mnist_test_y)
cm$overall[1]*100 # accuracy
cm$byClass[c(1:2)]*100 # sensitivity specificity accuracy

#### MNIST dataset after MDS ####

#80% train split
mnist_train = mnist_mds[sample_mnist,]
mnist_train$X5 <- as.factor(mnist_train$X5)
head(mnist_train)

#20% test split
mnist_test = mnist_mds[-sample_mnist,]
mnist_test$X5 <- as.factor(mnist_test$X5)
head(mnist_test)

mnist_train_x <- mnist_train[,2:7]
mnist_train_y <- mnist_train[,1]

# Rename features
colnames(mnist_train_x) <- paste0("V", 1:ncol(mnist_train_x))

# Use train/validate resampling method
cv <- trainControl(
  method = "LGOCV", 
  p = 0.7,
  number = 1,
  savePredictions = TRUE)

# Create a hyperparameter grid search
hyper_grid <- expand.grid(k = seq(3, 25, by = 2))

# Execute grid search
knn_mnist <- train(
  mnist_train_x,
  mnist_train_y,
  method = "knn",
  tuneGrid = hyper_grid,
  preProc = c("center", "scale"),
  trControl = cv
)

#establish the best value for # neighbours
ggplot(knn_mnist) + labs(title = "Best Value for # Neighbours",
                         x ="# Neighbours", y = "Classification Accuracy") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_point(color = '#FF00CC') +
  geom_line(color = '#FF00CC') +
  geom_vline(xintercept = 7, linetype = 'dashed')

mnist_train_x <- mnist_train[,2:7]
View(mnist_train_x)
mnist_train_y <- mnist_train[,1]
View(mnist_train_y)

mnist_test_x <- mnist_test[,2:7]
View(mnist_test_x)
mnist_test_y <- mnist_test[,1]
View(mnist_test_y)

library(class)
mnist_test_pred <- knn(train = mnist_train_x, test = mnist_test_x,
                       cl = mnist_train_y, k = 7)

table <- CrossTable(x = mnist_test_y, y = mnist_test_pred, prop.chisq = FALSE)
cm <- confusionMatrix(mnist_test_pred, mnist_test_y)
cm$overall[1]*100 # accuracy
cm$byClass[c(1:2, 11)]*100 # sensitivity specificity accuracy

#### MNIST dataset after ISOmap ####

# runtime error - jusfified in text why a poor application to MNIST




  