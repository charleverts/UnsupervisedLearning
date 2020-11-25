#### Data Processing ####

# clear environment and set working directory

rm(list=ls())
setwd("C:/Users/user/Desktop/FinTech 2020/Unsupervised Learning/Assignment 1")
getwd()

# read data

library("readxl")

data <- read_excel("owid-covid-data.xlsx", sheet = 2)
data <- as.data.frame(data)
head(data)
colnames(data)
View(data)

# deleting iso_code, date and "new" columns

data$iso_code <- NULL
data$date <- NULL
data$new_cases <- NULL
data$new_cases_smoothed <- NULL
data$new_deaths <- NULL
data$new_deaths_smoothed <- NULL
data$new_cases_per_million <- NULL
data$new_cases_smoothed_per_million <- NULL
data$new_deaths_per_million <- NULL
data$new_deaths_smoothed_per_million <- NULL
colnames(data) # inspect remaining 20 columns
View(data)

# identify columns without missing values

colnames(data)[colSums(is.na(data)) == 0]

# identify columns with missing values

colnames(data)[colSums(is.na(data)) > 0]

rowSums(is.na(data)) >= 5

# omit entries with more than five missing values

data <- data[-which(rowSums(is.na(data)) >= 5),]
str(data)
View(data)

# find the number of missing values in each row

rowSums(is.na(data))

## Missing Values 

# imputation of missing values in extreme_poverty column

library(tidyverse)
library(ggpubr)
theme_set(theme_pubr())

ggplot(data, aes(x = gdp_per_capita, y = extreme_poverty)) +
  geom_point() + stat_smooth() + 
  scale_x_continuous(limits = c(0, 75000)) +
  scale_y_continuous(limits = c(0, 80))

# required for deterministic and stochastic regression imputation

library("mice") 
library("lattice")
View(data)

# Cuba and Somalia do not have gdp_per_capita entries.
# These values will be manually entered: based on WorldBank Stats. 

data$gdp_per_capita[data$location == "Cuba"] <- 6816.90
data$gdp_per_capita[data$location == "Somalia"] <- 
  (1136.103 + 1390.300)/2 # average of Sierra leone and Mozambique

# Create new data frame containing gdp_per_capita and extreme_poverty
# to conduct mean value, deterministic, and stochastic regression imputation.

data_new1 <- data[, c("location", "gdp_per_capita","extreme_poverty")]
data_new1
md.pattern(data_new1) # visualise missing values (54)

# Inspect correlation and significance between gdp_per_capita and extreme_poverty.
# Evaluate whether or not a significant relationship exists between
# the predictor and response variable.

reduced_EP <- na.omit(data_new1)
cor(reduced_EP$gdp_per_capita, reduced_EP$extreme_poverty)

fit1 <- with(data_new1, lm(extreme_poverty ~ gdp_per_capita))
summary(fit1) #gdp_per_capita highly significant in predicting extreme_poverty

# mean value imputation (worst)

imp_mean1 <- mice(data_new1, method = "mean", m = 1, maxit = 1)
complete(imp_mean1)

# stochastic regression model imputation (poor)

imp_stochastic1 <- mice(data_new1, method = "norm.nob", m = 1, maxit = 1)
complete(imp_stochastic1)

# deterministic regression model imputation (best)

imp_deterministic1 <- mice(data_new1, method = "norm.predict", m = 1, maxit = 1)
extreme_poverty_fixed <- complete(imp_deterministic1)
extreme_poverty_fixed <- replace(extreme_poverty_fixed$extreme_poverty,
                                 extreme_poverty_fixed$extreme_poverty < 0,
                                 0)
extreme_poverty_fixed

# replace orginal extreme_poverty column with extreme_poverty_fixed

data$extreme_poverty <- extreme_poverty_fixed
data$extreme_poverty <- round(data$extreme_poverty, 2)
sum(is.na(data$extreme_poverty)) # equals zero => no NA values exist

## imputation of missing values in handwashing_facilities

ggplot(data, aes(x = gdp_per_capita, y = handwashing_facilities)) +
  geom_point() + scale_x_continuous(limits = c(0, 40000)) +
  stat_smooth() + scale_y_continuous(limits = c(0, 100)) 

# Create new data frame containing gdp_per_capita and handwashing_facilities
# to conduct mean value, deterministic, and stochastic regression imputation.

data_new2 <- data[, c("location", "gdp_per_capita","handwashing_facilities")]
data_new2
md.pattern(data_new2) # visualise missing values
sum(is.na(data$handwashing_facilities)) # 83/173 => 48% missing values

# Inspect correlation and significance between 
# gdp_per_capita and handwashing_facilities.
# Evaluate whether or not a significant relationship exists between
# the predictor and response variable.

reduced_hand <- na.omit(data_new2)
cor(reduced_hand$gdp_per_capita, reduced_hand$handwashing_facilities)

fit2 <- with(data_new2, lm(handwashing_facilities ~ gdp_per_capita))
summary(fit2) #gdp_per_capita highly significant in predicting handwashing_facilities

# mean value imputation 

imp_mean2 <- mice(data_new2, method = "mean", m = 1, maxit = 1)
complete(imp_mean2)

# deterministic regression model imputation 

imp_deterministic2 <- mice(data_new2, method = "norm.predict", m = 1, maxit = 1)
complete(imp_deterministic2)

# stochastic regression model imputation 

imp_stochastic2 <- mice(data_new2, method = "norm.nob", m = 1, maxit = 1)
complete(imp_stochastic2)

# no suitable imputation method found for handwashing_facilities
# drop column from dataframe.

data$handwashing_facilities <- NULL
View(data) #inspect

## imputation of missing values in hospital_beds_per_thousand

ggplot(data, aes(x = gdp_per_capita, y = hospital_beds_per_thousand)) +
  geom_point() + stat_smooth() + scale_y_continuous(limits = c(0, 13)) + 
  scale_x_continuous(limits = c(0, 75000))

# Create new data frame containing gdp_per_capita and hospital_beds_per_thousand
# to conduct mean value, deterministic, and stochastic regression imputation.

data_new3 <- data[, c("location", "gdp_per_capita","hospital_beds_per_thousand")]
data_new3
md.pattern(data_new3) # visualise missing values (16)

# inspect correlation between gdp_per_capita and hospital_beds_per_thousand

reduced_hosp <- na.omit(data_new3)
cor(reduced_hosp$gdp_per_capita, reduced_hosp$hospital_beds_per_thousand)

fit3 <- with(data_new3, lm(hospital_beds_per_thousand ~ gdp_per_capita))
summary(fit3) #gdp_per_capita highly significant in predicting hospital_beds_per_thousand

# mean value imputation (poor)

imp_mean3 <- mice(data_new3, method = "mean", m = 1, maxit = 1)
complete(imp_mean3)

# stochastic regression model imputation (poor)

imp_stochastic3 <- mice(data_new3, method = "norm.nob", m = 1, maxit = 1)
complete(imp_stochastic3)

# deterministic regression model imputation (best)

imp_deterministic3 <- mice(data_new3, method = "norm.predict", m = 1, maxit = 1)
hospital_beds_fixed <- complete(imp_deterministic3)
hospital_beds_fixed <- hospital_beds_fixed$hospital_beds_per_thousand
hospital_beds_fixed <- round(hospital_beds_fixed, 2)
hospital_beds_fixed

# replace orginal hospital_beds_per_thousand column with hospital_beds_fixed

data$hospital_beds_per_thousand <- hospital_beds_fixed
sum(is.na(data$hospital_beds_per_thousand)) # equals zero => no NA values exist

## imputation of missing values in smoking data

data_new4 <- data[, c("location", "female_smokers","male_smokers")]
data_new4
md.pattern(data_new4) # visualise missing values

# Africa

female_africa <- data$female_smokers[data$continent == "Africa"]
female_africa <- na.omit(female_africa)
mean(female_africa)
median(female_africa)
hist(female_africa, main = "female_africa", col = "#FF00CC")

male_africa <- data$male_smokers[data$continent == "Africa"]
male_africa <- na.omit(male_africa)
mean(male_africa)
median(male_africa)
hist(male_africa, main = "male_africa", col = "#3333FF")

# Asia

female_asia <- data$female_smokers[data$continent == "Asia"]
female_asia <- na.omit(female_asia)
mean(female_asia)
median(female_asia)
hist(female_asia, main = "female_asia", col = "#FF00CC")

male_asia <- data$male_smokers[data$continent == "Asia"]
male_asia <- na.omit(male_asia)
mean(male_asia)
median(male_asia)
hist(male_asia, main = "male_asia", col = "#3333FF")

# Europe

female_europe <- data$female_smokers[data$continent == "Europe"]
female_europe <- na.omit(female_europe)
mean(female_europe)
median(female_europe)
hist(female_europe, main = "female_europe", col = "#FF00CC")

male_europe <- data$male_smokers[data$continent == "Europe"]
male_europe <- na.omit(male_europe)
mean(male_europe)
median(male_europe)
hist(male_europe, main = "male_europe", col = "#3333FF")

# North America

female_NA <- data$female_smokers[data$continent == "North America"]
female_NA <- na.omit(female_NA)
mean(female_NA)
median(female_NA)
hist(female_NA, main = "female_NorthAm", col = "#FF00CC")

male_NA <- data$male_smokers[data$continent == "North America"]
male_NA <- na.omit(male_NA)
mean(male_NA)
median(male_NA)
hist(male_NA, main = "male_NorthAm", col = "#3333FF")

# Oceania

female_oceania <- data$female_smokers[data$continent == "Oceania"]
female_oceania <- na.omit(female_oceania)
mean(female_oceania)
median(female_oceania)
hist(female_oceania, main = "female_oceania", col = "#FF00CC")

male_oceania <- data$male_smokers[data$continent == "Oceania"]
male_oceania <- na.omit(male_oceania)
mean(male_oceania)
median(male_oceania)
hist(male_oceania, main = "male_oceania", col = "#3333FF")

# South America

female_SA <- data$female_smokers[data$continent == "South America"]
female_SA <- na.omit(female_SA)
mean(female_SA)
median(female_SA)
hist(female_SA, main = "female_SouthAm", col = "#FF00CC")

male_SA <- data$male_smokers[data$continent == "South America"]
male_SA <- na.omit(male_SA)
mean(male_SA)
median(male_SA)
hist(male_SA, main = "male_SouthAm", col = "#3333FF")

# Replace missing values with mean/median of the respective gender and continents

# Africa

sd(female_africa)
sd(male_africa)

data$female_smokers[data$continent == "Africa"] <- 
                      replace_na(data$female_smokers[data$continent == "Africa"], 
                                 mean(female_africa))
round(data$female_smokers[data$continent == "Africa"], 2)

data$male_smokers[data$continent == "Africa"] <- 
  replace_na(data$male_smokers[data$continent == "Africa"], 
             median(male_africa))
round(data$male_smokers[data$continent == "Africa"], 2)

# Asia

sd(female_asia)
sd(male_asia)

data$female_smokers[data$continent == "Asia"] <- 
  replace_na(data$female_smokers[data$continent == "Asia"], 
             mean(female_asia))
round(data$female_smokers[data$continent == "Asia"], 2)

data$male_smokers[data$continent == "Asia"] <- 
  replace_na(data$male_smokers[data$continent == "Asia"], 
             mean(male_asia))
round(data$male_smokers[data$continent == "Asia"], 2)

# Europe

sd(female_europe)
sd(male_europe)

data$female_smokers[data$continent == "Europe"] <- 
  replace_na(data$female_smokers[data$continent == "Europe"], 
             mean(female_europe))
round(data$female_smokers[data$continent == "Europe"], 2)

data$male_smokers[data$continent == "Europe"] <- 
  replace_na(data$male_smokers[data$continent == "Europe"], 
             mean(male_europe))
round(data$male_smokers[data$continent == "Europe"], 2)

# North America

sd(female_NA)
sd(male_NA)

data$female_smokers[data$continent == "North America"] <- 
  replace_na(data$female_smokers[data$continent == "North America"], 
             mean(female_NA))
round(data$female_smokers[data$continent == "North America"], 2)

data$male_smokers[data$continent == "North America"] <- 
  replace_na(data$male_smokers[data$continent == "North America"], 
             median(male_NA))
round(data$male_smokers[data$continent == "North America"], 2)

# Oceania has no missing values in the manipulated dataset

# South America

sd(female_SA)
sd(male_SA)

data$female_smokers[data$continent == "South America"] <- 
  replace_na(data$female_smokers[data$continent == "South America"], 
             median(female_SA))
round(data$female_smokers[data$continent == "South America"], 2)

data$male_smokers[data$continent == "South America"] <- 
  replace_na(data$male_smokers[data$continent == "South America"], 
             median(male_SA))
round(data$male_smokers[data$continent == "South America"], 2)

# Round all final values in female_smoker and male_smoker 

data$female_smokers <- round(data$female_smokers, 2)
data$male_smokers <- round(data$male_smokers, 2)

## aged_70_older missing value for Serbia. 

# Cacluate the average ratio of 65 year olds to 70 year olds 
# and calculate the value for Serbia.

data_new4 <- data[, c("location", "aged_65_older","aged_70_older")]
data_new4
md.pattern(data_new4)

sum_65 <- sum(data[-c(130), "aged_65_older"])
sum_65
sum_70 <- sum(data[-c(130), "aged_70_older"])
sum_70
ratio <- sum_70/sum_65
ratio

data$aged_70_older[data$location == "Serbia"] <- 
  ratio*(data$aged_65_older[data$location == "Serbia"])
data$aged_70_older[data$location == "Serbia"]

## descriptive stats

data_novia <- subset(data , select = -c(continent, location))
desc_stats <- data.frame(
  Min = apply(data_novia, 2, min), # minimum
  Med = apply(data_novia, 2, median), # median
  Mean = apply(data_novia, 2, mean), # mean
  SD = apply(data_novia, 2, sd), # Standard deviation
  Max = apply(data_novia, 2, max) # Maximum
)
desc_stats <- round(desc_stats, 1)
View(desc_stats)

# Boxplots
# the following boxplots indicate the need for scaling data

ggplot(data = data , mapping = aes(x = continent, y = total_deaths, colour = continent)) +
  geom_boxplot()
ggplot(data = data , mapping = aes(x = continent, y = total_cases, colour = continent)) +
  geom_boxplot()
ggplot(data = data , mapping = aes(x = continent, y = total_deaths_per_million, colour = continent)) +
  geom_boxplot()

#### Partitional Clustering ####

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

## KMEANS

df <- subset(data, select = -c(location, continent))
df <- scale(df)
View(df)

# kmeans clustering conducted for 2,3,4,5,6 clusters and evaluated

kmeans2 <- kmeans(df, centers = 2, nstart = 25)
str(kmeans2)
fviz_cluster(kmeans2, data = df)
kmeans2$cluster

kmeans3 <- kmeans(df, centers = 3, nstart = 25)  
kmeans4 <- kmeans(df, centers = 4, nstart = 25)  
kmeans5 <- kmeans(df, centers = 5, nstart = 25)  
kmeans6 <- kmeans(df, centers = 6, nstart = 25)

plot(silhouette(kmeans2$cluster, dist(df)), col = 2:3, main = "Silhouette Plot: 2 Clusters")
plot(silhouette(kmeans3$cluster, dist(df)), col = 2:4, main = "Silhouette Plot: 3 Clusters")
plot(silhouette(kmeans4$cluster, dist(df)), col = 2:5, main = "Silhouette Plot: 4 Clusters")
plot(silhouette(kmeans5$cluster, dist(df)), col = 2:6, main = "Silhouette Plot: 5 Clusters")
plot(silhouette(kmeans6$cluster, dist(df)), col = 2:7, main = "Silhouette Plot: 6 Clusters")

fviz_cluster(kmeans6, geom = "point", data = df) + ggtitle("K-means Cluster: k = 6") +
  theme(plot.title = element_text(hjust = 0.5))
# outliers @ [66 = India][89 = Singapore][186 = USA][198 = Brazil]

data_novia <- subset(data , select = -c(continent, location))
data_novia <- scale(data_novia)
data_novia <- data_novia[-c(64, 87, 157, 164),] # excl India, Brazil, Singapore, US

# find optimal number of clusters

# Silhouette method

fviz_nbclust(data_novia, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(plot.subtitle = element_text(hjust = 0.5))

# Gap statistic

fviz_nbclust(data_novia, kmeans, method = "gap_stat", nboot = 500) +
  labs(subtitle = "Gap statistic method") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(plot.subtitle = element_text(hjust = 0.5))

# Elbow Method

fviz_nbclust(data_novia, kmeans, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2) +
  labs(subtitle = "Elbow method") + 
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(plot.subtitle = element_text(hjust = 0.5))

kmeans2a <- kmeans(data_novia, centers = 2, nstart = 25)
str(kmeans2a)
fviz_cluster(kmeans2a, data = data_novia)
kmeans2a$cluster

kmeans3a <- kmeans(data_novia, centers = 3, nstart = 25)  
kmeans4a <- kmeans(data_novia, centers = 4, nstart = 25)  
kmeans5a <- kmeans(data_novia, centers = 5, nstart = 25)  
kmeans6a <- kmeans(data_novia, centers = 6, nstart = 25)

plot(silhouette(kmeans2a$cluster, dist(data_novia)), col = 2:3, main = "Silhouette Plot: 2 Clusters")
plot(silhouette(kmeans3a$cluster, dist(data_novia)), col = 2:4, main = "Silhouette Plot: 3 Clusters")
plot(silhouette(kmeans4a$cluster, dist(data_novia)), col = 2:5, main = "Silhouette Plot: 4 Clusters")
plot(silhouette(kmeans5a$cluster, dist(data_novia)), col = 2:6, main = "Silhouette Plot: 5 Clusters")
plot(silhouette(kmeans6a$cluster, dist(data_novia)), col = 2:7, main = "Silhouette Plot: 6 Clusters")

# Comparing the Plots - including US, Brazil, India, Singapore

plot1 <- fviz_cluster(kmeans2, geom = "point", data = df) + ggtitle("k = 2")
plot2 <- fviz_cluster(kmeans3, geom = "point", data = df) + ggtitle("k = 3")
plot3 <- fviz_cluster(kmeans4, geom = "point", data = df) + ggtitle("k = 4")
plot4 <- fviz_cluster(kmeans5, geom = "point", data = df) + ggtitle("k = 5")
plot5 <- fviz_cluster(kmeans6, geom = "point", data = df) + ggtitle("k = 6")   

plot1
plot2
plot3
plot4
plot5 # final 6 cluster plot

# Found that India, Brazil, Singapore, US cluster on their own, outliers

data_novia <- subset(data , select = -c(continent, location))
data_novia <- scale(data_novia)
data_novia <- data_novia[-c(64, 87, 157, 164),] # excl India, Brazil, Singapore, US
View(data_novia)

# find optimal number of clusters

# Silhouette method

fviz_nbclust(data_novia, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(plot.subtitle = element_text(hjust = 0.5))

# Gap statistic

fviz_nbclust(data_novia, kmeans, method = "gap_stat", nboot = 500) +
  labs(subtitle = "Gap statistic method") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(plot.subtitle = element_text(hjust = 0.5))

# Elbow Method

fviz_nbclust(data_novia, kmeans, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2) +
  labs(subtitle = "Elbow method") + 
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(plot.subtitle = element_text(hjust = 0.5))

# Compute K-Means clustering for 2,3,4,5,6 clusters after excluding
# outliers US, Singapore, Brazil, India

# k = 2

set.seed(123)
km.res2 <- kmeans(data_novia, 2, nstart = 25)

plot0 <- fviz_cluster(km.res2, data = data_novia, 
                      ellipse.type = "convex",
                      palette = "ucscgb",
                      repel = TRUE,
                      labelsize = 0,
                      ggtheme = theme_light()) + ggtitle("K-means Cluster Plot: k = 2") +
  theme(plot.title = element_text(hjust = 0.5))

# k = 3

set.seed(123)
km.res3 <- kmeans(data_novia, 3, nstart = 25)

plot1 <- fviz_cluster(km.res3, data = data_novia, 
                      ellipse.type = "convex",
                      palette = "ucscgb",
                      repel = TRUE,
                      labelsize = 0,
                      ggtheme = theme_light()) + ggtitle("K-means Cluster Plot: k = 3") +
  theme(plot.title = element_text(hjust = 0.5))

# k = 4

set.seed(123)
km.res4 <- kmeans(data_novia, 4, nstart = 25)

plot2 <- fviz_cluster(km.res4, data = data_novia, 
                      ellipse.type = "convex",
                      palette = "ucscgb",
                      repel = TRUE,
                      labelsize = 0,
                      ggtheme = theme_light()) + ggtitle("K-means Cluster Plot: k = 4") +
  theme(plot.title = element_text(hjust = 0.5))

# k = 5

set.seed(123)
km.res5 <- kmeans(data_novia, 5, nstart = 25)

plot3 <- fviz_cluster(km.res5, data = data_novia, 
                      ellipse.type = "convex",
                      palette = "ucscgb",
                      repel = TRUE,
                      labelsize = 0,
                      ggtheme = theme_light()) + ggtitle("K-means Cluster Plot: k = 5") +
  theme(plot.title = element_text(hjust = 0.5))

# k = 6

set.seed(123)
km.res6 <- kmeans(data_novia, 6, nstart = 25)

plot4 <- fviz_cluster(km.res6, data = data_novia, 
                      ellipse.type = "convex",
                      palette = "ucscgb",
                      repel = TRUE,
                      labelsize = 0,
                      ggtheme = theme_light()) + ggtitle("K-means Cluster Plot: k = 6") +
  theme(plot.title = element_text(hjust = 0.5))

plot0
plot1 
plot2 
plot3 
plot4

# Compare clusters and continents

location_continent <- data[, c(1:2)]
location_continent <- location_continent[-c(64, 87, 157, 164),]
location_continent

new_data_novia <- cbind(location_continent, km.res6$cluster)
names(new_data_novia)[names(new_data_novia)=="km.res6$cluster"] <- "cluster"
names(new_data_novia)

table(new_data_novia$cluster, new_data_novia$continent)

cluster_1 <- new_data_novia[new_data_novia$cluster == 1,]
cluster_2 <- new_data_novia[new_data_novia$cluster == 2,]
cluster_3 <- new_data_novia[new_data_novia$cluster == 3,]
cluster_4 <- new_data_novia[new_data_novia$cluster == 4,]
cluster_5 <- new_data_novia[new_data_novia$cluster == 5,]
cluster_6 <- new_data_novia[new_data_novia$cluster == 6,]

# the following code is used to write the output to xlsx workbook
# library(openxlsx)

# writing a data.frame or list of data.frames to an xlsx file

# write.xlsx(cluster_1, 'cluster_1.xlsx')
# write.xlsx(cluster_2, 'cluster_2.xlsx')
# write.xlsx(cluster_3, 'cluster_3.xlsx')
# write.xlsx(cluster_4, 'cluster_4.xlsx')
# write.xlsx(cluster_5, 'cluster_5.xlsx')
# write.xlsx(cluster_6, 'cluster_6.xlsx')

## PAM

data_novia <- subset(data , select = -c(continent, location))
data_novia <- scale(data_novia)
data_novia <- data_novia[-c(64, 87, 157, 164),]
View(data_novia)

pam.res2 <- pam(data_novia, 2)
pam.res3 <- pam(data_novia, 3)
pam.res4 <- pam(data_novia, 4)
pam.res5 <- pam(data_novia, 5)
pam.res <- pam(data_novia, 6)

plot(silhouette(pam.res2$clustering, dist(data_novia)), col = 2:3, main = "PAM Silhouette: 2 Clusters")
plot(silhouette(pam.res3$clustering, dist(data_novia)), col = 2:4, main = "PAM Silhouette: 3 Clusters")
plot(silhouette(pam.res4$clustering, dist(data_novia)), col = 2:5, main = "PAM Silhouette: 4 Clusters")
plot(silhouette(pam.res5$clustering, dist(data_novia)), col = 2:6, main = "PAM Silhouette: 5 Clusters")
plot(silhouette(pam.res$clustering, dist(data_novia)), col = 2:7, main = "PAM Silhouette: 6 Clusters")

# Visualize only the 6 cluster PAM plot

plot5 <- fviz_cluster(pam.res, 
                      ellipse.type = "convex",
                      palette = "ucscgb",
                      repel = TRUE,
                      labelsize = 0,
                      ggtheme = theme_minimal()) + 
                      ggtitle("PAM Cluster Plot: k = 6") +
                      theme(plot.title = element_text(hjust = 0.5))

grid.arrange(plot4, plot5, nrow = 1) #comparing kmeans and PAM 6 cluster models
plot5

## CLARA

data_novia <- subset(data , select = -c(continent, location))
data_novia <- scale(data_novia)
data_novia <- data_novia[-c(64, 87, 157, 164),]
View(data_novia)

clara.res2 <- clara(data_novia, 2, samples = 50, pamLike = TRUE)
clara.res3 <- clara(data_novia, 3, samples = 50, pamLike = TRUE)
clara.res4 <- clara(data_novia, 4, samples = 50, pamLike = TRUE)
clara.res5 <- clara(data_novia, 5, samples = 50, pamLike = TRUE)
clara.res <- clara(data_novia, 6, samples = 50, pamLike = TRUE)

plot(silhouette(clara.res2$clustering, dist(data_novia)), col = 2:3, main = "CLARA Silhouette: 2 Clusters")
plot(silhouette(clara.res3$clustering, dist(data_novia)), col = 2:4, main = "CLARA Silhouette: 3 Clusters")
plot(silhouette(clara.res4$clustering, dist(data_novia)), col = 2:5, main = "CLARA Silhouette: 4 Clusters")
plot(silhouette(clara.res5$clustering, dist(data_novia)), col = 2:6, main = "CLARA Silhouette: 5 Clusters")
plot(silhouette(clara.res$clustering, dist(data_novia)), col = 2:7, main = "CLARA Silhouette: 6 Clusters")

# Visualize only the 6 cluster CLARA plot

plot6 <- fviz_cluster(clara.res, 
                      ellipse.type = "convex",
                      palette = "ucscgb",
                      repel = TRUE,
                      labelsize = 0,
                      ggtheme = theme_minimal()) + 
                      ggtitle("CLARA Cluster Plot: k = 6") +
                      theme(plot.title = element_text(hjust = 0.5))

# comparing kmeans, PAM, CLARA 6 cluster models

plot4
plot5
plot6 

## Principal Component Analysis 

# in this section, variable importance is evaluated and 
# dimensionality reduction conducted.

data_novia <- subset(data , select = -c(continent, location))
data_novia <- scale(data_novia)
data_novia <- data_novia[-c(64, 87, 157, 164),]
View(data_novia)

corrplot(cor(data_novia), type = "upper", method = "color", 
         tl.cex = 0.9, tl.col = "#0033CC")

res.pca <- PCA(data_novia,  graph = FALSE)

# Visualize eigenvalues/variances

plot7 <- fviz_screeplot(res.pca, addlabels = TRUE) + ggtitle("Scree Plot") +
  theme(plot.title = element_text(hjust = 0.5))
plot7

# Extract the results for variables

var <- get_pca_var(res.pca)
dim1 <- var$contrib[,1]
dim2 <- var$contrib[,2]

# Contributions of variables to PC1

plot8 <- fviz_contrib(res.pca, choice = "var", axes = 1, top = 20) + 
  ggtitle("Contribution of variables to Dim-1") +
  theme(plot.title = element_text(hjust = 0.5))
plot8

# Contributions of variables to PC2

plot9 <- fviz_contrib(res.pca, choice = "var", axes = 2, top = 20) + 
  ggtitle("Contribution of variables to Dim-2") +
  theme(plot.title = element_text(hjust = 0.5))
plot9

# Control variable colors using their contributions to the principle axis

plot10 <- fviz_pca_var(res.pca, col.var="contrib",
                       gradient.cols = c("#FFFF00", "#00FF00"),
                       repel = TRUE) + # Avoid text overlapping 
  theme_dark() + ggtitle("Variables - PCA") +
  theme(plot.title = element_text(hjust = 0.5))
plot10

grid.arrange(plot8, plot9, nrow = 1)
plot10
# conduct dimensionality reduction and implement kmeans/PAM/CLARA

# look at removing pop, pop density, diabetes, all smokers
# repeat Kmeans/PAM/CLARA with this new data set and evaluate silhouette widths
# talk about why removing these variables does not yield better results/insigt into the 
# regional patterns of COVID-19. Instead focus on only cases and deaths

#### Hierarchical Clustering ####

# Compute pairewise distance matrices

data_novia <- subset(data , select = -c(continent, location))
data_novia <- scale(data_novia)
data_novia <- data_novia[-c(62, 64, 85, 87, 157, 164),] # excl China, India, Qatar, Singapore, US, Brazil
dist.out <- dist(data_novia, method = "euclidean")

# Hierarchical clustering results (complete linkage)

hc_complete <- hclust(dist.out, method = "complete")

hcd <- as.dendrogram(hc_complete)
plot(hcd, type = "rectangle", ylab = "Height", main = "Complete Linkage")
rect.hclust(hc_complete, k = 2, border = 2:3)
rect.hclust(hc_complete, k = 3, border = 2:4)
rect.hclust(hc_complete, k = 4, border = 2:5)
rect.hclust(hc_complete, k = 5, border = 2:6)
rect.hclust(hc_complete, k = 6, border = 2:7)

# Visualization of hclust

plot(silhouette(cutree(hc_complete, 2),dist.out), main = "Silhouette Plot: 2 Clusters")
plot(silhouette(cutree(hc_complete, 3),dist.out), main = "Silhouette Plot: 3 Clusters")
plot(silhouette(cutree(hc_complete, 4),dist.out), main = "Silhouette Plot: 4 Clusters")
plot(silhouette(cutree(hc_complete, 5),dist.out), main = "Silhouette Plot: 5 Clusters")
plot(silhouette(cutree(hc_complete, 6),dist.out), main = "Silhouette Plot: 6 Clusters")

# Add rectangle around groups

hcd <- as.dendrogram(hc_complete)
plot(hcd, type = "rectangle", ylab = "Height")

# first cluster
plot(hcd, xlim = c(1, 18), ylim = c(1,10))
rect.hclust(hc_complete, k = 6, border = 2:7)

# second cluster
plot(hcd, xlim = c(19, 42), ylim = c(1,12))
rect.hclust(hc_complete, k = 6, border = 2:7)

# third cluster
plot(hcd, xlim = c(43, 54), ylim = c(1,12))
rect.hclust(hc_complete, k = 6, border = 2:7)

# fourth cluster
plot(hcd, xlim = c(56, 118), ylim = c(1,12))
rect.hclust(hc_complete, k = 6, border = 2:7)

# fifth cluster
plot(hcd, xlim = c(119, 124), ylim = c(1,12))
rect.hclust(hc_complete, k = 6, border = 2:7)

# sixth cluster
plot(hcd, xlim = c(124, 170), ylim = c(1,12))
rect.hclust(hc_complete, k = 6, border = 2:7)

# Hierarchical clustering results (single linkage)

hc_single <- hclust(dist.out, method = "single")

# Visualization of hclust

plot(hc_single)

plot(silhouette(cutree(hc_single, 2),dist.out))
plot(silhouette(cutree(hc_single, 3),dist.out))
plot(silhouette(cutree(hc_single, 4),dist.out))
plot(silhouette(cutree(hc_single, 5),dist.out))
plot(silhouette(cutree(hc_single, 6),dist.out))

# Add rectangle around groups

hcd <- as.dendrogram(hc_single)
plot(hcd, type = "rectangle", ylab = "Height")

rect.hclust(hc_single, k = 2, border = 2:3)
rect.hclust(hc_single, k = 3, border = 2:4)
rect.hclust(hc_single, k = 4, border = 2:5)
rect.hclust(hc_single, k = 5, border = 2:6)
rect.hclust(hc_single, k = 6, border = 2:7)

# Hierarchical clustering results (average linkage)

hc_average <- hclust(dist.out, method = "average")

# Visualization of hclust

plot(hc_average)

plot(silhouette(cutree(hc_average, 2),dist.out))
plot(silhouette(cutree(hc_average, 3),dist.out))
plot(silhouette(cutree(hc_average, 4),dist.out))
plot(silhouette(cutree(hc_average, 5),dist.out))
plot(silhouette(cutree(hc_average, 6),dist.out))

# Add rectangle around groups

plot(hc_average)
hcd <- as.dendrogram(hc_average)
plot(hcd, type = "rectangle", ylab = "Height")
rect.hclust(hc_average, k = 2, border = 2:3)
rect.hclust(hc_average, k = 3, border = 2:4)
rect.hclust(hc_average, k = 4, border = 2:5)
rect.hclust(hc_average, k = 5, border = 2:6)
rect.hclust(hc_average, k = 6, border = 2:7)

# Hierarchical clustering results (centroid linkage)

hc_centroid <- hclust(dist.out, method = "centroid")

# Visualization of hclust

plot(hc_centroid)

plot(silhouette(cutree(hc_centroid, 2),dist.out))
plot(silhouette(cutree(hc_centroid, 3),dist.out))
plot(silhouette(cutree(hc_centroid, 4),dist.out))
plot(silhouette(cutree(hc_centroid, 5),dist.out))
plot(silhouette(cutree(hc_centroid, 6),dist.out))

# Add rectangle around groups

plot(hc_centroid)
hcd <- as.dendrogram(hc_average)
plot(hcd, type = "rectangle", ylab = "Height")

rect.hclust(hc_centroid, k = 2, border = 2:3)
rect.hclust(hc_centroid, k = 3, border = 2:4)
rect.hclust(hc_centroid, k = 4, border = 2:5)
rect.hclust(hc_centroid, k = 5, border = 2:6)
rect.hclust(hc_centroid, k = 6, border = 2:7)




