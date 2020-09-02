---
  title: "Data Analysis of Severe Weather Events"
author: "Michael A. Esparza, PMP, PgMP, CPEM - Data Science Manager (DSM)"
date: "`r format(Sys.time(), '%d %B %Y')`"
output: word_document
subtitle: Most Harmful Event Types To the United States Public Health and Economy
tags:
  - nothing
- nothingness
abstract: The Objective of this Data Analysis of the United States (U.S.) National
Oceanographic and Atmospheric Administration (NOAA) Storm Data Base is to review
the available data and address the following questions in support of Decision     Dominance for those stakeholders involved in preparing and planning for severe  weather.
always_allow_html: yes
---
  ```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Final Data Analysis Project IST 707
### Synopsis
#### The Objective of this Data Analysis of the United States (U.S.) National Oceanographic and Atmospheric Administration's (NOAA's) Storm Data Base is to review the available data and address the following questions in support of Decision Dominance for those stakeholders involved in preparing and planning for severe weather.
#### 1. In the U.S. which severe weather event types are are most harmful to population health?
#### 2. In the U.S. which severe weather event types have the greatest economic consequences?
#### The Raw Data Frame was downloaded from the NOAA Data Base and processed using R. After getting and cleaning the Data Frame, the study was executed to identify Severe Weather Event Types (EVTYPEs) in three categories: Most Fatalities, Most Injuries, and events with the greatest Economic Impact. The results of the data frame examination are then presented in the applicable graphs. Data Processing and Results are explained and elaborated upon as necessary.

### DATA PROCESSING
### 1. Load the Necessary Library
```{r results='hide', message=FALSE, warning=FALSE}
library(factoextra)
library(R.utils)
library(dplyr)
library(ggplot2)
library(cluster)
library(kableExtra)
library(readr)
library(dplyr)
library(magrittr)
library(ggplot2)
library(maps)
library(ggmap)
library(naivebayes)
library(rpart)
library(rpart.plot)
library(caret)
library(ggplot2)
library(naivebayes)
library(randomForest)
library(e1071)
library(arules)
library(arulesViz)
```
### 2. Download the Raw Data Frame.
url <- "https://d396qusza40orc.cloudfront.net/repdata%2Fdata%2FStormData.csv.bz2"
download.file(url, "StormData.csv.bz2")
library(R.utils)

### 3. Read the Data Frame.

setwd("~/IST707")
df <- read.csv("StormData.csv.bz2")
##If perfoming na.omit or complete clases 
##After initially loading dataframe
##Results in an empty dataframe
##Will remove clean dataframe once
##Damage calculations are done.


### 4. Review and Examine a Summary of the Data Frame.

summary(df)
df_temp <- df[complete.cases(df),]
df_temp <- na.omit(df)

### 5.  Determine Health Impacts for the U.S. Population (Fatalities)
### A.  Total Fatalities for each Severe Weather Event Type (EVTYPE). 
library(dplyr)
df.fatalities <- df %>% select(EVTYPE, FATALITIES) %>% group_by(EVTYPE) %>% summarise(total.fatalities = sum(FATALITIES)) %>% arrange(-total.fatalities)
head(df.fatalities, 10)
kable(df.fatalities) %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive", full_width = F, font_size = 16))

### 6. Determine Health Impacts for the U.S. Population (Injuries).
### A. Total Injuries for each Severe Weather Event Type (EVTYPE).
df.injuries <- df %>% select(EVTYPE, INJURIES) %>% group_by(EVTYPE) %>% summarise(total.injuries = sum(INJURIES)) %>% arrange(-total.injuries)
head(df.injuries, 10)
kable(df.injuries) %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive", full_width = F, font_size = 16))

### 7. Determine Impacts of Severe Weather Events on the U.S. Economy.  The Data Frame reveals two types of economic impacts: Property Damage (EVTYPE - PROPDMG) and Crop Damage (EVTYPE - CROPDMG). The Monetary Value of Damage is measured in United States Dollars ($USD) for both of these EVTYPEs. 

df.damage <- df %>% select(EVTYPE, STATE, STATE__, INJURIES, FATALITIES, PROPDMG,PROPDMGEXP,CROPDMG,CROPDMGEXP)
df_temp <- df.damage[complete.cases(df.damage),]
Symbol <- sort(unique(as.character(df.damage$PROPDMGEXP)))
Multiplier <- c(0,0,0,1,10,10,10,10,10,10,10,10,10,10^9,10^2,10^2,10^3,10^6,10^6)
convert.Multiplier <- data.frame(Symbol, Multiplier)

df.damage$Prop.Multiplier <- convert.Multiplier$Multiplier[match(df.damage$PROPDMGEXP, convert.Multiplier$Symbol)]
df.damage$Crop.Multiplier <- convert.Multiplier$Multiplier[match(df.damage$CROPDMGEXP, convert.Multiplier$Symbol)]

df.damage <- df.damage %>% mutate(PROPDMG = PROPDMG*Prop.Multiplier) %>% mutate(CROPDMG = CROPDMG*Crop.Multiplier) %>% mutate(TOTAL.DMG = PROPDMG+CROPDMG)

df.damage.total <- df.damage %>% group_by(EVTYPE) %>% summarize(TOTAL.DMG.EVTYPE = sum(TOTAL.DMG))%>% arrange(-TOTAL.DMG.EVTYPE) 
head(df.damage.total,10)
kable(df.damage.total) %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive", full_width = F, font_size = 16))

### RESULTS
### 1. Health Impacts.
### A. The previously determined health impacts (Fatalities and Injuries) resulting from Severe Weather Events (EVTYPEs) are then plotted graphically to present the data analysis results.
### 2. Top 10 Severe Weather Events (EVTYPEs) causing the most fatalities.

library(ggplot2)
g <- ggplot(df.fatalities[1:10,], aes(x=reorder(EVTYPE, -total.fatalities), y=total.fatalities))+geom_bar(stat="identity", color = "red", fill = "red") + theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1))+ggtitle("Top 10 Events with Highest Total Fatalities") +labs(x="EVENT TYPE", y="Total Fatalities")
g


### 3. Top 10 Severe Weather Events (EVTYPEs) causing the most injuries.

g <- ggplot(df.injuries[1:10,], aes(x=reorder(EVTYPE, -total.injuries), y=total.injuries))+geom_bar(stat="identity", color = "blue", fill = "blue") + theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1))+ggtitle("Top 10 Events with Highest Total Injuries") +labs(x="EVENT TYPE", y="Total Injuries")
g


### 4.  Conclusion: Upon review of the presented graphs, the Severe Weather Event Type (EVTYPE) TORNADO causes the greatest Health Impact to the U.S. Population, in both Fatalities and Injuries.
### RESULTS
### 1. Economic Impact.
### A. The previously determined U.S. Economic impacts, presented in the data frame as Property Damage (PROPDMG) and Crop Damage (CROPDMG), resulting from Severe Weather Events (EVTYPEs), are then plotted graphically to present the data analysis results, with the extent of the damage determined by the amount of monetary loss in $USD.
g <- ggplot(df.damage.total[1:10,], aes(x=reorder(EVTYPE, -TOTAL.DMG.EVTYPE), y=TOTAL.DMG.EVTYPE))+geom_bar(stat="identity", color = "orange", fill = "orange") + theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1))+ggtitle("Top 10 Events with Highest Economic Impact") +labs(x="EVENT TYPE", y="Total Economic Impact ($USD)")
g


model_data <- df.damage[complete.cases(df.damage),]
model_data <- model_data[,-c(7,9:11)]
str(model_data)
model_data$STATE__ <- as.integer(model_data$STATE__)
model_data$PROPDMG <- as.integer(model_data$PROPDMG)
model_data$CROPDMG <- as.integer(model_data$CROPDMG)
model_data$TOTAL.DMG <- as.integer(model_data$TOTAL.DMG)
model_data$INJURIES <- as.integer(model_data$INJURIES)
model_data$FATALITIES <- as.integer(model_data$FATALITIES)
sort(unique(model_data$CROPDMG))

model_data <- model_data[complete.cases(model_data),]

#need incredibly small sample for training data as it kept breaing my machine if it was greater than 10K
smp_size <- floor(0.98 * nrow(model_data))
set.seed(100)
IND <- sample(seq_len(nrow(model_data)), size = smp_size)
trainData <- model_data[-IND,]
testData <- model_data[IND,]


##Decision Tree/Random Forest

##Feature of Random Forest and by abstraction Decision trees.  
##Since there are more than 53 possible labels, Random Forest Fails to run
##And Decision Trees have an incredible long run time (over 6 hours)
##So Random Forest and Decision trees have terrible or no performance 
##when dealing with labels measuring in the hundreds.

label <- trainData$EVTYPE
temp <- trainData[,2:6]
randfor <- randomForest(label~., data=temp, ntree=50)
pred1 <- predict(randfor, test_digit2[,2:785], type=c("class"))

```{r}
treefit <- rpart(trainData$EVTYPE~., data=trainData, method = "class")
randomForest()

#SVM
svm1 <- svm(trainData$EVTYPE~., data=trainData[,3:8], kernel = "linear", cost = 1)
pred2 <- predict(svm1, testData[1:10000,3:8], type = "class")
cf_svm <- confusionMatrix(table(pred2, testData[1:10000,1]))
cf_svm$overall

str(trainData)
#Naive Bayes
##Naive Bayes has the worst accuracy of the available 
nb1 <- naiveBayes(trainData$EVTYPE~., data=trainData[,2:8], laplace = 0.5 )
pred3 <- predict(nb1, testData[1:10000,2:8], type = "class")
cf_nb <- confusionMatrix(table(pred3, testData[1:10000,1]))

cf_nb$overall

#K-means
##While increaing the number of clusters decreases the 
##total value ot tot.withinss
sort(unique(trainData$EVTYPE))
x <- scale(trainData[,3:8])

kclus <- kmeans(x, 30, nstart = 20)
kclus$cluster
length(kclus$cluster)
kclus$tot.withinss
View(x[1:10,])
fviz_cluster(kclus, data = x)

treefit <- rpart(trainData4$EVTYPE~., trainData4, method = "class")
pred1 <- predict(treefit, testData[,2:8] , type = "class")
cf_mat <- confusionMatrix(table(pred1, testData$EVTYPE))


##associate rule 

assoc_train <- trainData
min_propdmg <- min(assoc_train$PROPDMG)
max_propdmg <- max(assoc_train$PROPDMG)
seq(min_propdmg, max_propdmg, width_prop)
width_prop <- (max_propdmg - min_propdmg)/3

min_cropdmg <- min(assoc_train$CROPDMG)
max_cropdmg <- max(assoc_train$CROPDMG)
seq(min_propdmg, max_propdmg, width_prop)
width_crop <- (max_cropdmg - min_cropdmg)/3


min_totldmg <- min(assoc_train$TOTAL.DMG)
max_totldmg <- max(assoc_train$TOTAL.DMG)
seq(min_propdmg, max_propdmg, width_prop)
width_totl <- (max_totldmg - min_totldmg)/3

min_inj <- min(assoc_train$INJURIES)
max_inj <- max(assoc_train$INJURIES)
seq(min_propdmg, max_propdmg, width_prop)
width_inj <- (max_inj - min_inj)/3

min_fatl <- min(assoc_train$FATALITIES)
max_fatl <- max(assoc_train$FATALITIES)
seq(min_propdmg, max_propdmg, width_prop)
width_fatl <- (max_fatl - min_fatl)/3

assoc_train$INJURIES <- cut(assoc_train$INJURIES, breaks = seq(min_inj, max_inj, width_inj))
assoc_train$FATALITIES <- cut(assoc_train$FATALITIES, breaks = seq(min_fatl, max_fatl, width_fatl))
assoc_train$PROPDMG <- cut(assoc_train$PROPDMG, breaks = seq(min_propdmg, max_propdmg, width_prop))
assoc_train$CROPDMG <- cut(assoc_train$CROPDMG, breaks = seq(min_cropdmg, max_cropdmg, width_crop))
assoc_train$TOTAL.DMG <- cut(assoc_train$TOTAL.DMG, breaks = seq(min_totldmg, max_totldmg, width_totl))

assoc_train <- assoc_train[,-c(3)]

rules_l <- apriori(assoc_train, parameter = list(supp = 0.06, conf = 0.05))
rules_l<-sort(rules_l, decreasing=TRUE,by="confidence")
inspect(rules_l[1:10])
plot(rules_l[1:10],method = "graph", interactive =FALSE,shading=NA)





### 2. Conclusion: Upon review of the presented graph, the Severe Weather Event Type (EVTYPE) Flood (Flooding) is the cause of the greatest economic (monetary) loss; and thus the greatest impact on the U.S. Economy.