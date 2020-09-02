install.packages("rworldmap")
install.packages("rattle")
install.packages("RGtk2")
install.packages("plotly")
#load necessary libraries
library(maps)
library(ggplot2)
library(ggmap)
library(mapproj)
library(reshape2)
library(rworldmap)
library(Rcmdr)
library(rattle)
library(RGtk2)
library(sqldf)
library(RCurl)
library(RJSONIO)
library(jsonlite)
library(bitops)
library(plotly)
#load data from files
accidents_2017 <- read.csv("C:\\Users\\aaron\\Downloads\\accidents_2017.csv\\accidents_2017.csv")
str(accidents_2017)
x <- accidents_2017$Mild.injuries + accidents_2017$Serious.injuries
accidents_2017$Tot.injuries <- x
View(accidents_2017)

#loop throuugh datafram to find the numeric elements of the matrix and print summary
for(i in 1:ncol(accidents_2017)){
  if(is.numeric(accidents_2017[,i])){
    cat("Summary of Vector: ", colnames(accidents_2017)[i],"\n")
    print(summary(accidents_2017[,i]))
  }
}
#renaming column header for sql statement later on
colnames(accidents_2017)[2] <-"District"
colnames(accidents_2017)[3] <-"Neighborhood"
colnames(accidents_2017)[10] <-"POD"
colnames(accidents_2017)[11] <- "Mild"
colnames(accidents_2017)[12] <- "Serious"
colnames(accidents_2017)[14] <- "Vehicles"
colnames(accidents_2017)[17] <- "Total"
library(plotly)
accidents_2017$Mild.injuries
accidents_2017$Serious.injuries
accidents_2017$Tot.injuries
accidents_2017$Vehicles.involved
str(accidents_2017)
#create date field for dataset
accidents_2017$date <- paste("2017", match(accidents_2017$Month, month.name), accidents_2017$Day, sep="-") 
accidents_2017$mon <- match(accidents_2017$Month, month.name)
accidents_2017$date <- as.Date(accidents_2017$date, "%Y-%m-%d")
#Base SQL to create aggregate data ####################################################

district_accidents  <- sqldf("Select * from accidents_2017 where District = 'Horta-Guinardó'")
district <- sqldf("select distinct District from accidents_2017 ")
neighborhood <- sqldf("select distinct Neighborhood from accidents_2017 ")
serious_accidents <- sqldf("select * from accidents_2017 where Serious > 0")
mild_accidents <- sqldf("select * from accidents_2017 where Mild > 0")
vehicles_accidents <- sqldf("select * from accidents_2017 where Vehicles > 0")
accidents_wkMn <- sqldf("select Weekday, Month, District, sum(Mild) tot_mild, sum(Serious) tot_serious
                        from accidents_2017 group by Weekday, Month, District
                        order by tot_mild, tot_serious desc")
View(accidents_wkMn)
summary(Linear.Model.5)

View(sum_accidents)
accidents_2017$DistrictID <- match(accidents_2017$District, district$District)
accidents_2017$neighborhoodID <- match(accidents_2017$Neighborhood, neighborhood$Neighborhood)
sum_accidents <- sqldf("select District, Neighborhood, Month, POD, sum(Mild) tot_mild, sum(Serious) tot_serious, 
                       sum(Vehicles) tot_veh, sum(Total) tot_totl from accidents_2017 
                       group by District, Neighborhood,  Month, POD")
####################################################################################################################



accidents_2017 <- accidents_2017[order(accidents_2017$date),]

weekday  <- c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")

accidents_2017$WeekdayID <- match(accidents_2017$Weekday, weekday)

View(accidents_2017)

PODX <- c("Morning", "Afternoon", "Night")
accidents_2017$PODID <- match(accidents_2017$POD, PODX)

#Create Heat map

range(accidents_2017$Serious)
serious_accidents <- sqldf("select * from accidents_2017 where Serious > 0")
savg <- mean(serious_accidents$Serious)
accidents_2017$SeriousLikelihood <- ifelse(accidents_2017$Serious<savg, 0, 1)
median(accidents_2017$Serious)
sum_accidents$MonthID <- match(sum_accidents$Month, month.name)

sum_accidents <- sum_accidents[order(sum_accidents$MonthID),]

p <- plot_ly(x=sum_accidents$District, y=sum_accidents$Month, z=sum_accidents$vehicles,  type="heatmap") %>%
  layout(title="Accidents Totals by Month and District")
#Create pie chart for total accidents by district
plot_ly(accidents_2017, labels = ~accidents_2017$District, values = ~accidents_2017$Total, type = 'pie') %>%
        layout(title= 'Total Number of accidents in Barcelona by District in 2017')
plot_ly(x=district_accidents$date, y=district_accidents$Total, type="scatter")
plot_ly

p.1 <- ggplot(data=accidents_2017, aes(x=accidents_2017$mon, y=accidents_2017$Total, group=accidents_2017$District=="Horta-Guinardó", colour=accidents_2017$District=="Horta-Guinardó")) +
      geom_line(aes(linetype=accidents_2017$District=="Horta-Guinardó"), size=1)

mean(sample(accidents_2017$Mild, 100, replace=TRUE))
  replicate(10,mean(replicate(1000, mean(sample(accidents_2017$Vehicles, 10000, replace=TRUE)))))
  mean(accidents_2017$Mild)
  mean(accidents_2017$Serious)
  mean(accidents_2017$Vehicles)

mean(vehicles_accidents$Vehicles)

View(accidents_2017)


accidents_2017$Severity <- ifelse(accidents_2017$Serious<mean(serious_accidents$Serious) , 0, 1)
accidents_2017$Sevr_Vec <- ifelse(accidents_2017$Vehicles<mean(vehicles_accidents$Vehicles) , 0, 1)


plot(x=sum_accidents$sum_vic, y=sum_accidents$sum_mild_inj)
dis_sum <- sum_accidents[which(sum_accidents$District=="Les Corts"),]
g1 <-ggplot(sum_accidents)
g1 + geom_point(aes(x=Month, y=vehicles, color=sum_tot) )

lm.1 <- lm(Vehicles~ mon + DistrictID + neighborhoodID + PODNUM + Hour, data=accidents_2017)
vif(lm.1)
summary(lm.1)
library(Rcmdr)
str(Credit)
View(Credit)
install.packages("neuralnet")
library(neuralnet)
trainingdata <-Credit[1:800,]
testingdata <- Credit[801:2000,]
head(trainingdata)
creditnet <- neuralnet(default10yr ~ LTI + age, trainingdata, hidden=4,lifesign="minimal", linear.output = FALSE, threshold=0.1)
plot(creditnet, rep="best")
temp_test <- subset(testingdata, select = c("LTI", "age"))
creditnet.results <- compute(creditnet, temp_test)
results <- data.frame(actual=testingdata$default10yr, prediction=creditnet$response)

stat_lookup <- function(x){
  return(paste("Mean: ", mean(x), "Range: ", range(x), "Median: ", median(x)))
}
stat_lookup(accidents_2017$Mild)
sd(accidents_2017$Mild)

mean(accidents_2017$Sevr_Vec)
mean(accidents_2017$Severity)
##First Neural Network
creditnet <- neuralnet(PersonalLoan ~ CCAvg + CDAccount + CreditCard + Education + Family + Income + Online + SecuritiesAccount, Credit, hidden=5,lifesign="minimal", linear.output = FALSE, threshold=0.1)
plot(creditnet, rep="best")
creditnet.results <- compute(creditnet, Credit)
results <- data.frame(actual=Credit$PersonalLoan,prediction=creditnet.results$net.result)
results$prediction <- round(results$prediction)
View(results)
summary(creditnet)
View(creditnet$result.matrix)

p.3 <- ggplot(accidents_2017, aes(x=date))
p.3 + geom_line(aes(y=Mild, group=District, color="blue")) + geom_line(aes(y=Serious, group=District, color="red"))

glm(formula = PersonalLoan ~ Age + CCAvg + CDAccount + CreditCard + 
      Education + Experience + Family + Income + Mortgage + Online + 
      SecuritiesAccount, family = binomial(logit), data = Credit)

Linear.Model.5 <- lm(formula = Mild ~ Hour + mon + Vehicles, data = accidents_2017)

summary(Linear.Model.5)
results.2 <- predict(Linear.Model.5, accidents_2017)
mild <- accidents_2017$Mild
compareVals <- data.frame(mild, results, results.2)
compareVals$diff <- abs(compareVals$mild - compareVals$results)
mean(compareVals$diff)*mean(compareVals$diff)
compareVals$diff <- abs(compareVals$accidents_2017.Mild, compareVals$predict.LinearModel.4..accidents_2017)
mean(abs(compareVals$mild - compareVals$results))
hist(accidents_2017$Mild)
hist(accidents_2017$Serious)
hist(accidents_2017$Vehicles)

##Secon Neural Network
creditnet.2 <- neuralnet(PersonalLoan ~ CCAvg + CDAccount + CreditCard + Education + Family + Income + Online + SecuritiesAccount, Credit, hidden=5,lifesign="minimal", linear.output = FALSE, threshold=0.1)
plot(creditnet, rep="best")
creditnet.results <- compute(creditnet, Credit)
results <- data.frame(actual=Credit$PersonalLoan,prediction=creditnet.results$net.result)
results$prediction <- round(results$prediction)
View(results)

summary(LinearModel.9)

linear.m.2 <- lm(formula = Mild ~ Latitude + Vehicles, data = accidents_2017[1:round(dim(accidents_2017)[1]/2),])
 
test_accident <- accidents_2017[round(dim(accidents_2017)[1]/2):dim(accidents_2017)[1],]
test_accident$predMild <- predict(linear.m.1, accidents_2017[round(dim(accidents_2017)[1]/2):dim(accidents_2017)[1],])

compareVals <- data.frame(mild = test_accident$Mild, pred = test_accident$predMild, diff_mild = abs(test_accident$Mild - test_accident$predMild))
mean(compareVals$diff_mild
dim(accidents_2017)[1]

summary(linear.m.2)

