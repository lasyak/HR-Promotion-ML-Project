library(arules)
library(arulesViz)
library(ggplot2)
library(dplyr)
employee <- read.csv("train.csv")
employee <- na.omit(employee)
employee$is_promoted <- factor(ifelse(employee$is_promoted == 1, "Promoted", "Not Promoted"))
employee$education <- factor(employee$education)
employee$gender <- factor(employee$gender)
employee$region <- factor(employee$region)
employee$recruitment_channel <- factor(employee$recruitment_channel)
employee$department <- factor(employee$department)
employee$awards_won. <- factor(employee$awards_won.)

# Apriori analysis
employee$length_of_service <- cut(employee$length_of_service, breaks=c(0,5,10,15,20,25,30,35,40),
                                  labels=c("1-5","6-10","11-15","16-20","21-25","26-30","31-35","36-40"))
employee$avg_training_score <- cut(employee$avg_training_score, breaks=c(30,40,50,60,70,80,90,100),
                                  labels=c("<=40","41-50","51-60","61-70","71-80","81-90","91-100"))
employee$age <- cut(employee$age, breaks=c(10,20,30,40,50,60,70),
                    labels=c("11-20","21-30","31-40","41-50","51-60","61-70"))

employee <- employee[, -which(names(employee) == "employee_id")]

rules <- apriori(employee, parameter = list(supp=0.007, conf=0.22),
                 appearance = list(default="lhs", rhs="is_promoted=Promoted"))
rules <- sort(rules, by="confidence", decreasing=TRUE)
inspect(rules[1:10])
plot(rules)