library(ggplot2)
library(caret)
library(rpart.plot)
library(e1071)
library(neuralnet)

emp_per <- read.csv("INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.csv")
colnames(emp_per)

emp_per$ï..EmpNumber <- NULL
emp_per$EmpEducationLevel = as.ordered(emp_per$EmpEducationLevel)
emp_per$EmpEnvironmentSatisfaction = as.ordered(emp_per$EmpEnvironmentSatisfaction)
emp_per$EmpJobInvolvement = as.ordered(emp_per$EmpJobInvolvement)
emp_per$EmpJobSatisfaction = as.ordered(emp_per$EmpJobSatisfaction)
emp_per$PerformanceRating = as.ordered(emp_per$PerformanceRating)
emp_per$EmpRelationshipSatisfaction = as.ordered(emp_per$EmpRelationshipSatisfaction)
emp_per$EmpWorkLifeBalance = as.ordered(emp_per$EmpWorkLifeBalance)
emp_per$EmpJobLevel = as.ordered(emp_per$EmpJobLevel)
#For Tableau
levels(emp_per$EmpEducationLevel)
emp_per$EmpEducationLevel = ordered(emp_per$EmpEducationLevel,
    levels = c(1, 2, 3, 4, 5),
    labels = c('Below College', 'College', 'Bachelor', 'Master', 'Doctor'))

levels(emp_per$EmpEnvironmentSatisfaction)
emp_per$EmpEnvironmentSatisfaction = ordered(emp_per$EmpEnvironmentSatisfaction,
    levels = c(1, 2, 3, 4),
    labels = c('Low', 'Medium', 'High', 'Very High'))

levels(emp_per$EmpJobInvolvement)
emp_per$EmpJobInvolvement = ordered(emp_per$EmpJobInvolvement,
    levels = c(1, 2, 3, 4),
    labels = c('Low', 'Medium', 'High', 'Very High'))

levels(emp_per$EmpJobSatisfaction)
emp_per$EmpJobSatisfaction = ordered(emp_per$EmpJobSatisfaction,
    levels = c(1, 2, 3, 4),
    labels = c('Low', 'Medium', 'High', 'Very High'))

levels(emp_per$PerformanceRating)
emp_per$PerformanceRating = ordered(emp_per$PerformanceRating,
    levels = c(2, 3, 4),
    labels = c('Good', 'Excellent', 'Outstanding'))

levels(emp_per$EmpRelationshipSatisfaction)
emp_per$EmpRelationshipSatisfaction = ordered(emp_per$EmpRelationshipSatisfaction,
    levels = c(1, 2, 3, 4),
    labels = c('Bad', 'Good', 'Better', 'Best'))

levels(emp_per$EmpWorkLifeBalance)
emp_per$EmpWorkLifeBalance = ordered(emp_per$EmpWorkLifeBalance,
    levels = c(1, 2, 3, 4),
    labels = c('Bad', 'Good', 'Better', 'Best'))
#For Tableau
write.csv(emp_per, "emp_per.csv")

# Checking significance
emp_mod <- glm(PerformanceRating ~ OverTime, family = "binomial", data = emp_per)
anova(emp_mod, test = "Chisq")
emp_mod <- glm(PerformanceRating ~ EmpJobLevel, family = "binomial", data = emp_per)
anova(emp_mod, test = "Chisq")
emp_mod <- glm(PerformanceRating ~ EmpJobRole, family = "binomial", data = emp_per)
anova(emp_mod, test = "Chisq")
emp_mod <- glm(PerformanceRating ~ EmpDepartment, family = "binomial", data = emp_per)
anova(emp_mod, test = "Chisq")
emp_mod <- glm(PerformanceRating ~ EmpEnvironmentSatisfaction, family = "binomial", data = emp_per)
anova(emp_mod, test = "Chisq")

#Normalization
qplot(sqrt(YearsWithCurrManager), data = emp_per, bins = 10)
qplot(sqrt(ExperienceYearsAtThisCompany), data = emp_per, bins = 5)
qplot(sqrt(ExperienceYearsInCurrentRole), data = emp_per, bins = 5)
qplot(sqrt(TotalWorkExperienceInYears), data = emp_per, bins = 10)
qplot(sqrt(emp_per$YearsSinceLastPromotion), bins = 7)

# Checking Significance
emp_mod_c <- glm(PerformanceRating ~ sqrt(ExperienceYearsAtThisCompany), family = "binomial", data = emp_per)
summary(emp_mod_c)
emp_mod_c <- glm(PerformanceRating ~ sqrt(ExperienceYearsInCurrentRole), family = "binomial", data = emp_per)
summary(emp_mod_c)
emp_mod_c <- glm(PerformanceRating ~ sqrt(TotalWorkExperienceInYears), family = "binomial", data = emp_per)
summary(emp_mod_c)
emp_mod_c <- glm(PerformanceRating ~ sqrt(YearsSinceLastPromotion), family = "binomial", data = emp_per)
summary(emp_mod_c)
emp_mod_c <- glm(PerformanceRating ~ sqrt(YearsWithCurrManager), family = "binomial", data = emp_per)
summary(emp_mod_c)

# Necessary Variables to another dataframe
emp_per_modf <- data.frame('PerformanceRating' = emp_per$PerformanceRating,
                           'OverTime' = emp_per$OverTime,
                           'EmpJobLevel' = emp_per$EmpJobLevel,
                           'EmpJobRole' = emp_per$EmpJobRole,
                           'EmpDepartment' = emp_per$EmpDepartment,
                           'EmpEnvironmentSatisfaction' = emp_per$EmpEnvironmentSatisfaction,
                           'ExperienceYearsAtThisCompany' = sqrt(emp_per$ExperienceYearsAtThisCompany),
                           'ExperienceYearsInCurrentRole' = sqrt(emp_per$ExperienceYearsInCurrentRole),
                           'TotalWorkExperienceInYears' = sqrt(emp_per$TotalWorkExperienceInYears),
                           'YearsSinceLastPromotion' = sqrt(emp_per$YearsSinceLastPromotion),
                           'YearsWithCurrManager' = sqrt(emp_per$YearsWithCurrManager))

# Data Split
set.seed(1)
intrain <- createDataPartition(y = emp_per_modf$PerformanceRating, p = 0.75, list = FALSE)
train <- emp_per_modf[intrain,]
test <- emp_per_modf[-intrain,]

# k fold
trainCtrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

set.seed(2)
dtree <- train(PerformanceRating ~ ., method = "rpart", parms = list(split = "information"), trControl = trainCtrl, tuneLength = 10, data = train)
prp(dtree$finalModel)
test_pred <- predict(dtree, newdata = test)
confusionMatrix(test_pred, test$PerformanceRating) # Accuracy 93.31%

set.seed(2)
randf <- train(PerformanceRating ~ ., method = "rf", trControl = trainCtrl, tuneLength = 10, data = train, prox = TRUE, allowParallel = TRUE)
test_pred <- predict(randf, newdata = test)
confusionMatrix(test_pred, test$PerformanceRating) # Accuracy 94.31%

set.seed(2)
svmt <- train(PerformanceRating ~ . - EmpJobRole, method = "svmLinear", preProcess = c("center", "scale"), trControl = trainCtrl,
              tuneLength = 10, data = train)
test_pred <- predict(svmt, newdata = test)
confusionMatrix(test_pred, test$PerformanceRating) # Accuracy 86.29%

# For Neural Network
emp_per$PerformanceRating = ordered(emp_per$PerformanceRating,
                                    levels = c('Good', 'Excellent', 'Outstanding'),
                                    labels = c(2,3,4))

emp_per_fd <- data.frame('PerformanceRating' = as.numeric(emp_per$PerformanceRating),
                           'OverTime' = as.character(emp_per$OverTime),
                           'EmpJobLevel' = as.character(emp_per$EmpJobLevel),
                           'EmpJobRole' = as.character(emp_per$EmpJobRole),
                           'EmpDepartment' = as.character(emp_per$EmpDepartment),
                           'EmpEnvironmentSatisfaction' = as.character(emp_per$EmpEnvironmentSatisfaction),
                           'ExperienceYearsAtThisCompany' = sqrt(emp_per$ExperienceYearsAtThisCompany),
                           'ExperienceYearsInCurrentRole' = sqrt(emp_per$ExperienceYearsInCurrentRole),
                           'TotalWorkExperienceInYears' = sqrt(emp_per$TotalWorkExperienceInYears),
                           'YearsSinceLastPromotion' = sqrt(emp_per$YearsSinceLastPromotion),
                           'YearsWithCurrManager' = sqrt(emp_per$YearsWithCurrManager))

dum <- dummyVars("~ .", data = emp_per_fd, fullRank = TRUE, levelsOnly = FALSE)
dum_emp <- data.frame(predict(dum, newdata = emp_per_fd))

intrain2 <- createDataPartition(y = dum_emp$PerformanceRating, p = 0.75, list = FALSE)
train2 <- dum_emp[intrain,]
test2 <- dum_emp[-intrain,]

set.seed(2)
my.grid <- expand.grid(decay = seq(0.1, 0.5, 0.1), size = seq(1, 10 ,1))
neunet <- train(PerformanceRating ~ ., method = "nnet", tuneGrid = my.grid, trace = F, linout = T, data = train2)
test_pred <- predict(neunet, test2)
confusionMatrix(round(test_pred), test2$PerformanceRating)
