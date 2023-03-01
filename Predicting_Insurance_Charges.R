#-------------------------------------------------------------------------------
## STAT 628 
## Final Project - Predicting Insurance Charges
## Bhattarai Aditee, Chilulveru Arun Kumar, and Natrajan Hemnath 

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
######### Preliminary Code #########
#-------------------------------------------------------------------------------
## Clear your work space
# If your work space is NOT empty, run:

rm(list=ls()) #to clear it


options(max.print = 3000)
#-------------------------------------------------------------------------------

##### Set Working Directory ##### 

setwd("/Users/arunchiluveru/Documents/AppliedRegression/FinalProject")

#-------------------------------------------------------------------------------

##### Load libraries ##### 

library(car)
library(MPV)
library(MASS)
library(leaps)
library(caret)


#-------------------------------------------------------------------------------
##### Load data  #####
#-------------------------------------------------------------------------------

df<- read.csv("insurance.csv")

#-------------------------------------------------------------------------------
######### Main Code #########
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
### 1. Data Overview ####
#-------------------------------------------------------------------------------


# In this project we will use the "insurance.csv" file. Here our research 
# objective is to identify whether factors such as age, gender, body mass index,
# number of children, smoker status, and region of residence affect insurance 
# charges. To test this, we have collected insurance charges data from Kaggle. 
# We plan to apply regression methods to identify the impact of the mentioned 
# factors on insurance charges.  This dataset contains 1338 records and
# 7 columns. The variables in the dataset include:

#age - The age (in years) of the primary beneficiary
#sex - The gender of the customer 
#bmi - Body Mass Index of the customer 
#children - Number of children covered by insurance 
#smoker - Whether the customer smokes or not 
#region - Region of the customer’s residence in the US 
#charges (target variable) - Individual medical costs billed by health insurance 

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#### 2. Data & Variables ####
#-------------------------------------------------------------------------------

##### 2.0 Preview the data #####
#-------------------------------------------------------------------------------

head(x = df )
tail(x = df)

##### 2.1 Describing the data #####
#-------------------------------------------------------------------------------
#  Structure of the dataset object
str(object = df)

#  Dimensions of dataset which gives rows and columns
dim(df)

#  column names of dataset
colnames(df)

#-------------------------------------------------------------------------------
### 4. Data Quality and Pre-processing ####
#-------------------------------------------------------------------------------

##### 4.0 Duplicate Values #####
#-------------------------------------------------------------------------------

# We are checking for duplicate values
df[duplicated(df),]#shows there is one duplicated value
df <- df[!duplicated(df),]# removing the 1 duplicate row.


##### 4.1 Missing Values #####
#-------------------------------------------------------------------------------

#  We are checking for missing values
any(is.na(df))#shows no missing values



#-------------------------------------------------------------------------------
### 5. Model and Variable Selection ####
#-------------------------------------------------------------------------------


###### 5.1 Initial Model Analysis ######
#-------------------------------------------------------------------------------

reg<-lm(charges ~ age+sex+bmi+children+smoker+region, df)
summary(reg)

#Checking for multicollinearity
vif(reg)

####Converting independent variables into factors

#Converting male to 1 and female to 0 in variable sex
df$sex<-ifelse(df$sex=="male",1,0)

#Converting yes to 1 and no to 0 in variable smoker
df$smoker<-ifelse(df$smoker=="yes",1,0)

#Converting region to binary
df$Regionsouthwest<-ifelse(df$region=="southwest",1,0)
df$Regionnorthwest<-ifelse(df$region=="northwest",1,0)
df$Regionnortheast<-ifelse(df$region=="northeast",1,0)
#Remaining region is: Southeast

head(df)

#Removing Region after creating binary variables
df <- df[,-c(6)]
head(df)

#Full model with binary variables
fmbv<-lm(charges ~ age+sex+bmi+children+smoker+Regionsouthwest+Regionnortheast+
           Regionnorthwest, df)
summary(fmbv)

#Initial Variable selection with binary variables
###################################################################
#Backward variable selection with full model using t-tests
bvsfm<-fmbv
summary(bvsfm)#Model is overall significant. Adj R2 is 0.7492
vif(bvsfm)#No issues with multi-collinearity

#Removing Regionsouthwest because of high p-value
bvsfm1<-lm(charges ~ age+sex+bmi+children+smoker+Regionnortheast+
           Regionnorthwest,df)
summary(bvsfm1)#Model is overall significant. Adj R2 is 0.7494
vif(bvsfm1)#No issues with multi-collinearity

#Removing sex because of high p-value
bvsfm2<-lm(charges ~ age+bmi+children+smoker+Regionnortheast+
           Regionnorthwest,df)
summary(bvsfm2)#Model is overall significant. Adj R2 is 0.7496
vif(bvsfm2)#No issues with multi-collinearity

#Removing Regionnorthwest because of high p-value
bvsfm3<-lm(charges ~ age+bmi+children+smoker+Regionnortheast,df)
summary(bvsfm3)#Model is overall significant. Adj R2 is 0.7493
vif(bvsfm3)#No issues with multi-collinearity

#final model with  binary variables using Backward stepwise variable selection is
# Model1
#All variables in this model are now significant and there are no multi-collinearity 
#issues.The variables selected from backward stepwise using t-test are age, bmi, 
#children, smoker,Regionnortheast. Adj R2 0.7493.
Model1<-bvsfm3
summary(Model1)#Model is overall significant. Adj R2 is 0.7493
vif(Model1)#No issues with multi-collinearity
PRESS(Model1)#49455076935
AIC(Model1)#27092.83
BIC(Model1)#27129.22
###################################################################

#Stepwise variable selection for initial model using F-tests
###################################################################

#Forward

#Empty model
embv<-lm(df$charges~1,df)
fsfmbv<-lm(df$charges~.,df)
#Forward stepwise full model

add1(embv,scope=fsfmbv,test="F")

#Adding/dropping will be done based on high F-value and low p-value

#Adding smoker F-value (2175.7369) and p-value(2.2e-16)
add1(update(embv,~.+smoker),scope=fsfmbv,test="F")
#Adding age F-value (485.5766) and p-value(2.2e-16)
drop1(update(embv,~.+age+smoker),test="F") 

add1(update(embv,~.+age+smoker),scope=fsfmbv,test="F")
#Adding bmi F-value (137.6772) and p-value(2.2e-16)
drop1(update(embv,~.+bmi+age+smoker),test="F")

add1(update(embv,~.+bmi+age+smoker),scope=fsfmbv,test="F")
#Adding Children F-value (11.7674) and p-value(0.0006212)
drop1(update(embv,~.+children+bmi+age+smoker),test="F")

add1(update(embv,~.+children+bmi+age+smoker),scope=fsfmbv,test="F")
#Adding Regionnortheast F-value (6.8796) and p-value(0.008818)
drop1(update(embv,~.+Regionnortheast+children+bmi+age+smoker),test="F")

add1(update(embv,~.+Regionnortheast+children+bmi+age+smoker),scope=fsfmbv,test="F")

#Final model after forward stepwise variable selection with binary variables using 
#F-test is 
#df$charges ~ Regionnortheast + children + bmi + age + smoker

#This model is similar to model1 
fsvsfmbv<- lm(charges ~ Regionnortheast + children + bmi + age + smoker, df) 
summary(fsvsfmbv)#Model is overall significant. Adj R2 is 0.7493
vif(fsvsfmbv)#No issues with multi-collinearity
###################################################################

#Backward
drop1(fsfmbv,test="F")

#Removing Regionsouthwest because of high p-value
drop1(update(fsfmbv,~.-Regionsouthwest),test="F")
#Removing sex because of high p-value
add1(update(fsfmbv,~.-Regionsouthwest-sex),scope=fsfmbv,test="F")

drop1(update(fsfmbv,~.-Regionsouthwest-sex),test="F")
#Removing Regionnorthwest because of high p-value
add1(update(fsfmbv,~.-Regionnorthwest-Regionsouthwest-sex),scope=fsfmbv,test="F")


drop1(update(fsfmbv,~.-Regionnorthwest-Regionsouthwest-sex),test="F")

#Final model after backward stepwise variable selection with binary variables 
#using F-test is
# charges ~ age + bmi + children + smoker + Regionnortheast 


#This model is similar to model1 
bsvsfmbv<- lm(charges ~ age + bmi + children + smoker + Regionnortheast, df) 
summary(bsvsfmbv)#Model is overall significant. Adj R2 is 0.7493
vif(bsvsfmbv)#No issues with multi-collinearity
################################################################

#Regsubsets for initial model with binary variables using BIC
################################################################

bicfmbv<-regsubsets(df[,c(1:5, 7:9)],df[,6],method="exhaustive",nbest=3, nvmax = 9)
bicfmbvsum<-summary(bicfmbv)
bicfmbvsum
bicfmbvsum$rsq #0.75074826[7(1)], 0.75074347[8(1)], 0.75071991[7(2)]
bicfmbvsum$adjr2  #0.74959053[6(1)], 0.74943061[7(1)], 0.74940692[7(2)]
bicfmbvsum$bic  # -1814.95600[4(1)], -1811.68148[5(1)], -1810.39446[3(1)]

#Following models are taken into consideration based on low BIC values

#we will reject this model when compared to model1 because of high PRESS statistic
#AIC and BIC values in comparison to model1
coef(bicfmbv,10)
lbicm1<-lm(charges ~ age + bmi + children + smoker, df) #4(1)
summary(lbicm1)#Model is overall significant. Adj R2 is 0.7488
vif(lbicm1)#No issues with multi-collinearity
PRESS(lbicm1)#49522277920
AIC(lbicm1)#27094.75
BIC(lbicm1)#27125.94

#this model is similar to model1
coef(bicfmbv,13)
lbicm2<-lm(charges ~ age + bmi + children + smoker + Regionnortheast, df) #5(1)
summary(lbicm2)#Model is overall significant. Adj R2 is 0.7493
vif(lbicm2)#No issues with multi-collinearity
PRESS(lbicm2)#49455076935
AIC(lbicm2)#27092.83
BIC(lbicm2)#27129.22

#we will reject this model when compared to model1 because of high PRESS statistic
#AIC and BIC values in comparison to model1
coef(bicfmbv,7)
lbicm3<-lm(charges ~ age + bmi + smoker, df) #3(1)
summary(lbicm3)#Model is overall significant. Adj R2 is 0.7467
vif(lbicm3)#No issues with multi-collinearity
PRESS(lbicm3)#49890252111
AIC(lbicm3)#27104.51
BIC(lbicm3)#27130.5

################################################################

#stepAIC for Initial model with binary variables using AIC
################################################################

#forward AIC
stepAIC(embv,direction='forward',scope=list(lower=embv,upper=fsfmbv))

#Final model after using forward stepAIC for variable selection is
#AIC value is 23296.15
# df$charges ~ smoker + age + bmi + children + Regionnortheast + 
# Regionnorthwest


#both AIC
stepAIC(embv,direction='both',scope=list(lower=embv,upper=fsfmbv))
#Final model after using both stepAIC for variable selection is
#AIC value is 23296.15
# df$charges ~ smoker + age + bmi + children + Regionnortheast + 
# Regionnorthwest


#backward AIC
stepAIC(fsfmbv,direction='backward',scope=list(lower=embv,upper=fsfmbv))
#Final model after using backward stepAIC for variable selection is
#AIC value is 23296.15
# charges ~ age + bmi + children + smoker + Regionnortheast + Regionnorthwest

#All the models are same. The final model is the following
Model2<-lm(charges ~ age + bmi + children + smoker + Regionnortheast + 
             Regionnorthwest, df)
summary(Model2)#Model is overall significant. Adj R2 is 0.7496
vif(Model2)#No issues with multi-collinearity
PRESS(Model2)#49440193287
AIC(Model2)#27092.39
BIC(Model2)#27133.98

 

#-------------------------------------------------------------------------------
###### 5.2 Model Analysis after Adding Interactions ######
#-------------------------------------------------------------------------------

#Age*bmi, Age*smoker, sex*smoker, Age*children, sex*bmi, 
#smoker*Regionsouthwest, smoker*Regionnortheast, smoker*Regionnorthwest. 
int_df<-df
head(int_df)

I1 <- int_df$age*int_df$bmi
I2 <- int_df$age*int_df$smoker
I3 <- int_df$age*int_df$children
I4 <- int_df$sex*int_df$smoker
I5 <- int_df$sex*int_df$bmi
I6 <- int_df$smoker*int_df$Regionsouthwest
I7 <- int_df$smoker*int_df$Regionnorthwest
I8 <- int_df$smoker*int_df$Regionnortheast

##Incorporating interactions into dataset

int_df$I1 <- int_df$age*int_df$bmi
int_df$I2 <- int_df$age*int_df$smoker
int_df$I3 <- int_df$age*int_df$children
int_df$I4 <- int_df$sex*int_df$smoker
int_df$I5 <- int_df$sex*int_df$bmi
int_df$I6 <- int_df$smoker*int_df$Regionsouthwest
int_df$I7 <- int_df$smoker*int_df$Regionnorthwest
int_df$I8 <- int_df$smoker*int_df$Regionnortheast

head(int_df)
dim(int_df)#1337*17

#Full model with interactions
fmi<-lm(charges ~ age+sex+bmi+children+smoker+Regionsouthwest+Regionnortheast+
           Regionnorthwest+I1+I2+I3+I4+I5+I6+I7+I8, int_df)
summary(fmi)#Model is overall significant. Adj R2 is 0.756
# there are multi-collinearity issues as vif is greater than 5 for few variables 
# which  is expected for the full model. We will remove irrelevant variables in 
# further process
vif(fmi)



#The following are the two possible final model after removing unwanted variables 
#during initial selection process and adding interactions to it.

Model3<-lm(charges ~ age+bmi+children+smoker+Regionnortheast+I1+I2+I3+I4+I5+I6+I7+I8, int_df)
summary(Model3)#Model is overall significant. Adj R2 is 0.756
vif(Model3)
PRESS(Model3)#49178048766
AIC(Model3)#27074.12
BIC(Model3)#27152.09

Model4<-lm(charges ~ age + bmi + children + smoker + Regionnortheast + 
             Regionnorthwest+I1+I2+I3+I4+I5+I6+I7+I8, int_df)
summary(Model4)#Model is overall significant. Adj R2 is 0.756
vif(Model4)
PRESS(Model4)#48815060515
AIC(Model4)#27065
BIC(Model4)#27148.17


anova(Model1 , fmi)
anova(Model2 , fmi)
anova(Model1 , Model3)
anova(Model2 , Model3)
anova(Model1 , Model4)
anova(Model2 , Model4)


#Variable selection with interactions
###################################################################
#Backward variable selection with full model and interactions using t-tests

bfmvst <- fmi
summary(bfmvst)
vif(bfmvst)

#Removing I5 because of next high pvalue 0.748792
bfmvst1<- lm(charges ~ age+sex+bmi+children+smoker+Regionsouthwest+Regionnortheast+
                  Regionnorthwest+I1+I2+I3+I4+I6+I7+I8, int_df) 
summary(bfmvst1)

#Removing I1 because of next high pvalue 0.749794
bfmvst2<- lm(charges ~ age+sex+bmi+children+smoker+Regionsouthwest+Regionnortheast+
                  Regionnorthwest+I2+I3+I4+I6+I7+I8, int_df) 
summary(bfmvst2)


#Removing I3 because of next high pvalue 0.726527
bfmvst3<- lm(charges ~ age+sex+bmi+children+smoker+Regionsouthwest+Regionnortheast+
                  Regionnorthwest+I2+I4+I6+I7+I8, int_df) 
summary(bfmvst3)

#Removing Regionsouthwest because of next high pvalue 0.284555
bfmvst4<- lm(charges ~ age+sex+bmi+children+smoker+Regionnortheast+
                  Regionnorthwest+I2+I4+I6+I7+I8, int_df) 
summary(bfmvst4)

#Removing I6 because of next high pvalue 0.337683
bfmvst5<- lm(charges ~ age+sex+bmi+children+smoker+Regionnortheast+
                  Regionnorthwest+I2+I4+I7+I8, int_df) 
summary(bfmvst5)

#Removing I2 because of next high pvalue 0.105783
bfmvst6<- lm(charges ~ age+sex+bmi+children+smoker+Regionnortheast+
                  Regionnorthwest+I4+I7+I8, int_df) 
summary(bfmvst6)


#Removing sex because of next high pvalue 0.100473
bfmvst7<- lm(charges ~ age+bmi+children+smoker+Regionnortheast+
                  Regionnorthwest+I4+I7+I8, int_df) 
summary(bfmvst7)

#final model with interactions using Backward variable
Model5<-lm(charges ~ age+bmi+children+smoker+Regionnortheast+
            Regionnorthwest+I4+I7+I8, int_df)
summary(Model5)#Model is overall significant. Adj R2 is 0.7559
vif(Model5)#No issues with multi-collinearity
PRESS(Model5)#48544435990
AIC(Model5)#27061.18
BIC(Model5)#27118.36


###################################################################

#Backward variable selection with full model and interactions using t-tests

bpmwit <- Model3 #partial model
summary(bpmwit)
vif(bpmwit)

#Removing I3 because of next high pvalue 0.794378
bpmwit1<- lm(charges ~ age + bmi + children + smoker + Regionnortheast + 
               I1 + I2  + I4 + I5 + I6 + I7 + I8, int_df) 
summary(bpmwit1)

#Removing I1 because of next high pvalue 0.765193
bpmwit2<- lm(charges ~ age + bmi + children + smoker + Regionnortheast + 
                I2  + I4 + I5 + I6 + I7 + I8, int_df) 
summary(bpmwit2)


#Removing I6 because of next high pvalue 0.001357
bpmwit3<- lm(charges ~ age + bmi + children + smoker + Regionnortheast + 
               I2  + I4 + I5 + I7 + I8, int_df)  
summary(bpmwit3)

#Removing I5 because of next high pvalue 0.123223
bpmwit4<- lm(charges ~ age + bmi + children + smoker + Regionnortheast + 
               I2  + I4 + I7 + I8, int_df) 
summary(bpmwit4)

#Removing I2 because of next high pvalue 0.112217
bpmwit5<- lm(charges ~ age + bmi + children + smoker + Regionnortheast + 
                 I4 + I7 + I8, int_df) 
summary(bpmwit5)


#final model with interactions using Backward variable
Model6<-lm(charges ~ age + bmi + children + smoker + Regionnortheast + 
             I4 + I7 + I8, int_df) #Model 3 of A
summary(Model6)#Model is overall significant. Adj R2 is 0.7541
vif(Model6)#No issues with multi-collinearity
PRESS(Model6)#48899229995
AIC(Model6)#27070.21
BIC(Model6)#27122.19


###################################################################

#Stepwise variable selection with full model and interactions using F-tests
###################################################################

#Forward

#Empty model
em<-lm(int_df$charges~1,int_df)
fsfm<-lm(int_df$charges~.,int_df)

#Forward stepwise full model
add1(em,scope=fsfm,test="F")

#Adding/dropping will be done based on high F-value and low p-value

#Adding I2 F-value (2205.3591) and p-value(2.2e-16)
add1(update(em,~.+I2),scope=fsfm,test="F")
#Adding I1 F-value (237.1830) and p-value(2.2e-16)
drop1(update(em,~.+I1+I2),test="F") 

add1(update(em,~.+I1+I2),scope=fsfm,test="F")
#Adding smoker F-value (338.7882) and p-value(2.2e-16)
drop1(update(em,~.+smoker+I1+I2),test="F")

add1(update(em,~.+smoker+I1),scope=fsfm,test="F")# Removing I2 p-value(0.08316)
#Adding Children F-value (11.4319) and p-value(0.0007428)
drop1(update(em,~.+children+smoker+I1),test="F")

add1(update(em,~.+children+smoker+I1),scope=fsfm,test="F")
#Adding I8 F-value (6.8796) and p-value(0.008818)
drop1(update(em,~.+children+smoker+I1+I8),test="F")

add1(update(em,~.+children+smoker+I1+I8),scope=fsfm,test="F")
#Adding Regionnortheast F-value (11.5685) and p-value(0.0006907)
drop1(update(em,~.+children+smoker+Regionnortheast+I1+I8),test="F")

add1(update(em,~.+children+smoker+Regionnortheast+I1+I8),scope=fsfm,test="F")
#Adding I7 F-value (10.4623) and p-value(0.001248)
drop1(update(em,~.+children+smoker+Regionnortheast+I1+I7+I8),test="F")

add1(update(em,~.+children+smoker+Regionnortheast+I1+I7+I8),scope=fsfm,test="F")
#Adding Regionnorthwest F-value (11.0454) and p-value(0.0009133)
drop1(update(em,~.+children+smoker+Regionnortheast+Regionnorthwest+I1+I7+I8),test="F")

add1(update(em,~.+children+smoker+Regionnortheast+Regionnorthwest+I1+I7+I8),scope=fsfm,test="F")
#Adding I4 F-value (4.5092) and p-value(0.03390)
drop1(update(em,~.+children+smoker+Regionnortheast+Regionnorthwest+I1+I4+I7+I8),test="F")

add1(update(em,~.+children+smoker+Regionnortheast+Regionnorthwest+I1+I4+I7+I8),scope=fsfm,test="F")

#Final model after forward stepwise variable selection using F-test is
#int_df$charges ~ children + smoker + Regionnortheast + Regionnorthwest + I1 + I4 
#+ I7 + I8
Model7<- lm(charges ~ children+smoker+Regionnortheast+
                  Regionnorthwest+I1+I4+I7+I8, int_df) 
summary(Model7)#Model is overall significant. Adj R2 is 0.7532
vif(Model7)#No issues with multi-collinearity
PRESS(Model7)#49044182722
AIC(Model7)#27075.21
BIC(Model7)#27127.2

###################################################################

#Backward

drop1(fsfm,test="F")

#Removing I5 because of high p-value
drop1(update(fsfm,~.-I5),test="F")
#Removing I1 because of high p-value
add1(update(fsfm,~.-I1-I5),scope=fsfm,test="F")

drop1(update(fsfm,~.-I1-I5),test="F")
#Removing I3 because of high p-value
add1(update(fsfm,~.-I1-I3-I5),scope=fsfm,test="F")


drop1(update(fsfm,~.-I1-I3-I5),test="F")
#Removing Regionsouthwest because of high p-value
add1(update(fsfm,~.-Regionsouthwest-I1-I3-I5),scope=fsfm,test="F")

drop1(update(fsfm,~.-Regionsouthwest-I1-I3-I5),test="F")
#Removing I6 because of high p-value
add1(update(fsfm,~.-Regionsouthwest-I1-I3-I5-I6),scope=fsfm,test="F")

drop1(update(fsfm,~.-Regionsouthwest-I1-I3-I5-I6),test="F")
#Removing I2 because of high p-value
add1(update(fsfm,~.-Regionsouthwest-I1-I2-I3-I5-I6),scope=fsfm,test="F")

drop1(update(fsfm,~.-Regionsouthwest-I1-I2-I3-I5-I6),test="F")
#Removing sex because of high p-value
add1(update(fsfm,~.-Regionsouthwest-sex-I1-I2-I3-I5-I6),scope=fsfm,test="F")

drop1(update(fsfm,~.-Regionsouthwest-sex-I1-I2-I3-I5-I6),test="F")

#Final model after backward stepwise variable selection using F-test is
# charges ~ age + bmi + children + smoker + Regionnortheast + Regionnorthwest + 
#   I4 + I7 + I8
#This model is similar to Model5
bsvsfm<- lm(charges ~ age + bmi +children+smoker+Regionnortheast+
              Regionnorthwest+I4+I7+I8, int_df) 
summary(bsvsfm)
vif(bsvsfm)
################################################################

#stepAIC for full model with interactions and binary variables using AIC
################################################################

#forward AIC

stepAIC(em,direction='forward',scope=list(lower=em,upper=fsfm))

#Final model after using forward stepAIC for variable selection is
#AIC value is 23276.86
# int_df$charges ~ I2 + I1 + smoker + children + I8 + 
# Regionnortheast + I7 + Regionnorthwest + I4 + sex


#both AIC
stepAIC(em,direction='both',scope=list(lower=em,upper=fsfm))
#Final model after using forward stepAIC for variable selection is
#AIC value is 23276.86
# int_df$charges ~ I2 + I1 + smoker + children + I8 + 
# Regionnortheast + I7 + Regionnorthwest + I4 + sex
Model8<-lm(charges ~ I2 + I1 + smoker + children + I8 + 
             Regionnortheast + I7 + Regionnorthwest + I4 + sex, int_df)
summary(Model8)#Model is overall significant. Adj R2 is 0.7532
# there are multi-collinearity issues as variables I2 and smoker has vif values 
#greater than 5
vif(Model8)
PRESS(Model8)#49007646358
AIC(Model8)#27073.11
BIC(Model8)#27135.48

#backward AIC
stepAIC(fsfm,direction='backward',scope=list(lower=em,upper=fsfm))
#Final model after using backward stepAIC for variable selection is
#AIC value is 23263.57
# charges ~ age + sex + bmi + children + smoker + Regionnortheast + 
# Regionnorthwest + I2 + I4 + I7 + I8

Model9<-lm(charges ~ age + sex + bmi + children + smoker + Regionnortheast + 
              Regionnorthwest + I2 + I4 + I7 + I8, int_df) #Model 5(b) of A
summary(Model9)#Model is overall significant. Adj R2 is 0.7532
# there are multi-collinearity issues as variables I2 and smoker has vif values 
#greater than 5
vif(Model9)
PRESS(Model9)#48525124174
AIC(Model9)#27059.81
BIC(Model9)#27127.39

################################################################

# Regsubsets for full model with interactions and binary variables using BIC
################################################################

bic_fmbvi<-regsubsets(int_df[,c(1:5, 7:17)],int_df[,6],method="exhaustive",nbest=3, nvmax = 18)
bic_fmbvisum<-summary(bic_fmbvi)
bic_fmbvisum
bic_fmbvisum$rsq #0.7589653, 0.7589466, 0.7589454
bic_fmbvisum$adjr2  #0.7565367, 0.7565245, 0.7565098
bic_fmbvisum$bic  # -1825.793, -1825.3785, -1822.5418

coef(bic_fmbvi,22)
Model10<-lm(charges ~ age + bmi + children + smoker + Regionnortheast + 
             Regionnorthwest + I7 + I8, int_df) #our best model 9
summary(Model10)#Model is overall significant. Adj R2 is 0.7554
vif(Model10)#No issues with multi-collinearity
PRESS(Model10)#48516281448
AIC(Model10)#27063.12
BIC(Model10)#27115.1

coef(bic_fmbvi,23)
Model11<-lm(charges ~ age + bmi + smoker + Regionnortheast + 
             Regionnorthwest + I3 + I7 + I8, int_df)
summary(Model11)#Model is overall significant. Adj R2 is 0.7553
vif(Model11)#No issues with multi-collinearity
PRESS(Model11)#48533686269
AIC(Model11)#27063.54
BIC(Model11)#27115.52

coef(bic_fmbvi,25)
Model12<-lm(charges ~ age + bmi + children + smoker + Regionnortheast + 
             Regionnorthwest + I4 + I7 + I8, int_df)
summary(Model12)#Model is overall significant. Adj R2 is 0.7559
vif(Model12)#No issues with multi-collinearity
PRESS(Model12)#48544435990
AIC(Model12)#27061.18
BIC(Model12)#27118.36


#-------------------------------------------------------------------------------
### 6. Model Validation ####
#-------------------------------------------------------------------------------
#PRESS statistic 
PRESS(Model3)#49178048766
PRESS(Model4)#48815060515
PRESS(Model5)#48544435990
PRESS(Model6)#48899229995
PRESS(Model7)#49044182722
PRESS(Model8)#49007646358
PRESS(Model9)#48525124174
PRESS(Model10)#48516281448
PRESS(Model11)#48533686269
PRESS(Model12)#48544435990

#RMSE values
Model3_pred<-predict(Model3,int_df)
sqrt(mean(Model3_pred-int_df$charges)^2)#1.191712e-10

Model4_pred<-predict(Model4,int_df)
sqrt(mean(Model4_pred-int_df$charges)^2)#1.164026e-10

Model5_pred<-predict(Model5,int_df)
sqrt(mean(Model5_pred-int_df$charges)^2)#1.32326e-10

Model6_pred<-predict(Model6,int_df)
sqrt(mean(Model6_pred-int_df$charges)^2)#1.301807e-10

Model7_pred<-predict(Model7,int_df)
sqrt(mean(Model7_pred-int_df$charges)^2)#9.002592e-12

Model8_pred<-predict(Model8,int_df)
sqrt(mean(Model8_pred-int_df$charges)^2)#9.113699e-12

Model9_pred<-predict(Model9,int_df)
sqrt(mean(Model9_pred-int_df$charges)^2)#1.271366e-10

Model10_pred<-predict(Model10,int_df)
sqrt(mean(Model10_pred-int_df$charges)^2)#1.326281e-10

Model11_pred<-predict(Model11,int_df)
sqrt(mean(Model11_pred-int_df$charges)^2)#1.304698e-10

Model12_pred<-predict(Model12,int_df)
sqrt(mean(Model12_pred-int_df$charges)^2)# 1.32326e-10



#K-fold cross validation
set.seed(100)
c<-trainControl(method = "cv", number=10)
M10<-train(charges ~ age + bmi + children + smoker + Regionnortheast + 
            Regionnorthwest + I7 + I8, int_df, trControl=c, method="lm")
M10$results
M10$resample


#-------------------------------------------------------------------------------
### 7. Residual Analysis ####
#-------------------------------------------------------------------------------

Model10$residuals
sum(Model10$residuals)#-1.066354e-09

qqnorm(Model10$residuals)
qqline(Model10$residuals)

o<-studres(Model10)
max(o)#4.764008
min(o)#-2.199371

plot(Model10$residuals~Model10$fitted.values, col=int_df$charges,
     main = "Residual Vs Fitted before removing influential points")

plot(Model10_pred,int_df$charges)
abline(Model10)

cor(Model10_pred,int_df$charges)#0.8699667

boxcox(Model10)
boxcox(Model10,lambda=seq(0,.6,.1))
int_df$tr_c<-int_df$charges^.15

reg_af_tr<-lm(tr_c~age + bmi + children + smoker + Regionnortheast + 
                Regionnorthwest + I7 + I8,int_df)#regression model after transformation
summary(reg_af_tr)
qqnorm(reg_af_tr$residuals)
qqline(reg_af_tr$residuals)
plot(reg_af_tr$residuals~reg_af_tr$fitted.values, col=int_df$charges,
     main = "Residual Vs Fitted after transformation")


CD<-cooks.distance(Model10)
max(CD)#0.04209745
which.max(CD)#578


n <- nrow(int_df)
n
plot(CD, main = "Cooks Distance for Influential Obs")
abline(h = 4/n,lwd=4, lty = 2, col = "steelblue") # add cutoff line

influential_obs <- as.numeric(names(CD)[(CD > (4/n))])

int_df2<-int_df[-influential_obs,]#removing the influential values

Model10_2<-lm(charges ~ age + bmi + children + smoker + Regionnortheast + 
               Regionnorthwest + I7 + I8,int_df2)

qqnorm(Model10_2$residuals)
qqline(Model10_2$residuals)
o2<-studres(Model10_2)
max(o2)#5.388597
min(o2)#-2.903705

plot(Model10_2$residuals~Model10_2$fitted.values, col=int_df2$charges,
     main = "Residual Vs Fitted after removing influential points")

Model10_2_pred<-predict(Model10_2,int_df2)
plot(Model10_2_pred,int_df2$charges)
abline(Model10_2)
cor(Model10_2_pred,int_df2$charges)#0.8946644

boxcox(Model10_2)
cooks.distance(Model10_2)
max(cooks.distance(Model10_2))
summary(Model10_2)

#-------------------------------------------------------------------------------
### 8. Robust Regression ####
#-------------------------------------------------------------------------------

#Robust with outliers
robust<-rlm(charges ~ age + bmi + children + smoker + Regionnortheast +
                     Regionnorthwest + I7 + I8,int_df)
robust
summary(robust)
robust$w
robust$residuals

#Robust without outliers
robust2<-rlm(charges ~ age + bmi + children + smoker + Regionnortheast +
              Regionnorthwest + I7 + I8,int_df2)
robust2
summary(robust2)
robust2$w
robust2$residuals

#-------------------------------------------------------------------------------
### 9. Creating Train and Test datasets ####
#-------------------------------------------------------------------------------

set.seed(100) 

#filtering variables and creating a new dataset as per model

#nd_fm_wint - new data full model with interactions
nd_fm_wint<-int_df
nd_fm_wint<-nd_fm_wint[-c(2,7,10:15,18)]

#creation of data partition
row.number <- sample(1:nrow(nd_fm_wint), 0.8*nrow(nd_fm_wint))
train <- nd_fm_wint[row.number,]
test <- nd_fm_wint[-row.number,]
dim(train)
dim(test)

M_tr_te<-lm(charges ~ age + bmi + children + smoker + Regionnortheast + 
              Regionnorthwest + I7 + I8,train)
summary(M_tr_te)


pred<-predict(M_tr_te,test)
sqrt(mean(pred-test$charges)^2)#134.3156
mean(pred)#13562.4

