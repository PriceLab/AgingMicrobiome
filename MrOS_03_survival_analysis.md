## This notebook contains survival analysis reported in Figure 5


```R
library(ggplot2)
library(OneR)
library(survival)
library(survminer)
library(dplyr)
library(robustHD)
library(smoothHR)
```


```R
#import csv file generated during microbiome analysis in Notebook1 (it contains the uniqueness and bacteroides variables)
d<-read.csv('df_uniqueness_new.csv')
#check dimensions
dim(d)
#set index
rownames(d)<-d$X
```


```R
#Preparing data for survival analysis (both discovery and validation cohorts)
#Replace NA (lost to follow-up) with 0
d$dead[is.na(d$dead)] <- 0
#remove individuals in assisted living, nursing homes and who have been hospitalized in the past 12 months.
dim(d)
d<-d[which(d$giliveh!=1),]
d<-d[which(d$giliven!=1),]
d<-d[which(d$mhhsp!=1),]
dim(d)
#recode self-percieved health to where anyone reported less than good health is grouped together
d$health[d$qlhealth=='excellent']<-'1excellent'
d$health[d$qlhealth!='excellent' & d$qlhealth!='good']<-'3<good'
d$health[d$qlhealth=='good']<-'2good'
```


```R
#Focus on oldest individuals in the cohort
extreme_age<-d[which(d$age>=85),]
#check dimensions
dim(extreme_age)
dim(d)
#standardize bacteroides and genus-level Bray-Curtis uniqueness
d$standardized_bac<-standardize(d$bacteroides)
d$standardized_bray<-standardize(d$min_bray_genus)
extreme_age$standardized_bac<-standardize(extreme_age$bacteroides)
extreme_age$standardized_bray<-standardize(extreme_age$min_bray_genus)
#generate the estimate of survival at time t+0
surv_object<-Surv(time=d$survival,event=d$dead)
surv_object_extreme_age<-Surv(time=extreme_age$survival,event=extreme_age$dead)
#fit univariate models
'Bacteroides all community dwelling participants unadjusted models'
fit.coxph <-coxph(surv_object ~ standardized_bac,data=d)
summary(fit.coxph)
'Uniqueness all community dwelling participants unadjusted models'
fit.coxph <-coxph(surv_object ~ standardized_bray,data=d)
summary(fit.coxph)

```


```R
'Bacteroides 85+ year old community dwelling only unadjusted'
fit.coxph <-coxph(surv_object_extreme_age ~ standardized_bac,data=extreme_age)
summary(fit.coxph)
'Uniqueness 85+ year old community dwelling only unadjusted'
fit.coxph <-coxph(surv_object_extreme_age ~ standardized_bray,data=extreme_age)
summary(fit.coxph)
```


```R
#fit models adjusted for clinical site, batch (discovery/validation) and age
'ALL COMMUNITY DWELLING PARTICIPANTS'
'Bacteroides adjusted for clinical site, batch (discovery/validation), and age'
fit.coxph <-coxph(surv_object ~ age+site+firstcohort+standardized_bac,data=d)
summary(fit.coxph)
'Uniqueness adjusted for clinical site, batch (discovery/validation), and age'
fit.coxph <-coxph(surv_object ~ age+site+firstcohort+standardized_bray,data=d)
summary(fit.coxph)
```


```R
#fit models adjusted for clinical site, batch (discovery/validation) and age in 85+ year olds
'85+ YEAR OLD COMMUNITY DWELLING'
'85+ year olds Bacteroides adjusted for clinical site, batch (discovery/validation), and age'
fit.coxph <-coxph(surv_object_extreme_age ~ age+site+firstcohort+standardized_bac,data=extreme_age)
summary(fit.coxph)
'85+ year olds Uniqueness adjusted for clinical site, batch (discovery/validation), and age'
fit.coxph <-coxph(surv_object_extreme_age ~ age+site+firstcohort+standardized_bray,data=extreme_age)
summary(fit.coxph)
```


```R
'ALL COMMUNITY DWELLING ADJUSTED FOR ALL COVARIATES'
fit.coxph <-coxph(surv_object ~ age+site+firstcohort+hwbmi+health+mhchf+standardized_bray,data=d)
summary(fit.coxph)
fit.coxph <-coxph(surv_object ~ age+site+firstcohort+hwbmi+health+mhchf+standardized_bac,data=d)
summary(fit.coxph)

```


```R
'85+ COMMUNITY DWELLING ADJUSTED FOR ALL COVARIATES'
fit.coxph <-coxph(surv_object_extreme_age ~ age+site+firstcohort+hwbmi+health+mhchf+standardized_bray,data=extreme_age)
summary(fit.coxph)
fit.coxph <-coxph(surv_object_extreme_age ~ age+site+firstcohort+hwbmi+health+mhchf+standardized_bac,data=extreme_age)
summary(fit.coxph)
```


```R
#Bin participants into tertiles of bacteroides abundance
extreme_age$bac <- bin(extreme_age$standardized_bac, nbins =3, labels = NULL, method = "content", na.omit = FALSE)
```


```R
fit.coxph <-coxph(surv_object_extreme_age ~ age+site+firstcohort+hwbmi+health+mhchf+bac,data=extreme_age)
summary(fit.coxph)
```


```R
#Remove the individuals in the middle tertile to generate survival curve for t1 vs t3
b <-extreme_age[which(extreme_age$bac!='(-0.541,0.35]'),]
#generate new survival object
surv_b<-Surv(time=b$survival,event=b$dead)
#fit model
fit1<-survfit(surv_b~bac,data=b)
#generate survival curve
p<-ggsurvplot(fit1,data=b,font.x = c(20, "plain", "black"), font.y = c(20, "plain", "black"),font.tickslab = c(16, "plain", "black"), pval=TRUE,conf.int=FALSE,risk.table=FALSE,fontsize=4,palette=c('darkblue','darkred'),test.for.trend
=FALSE)
p
ggsave('bacteroides_survival.jpg',width=5,height=5,font=16)
```


```R

```
