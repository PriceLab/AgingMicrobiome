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

    Loading required package: ggpubr
    Loading required package: magrittr
    
    Attaching package: ‘dplyr’
    
    The following objects are masked from ‘package:stats’:
    
        filter, lag
    
    The following objects are masked from ‘package:base’:
    
        intersect, setdiff, setequal, union
    
    Loading required package: perry
    Loading required package: parallel
    Loading required package: robustbase
    
    Attaching package: ‘robustbase’
    
    The following object is masked from ‘package:survival’:
    
        heart
    
    Loading required package: splines



```R
#import csv file generated during microbiome analysis in Notebook1 (it contains the uniqueness and bacteroides variables)
d<-read.csv('df_uniqueness.csv')
#check dimensions
dim(d)
#set index
rownames(d)<-d$X
```


<ol class=list-inline>
	<li>907</li>
	<li>143</li>
</ol>




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


<ol class=list-inline>
	<li>907</li>
	<li>143</li>
</ol>




<ol class=list-inline>
	<li>706</li>
	<li>143</li>
</ol>




```R
#Focus on oldest individuals in the cohort
extreme_age<-d[which(d$age>=85),]
#check dimensions
dim(extreme_age)
dim(d)
#standardize bacteroides and uniqueness
d$standardized_bac<-standardize(d$bacteroides)
d$standardized_bray<-standardize(d$min_bray)
extreme_age$standardized_bac<-standardize(extreme_age$bacteroides)
extreme_age$standardized_bray<-standardize(extreme_age$min_bray)
#generate the estimate of survival at time t+0
surv_object<-Surv(time=d$survival,event=d$dead)
surv_object_extreme_age<-Surv(time=extreme_age$survival,event=extreme_age$dead)
#fit univariate models
'Bacteroides all community dwelling participants'
fit.coxph <-coxph(surv_object ~ standardized_bac,data=d)
summary(fit.coxph)
'Uniqueness all community dwelling participants'
fit.coxph <-coxph(surv_object ~ standardized_bray,data=d)
summary(fit.coxph)

```


<ol class=list-inline>
	<li>257</li>
	<li>144</li>
</ol>




<ol class=list-inline>
	<li>706</li>
	<li>144</li>
</ol>




'Bacteroides all community dwelling participants'



    Call:
    coxph(formula = surv_object ~ standardized_bac, data = d)
    
      n= 706, number of events= 66 
    
                       coef exp(coef) se(coef)     z Pr(>|z|)
    standardized_bac 0.1871    1.2058   0.1227 1.525    0.127
    
                     exp(coef) exp(-coef) lower .95 upper .95
    standardized_bac     1.206     0.8294     0.948     1.534
    
    Concordance= 0.571  (se = 0.039 )
    Likelihood ratio test= 2.25  on 1 df,   p=0.1
    Wald test            = 2.33  on 1 df,   p=0.1
    Score (logrank) test = 2.33  on 1 df,   p=0.1




'Uniqueness all community dwelling participants'



    Call:
    coxph(formula = surv_object ~ standardized_bray, data = d)
    
      n= 706, number of events= 66 
    
                        coef exp(coef) se(coef)     z Pr(>|z|)
    standardized_bray 0.1524    1.1646   0.1218 1.251    0.211
    
                      exp(coef) exp(-coef) lower .95 upper .95
    standardized_bray     1.165     0.8587    0.9173     1.479
    
    Concordance= 0.519  (se = 0.042 )
    Likelihood ratio test= 1.51  on 1 df,   p=0.2
    Wald test            = 1.57  on 1 df,   p=0.2
    Score (logrank) test = 1.56  on 1 df,   p=0.2




```R
'Bacteroides 85+ year old community dwelling only'
fit.coxph <-coxph(surv_object_extreme_age ~ standardized_bac,data=extreme_age)
summary(fit.coxph)
'Uniqueness 85+ year old community dwelling only'
fit.coxph <-coxph(surv_object_extreme_age ~ standardized_bray,data=extreme_age)
summary(fit.coxph)
```


'Bacteroides 85+ year old community dwelling only'



    Call:
    coxph(formula = surv_object_extreme_age ~ standardized_bac, data = extreme_age)
    
      n= 257, number of events= 41 
    
                       coef exp(coef) se(coef)     z Pr(>|z|)    
    standardized_bac 0.5288    1.6970   0.1573 3.362 0.000773 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
                     exp(coef) exp(-coef) lower .95 upper .95
    standardized_bac     1.697     0.5893     1.247      2.31
    
    Concordance= 0.665  (se = 0.045 )
    Likelihood ratio test= 10.73  on 1 df,   p=0.001
    Wald test            = 11.3  on 1 df,   p=8e-04
    Score (logrank) test = 11.63  on 1 df,   p=6e-04




'Uniqueness 85+ year old community dwelling only'



    Call:
    coxph(formula = surv_object_extreme_age ~ standardized_bray, 
        data = extreme_age)
    
      n= 257, number of events= 41 
    
                         coef exp(coef) se(coef)      z Pr(>|z|)
    standardized_bray -0.2741    0.7603   0.1720 -1.594    0.111
    
                      exp(coef) exp(-coef) lower .95 upper .95
    standardized_bray    0.7603      1.315    0.5427     1.065
    
    Concordance= 0.575  (se = 0.055 )
    Likelihood ratio test= 2.66  on 1 df,   p=0.1
    Wald test            = 2.54  on 1 df,   p=0.1
    Score (logrank) test = 2.53  on 1 df,   p=0.1




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


'ALL COMMUNITY DWELLING PARTICIPANTS'



'Bacteroides adjusted for clinical site, batch (discovery/validation), and age'



    Call:
    coxph(formula = surv_object ~ age + site + firstcohort + standardized_bac, 
        data = d)
    
      n= 706, number of events= 66 
    
                        coef exp(coef) se(coef)     z Pr(>|z|)    
    age              0.15337   1.16576  0.02693 5.695 1.23e-08 ***
    siteIM           0.64343   1.90299  0.39555 1.627   0.1038    
    siteJF           0.30541   1.35718  0.37648 0.811   0.4172    
    siteQZ           0.23878   1.26970  0.42462 0.562   0.5739    
    siteRN           0.05879   1.06055  0.53150 0.111   0.9119    
    siteYA           0.25834   1.29478  0.44108 0.586   0.5581    
    firstcohort      0.36169   1.43575  0.32240 1.122   0.2619    
    standardized_bac 0.26318   1.30106  0.12744 2.065   0.0389 *  
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
                     exp(coef) exp(-coef) lower .95 upper .95
    age                  1.166     0.8578    1.1058     1.229
    siteIM               1.903     0.5255    0.8765     4.132
    siteJF               1.357     0.7368    0.6489     2.839
    siteQZ               1.270     0.7876    0.5524     2.918
    siteRN               1.061     0.9429    0.3742     3.006
    siteYA               1.295     0.7723    0.5454     3.074
    firstcohort          1.436     0.6965    0.7632     2.701
    standardized_bac     1.301     0.7686    1.0135     1.670
    
    Concordance= 0.691  (se = 0.041 )
    Likelihood ratio test= 38.84  on 8 df,   p=5e-06
    Wald test            = 41.19  on 8 df,   p=2e-06
    Score (logrank) test = 44.58  on 8 df,   p=4e-07




'Uniqueness adjusted for clinical site, batch (discovery/validation), and age'



    Call:
    coxph(formula = surv_object ~ age + site + firstcohort + standardized_bray, 
        data = d)
    
      n= 706, number of events= 66 
    
                         coef exp(coef) se(coef)     z Pr(>|z|)    
    age               0.13855   1.14860  0.02697 5.137 2.79e-07 ***
    siteIM            0.57324   1.77401  0.39725 1.443    0.149    
    siteJF            0.19358   1.21358  0.37319 0.519    0.604    
    siteQZ            0.22653   1.25425  0.42582 0.532    0.595    
    siteRN            0.02252   1.02278  0.53133 0.042    0.966    
    siteYA            0.10824   1.11431  0.43659 0.248    0.804    
    firstcohort       0.51257   1.66958  0.32334 1.585    0.113    
    standardized_bray 0.09097   1.09524  0.12502 0.728    0.467    
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
                      exp(coef) exp(-coef) lower .95 upper .95
    age                   1.149     0.8706    1.0895     1.211
    siteIM                1.774     0.5637    0.8144     3.865
    siteJF                1.214     0.8240    0.5840     2.522
    siteQZ                1.254     0.7973    0.5444     2.890
    siteRN                1.023     0.9777    0.3610     2.898
    siteYA                1.114     0.8974    0.4736     2.622
    firstcohort           1.670     0.5990    0.8859     3.147
    standardized_bray     1.095     0.9130    0.8572     1.399
    
    Concordance= 0.696  (se = 0.035 )
    Likelihood ratio test= 35.2  on 8 df,   p=2e-05
    Wald test            = 38.98  on 8 df,   p=5e-06
    Score (logrank) test = 41.6  on 8 df,   p=2e-06




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


'85+ YEAR OLD COMMUNITY DWELLING'



'85+ year olds Bacteroides adjusted for clinical site, batch (discovery/validation), and age'



    Call:
    coxph(formula = surv_object_extreme_age ~ age + site + firstcohort + 
        standardized_bac, data = extreme_age)
    
      n= 257, number of events= 41 
    
                         coef exp(coef) se(coef)      z Pr(>|z|)    
    age               0.11996   1.12745  0.05363  2.237 0.025303 *  
    siteIM            1.27148   3.56611  0.55764  2.280 0.022603 *  
    siteJF            1.21697   3.37695  0.51412  2.367 0.017929 *  
    siteQZ            0.88414   2.42090  0.57331  1.542 0.123030    
    siteRN            0.41741   1.51802  0.83112  0.502 0.615511    
    siteYA            1.07445   2.92838  0.60202  1.785 0.074303 .  
    firstcohort      -0.07691   0.92598  0.41585 -0.185 0.853277    
    standardized_bac  0.63150   1.88042  0.16831  3.752 0.000175 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
                     exp(coef) exp(-coef) lower .95 upper .95
    age                  1.127     0.8870    1.0150     1.252
    siteIM               3.566     0.2804    1.1954    10.638
    siteJF               3.377     0.2961    1.2328     9.250
    siteQZ               2.421     0.4131    0.7870     7.447
    siteRN               1.518     0.6588    0.2977     7.740
    siteYA               2.928     0.3415    0.8999     9.529
    firstcohort          0.926     1.0799    0.4099     2.092
    standardized_bac     1.880     0.5318    1.3520     2.615
    
    Concordance= 0.74  (se = 0.04 )
    Likelihood ratio test= 24.55  on 8 df,   p=0.002
    Wald test            = 22.72  on 8 df,   p=0.004
    Score (logrank) test = 24.35  on 8 df,   p=0.002




'85+ year olds Uniqueness adjusted for clinical site, batch (discovery/validation), and age'



    Call:
    coxph(formula = surv_object_extreme_age ~ age + site + firstcohort + 
        standardized_bray, data = extreme_age)
    
      n= 257, number of events= 41 
    
                          coef exp(coef) se(coef)      z Pr(>|z|)  
    age                0.12023   1.12775  0.05426  2.216   0.0267 *
    siteIM             1.20452   3.33516  0.56055  2.149   0.0316 *
    siteJF             1.00872   2.74208  0.50797  1.986   0.0471 *
    siteQZ             0.77428   2.16903  0.57400  1.349   0.1774  
    siteRN             0.39727   1.48776  0.83320  0.477   0.6335  
    siteYA             0.85709   2.35630  0.59547  1.439   0.1500  
    firstcohort       -0.02848   0.97192  0.40543 -0.070   0.9440  
    standardized_bray -0.36632   0.69328  0.17953 -2.040   0.0413 *
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
                      exp(coef) exp(-coef) lower .95 upper .95
    age                  1.1278     0.8867    1.0140    1.2543
    siteIM               3.3352     0.2998    1.1117   10.0059
    siteJF               2.7421     0.3647    1.0132    7.4210
    siteQZ               2.1690     0.4610    0.7042    6.6812
    siteRN               1.4878     0.6722    0.2906    7.6165
    siteYA               2.3563     0.4244    0.7334    7.5699
    firstcohort          0.9719     1.0289    0.4391    2.1514
    standardized_bray    0.6933     1.4424    0.4876    0.9857
    
    Concordance= 0.696  (se = 0.044 )
    Likelihood ratio test= 15.02  on 8 df,   p=0.06
    Wald test            = 13.7  on 8 df,   p=0.09
    Score (logrank) test = 14.3  on 8 df,   p=0.07




```R
'ALL COMMUNITY DWELLING ADJUSTED FOR ALL COVARIATES'
fit.coxph <-coxph(surv_object ~ age+site+firstcohort+hwbmi+health+mhchf+standardized_bray,data=d)
summary(fit.coxph)
fit.coxph <-coxph(surv_object ~ age+site+firstcohort+hwbmi+health+mhchf+standardized_bac,data=d)
summary(fit.coxph)

```


'ALL COMMUNITY DWELLING ADJUSTED FOR ALL COVARIATES'



    Call:
    coxph(formula = surv_object ~ age + site + firstcohort + hwbmi + 
        health + mhchf + standardized_bray, data = d)
    
      n= 706, number of events= 66 
    
                          coef exp(coef) se(coef)      z Pr(>|z|)    
    age                0.13578   1.14543  0.02742  4.953 7.32e-07 ***
    siteIM             0.43377   1.54307  0.40252  1.078  0.28119    
    siteJF             0.11417   1.12094  0.37516  0.304  0.76088    
    siteQZ             0.17470   1.19089  0.42735  0.409  0.68269    
    siteRN            -0.03718   0.96350  0.53226 -0.070  0.94430    
    siteYA             0.07545   1.07836  0.44145  0.171  0.86430    
    firstcohort        0.54856   1.73075  0.32328  1.697  0.08973 .  
    hwbmi             -0.03429   0.96629  0.03572 -0.960  0.33703    
    health2good        0.91226   2.48993  0.31631  2.884  0.00393 ** 
    health3<good       1.15706   3.18057  0.43713  2.647  0.00812 ** 
    mhchf              0.72056   2.05559  0.38754  1.859  0.06298 .  
    standardized_bray  0.10521   1.11095  0.12410  0.848  0.39657    
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
                      exp(coef) exp(-coef) lower .95 upper .95
    age                  1.1454     0.8730    1.0855     1.209
    siteIM               1.5431     0.6481    0.7011     3.396
    siteJF               1.1209     0.8921    0.5373     2.338
    siteQZ               1.1909     0.8397    0.5154     2.752
    siteRN               0.9635     1.0379    0.3395     2.735
    siteYA               1.0784     0.9273    0.4539     2.562
    firstcohort          1.7308     0.5778    0.9185     3.261
    hwbmi                0.9663     1.0349    0.9009     1.036
    health2good          2.4899     0.4016    1.3395     4.628
    health3<good         3.1806     0.3144    1.3503     7.492
    mhchf                2.0556     0.4865    0.9618     4.393
    standardized_bray    1.1109     0.9001    0.8711     1.417
    
    Concordance= 0.724  (se = 0.034 )
    Likelihood ratio test= 50.83  on 12 df,   p=1e-06
    Wald test            = 51.63  on 12 df,   p=7e-07
    Score (logrank) test = 56.91  on 12 df,   p=8e-08




    Call:
    coxph(formula = surv_object ~ age + site + firstcohort + hwbmi + 
        health + mhchf + standardized_bac, data = d)
    
      n= 706, number of events= 66 
    
                         coef exp(coef) se(coef)      z Pr(>|z|)    
    age               0.14668   1.15798  0.02737  5.360 8.33e-08 ***
    siteIM            0.54100   1.71772  0.40549  1.334  0.18214    
    siteJF            0.22896   1.25730  0.38017  0.602  0.54700    
    siteQZ            0.18696   1.20558  0.42820  0.437  0.66239    
    siteRN            0.01854   1.01872  0.53399  0.035  0.97230    
    siteYA            0.25407   1.28926  0.44820  0.567  0.57081    
    firstcohort       0.40626   1.50119  0.32128  1.264  0.20605    
    hwbmi            -0.04249   0.95840  0.03612 -1.176  0.23946    
    health2good       0.83457   2.30383  0.31447  2.654  0.00796 ** 
    health3<good      1.13217   3.10240  0.43564  2.599  0.00935 ** 
    mhchf             0.68960   1.99292  0.39019  1.767  0.07717 .  
    standardized_bac  0.24661   1.27968  0.12765  1.932  0.05337 .  
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
                     exp(coef) exp(-coef) lower .95 upper .95
    age                 1.1580     0.8636    1.0975     1.222
    siteIM              1.7177     0.5822    0.7759     3.803
    siteJF              1.2573     0.7954    0.5968     2.649
    siteQZ              1.2056     0.8295    0.5208     2.790
    siteRN              1.0187     0.9816    0.3577     2.901
    siteYA              1.2893     0.7756    0.5356     3.103
    firstcohort         1.5012     0.6661    0.7998     2.818
    hwbmi               0.9584     1.0434    0.8929     1.029
    health2good         2.3038     0.4341    1.2439     4.267
    health3<good        3.1024     0.3223    1.3209     7.286
    mhchf               1.9929     0.5018    0.9276     4.282
    standardized_bac    1.2797     0.7814    0.9964     1.643
    
    Concordance= 0.728  (se = 0.036 )
    Likelihood ratio test= 53.77  on 12 df,   p=3e-07
    Wald test            = 55.37  on 12 df,   p=2e-07
    Score (logrank) test = 60.31  on 12 df,   p=2e-08




```R
'85+ COMMUNITY DWELLING ADJUSTED FOR ALL COVARIATES'
fit.coxph <-coxph(surv_object_extreme_age ~ age+site+firstcohort+hwbmi+health+mhchf+standardized_bray,data=extreme_age)
summary(fit.coxph)
fit.coxph <-coxph(surv_object_extreme_age ~ age+site+firstcohort+hwbmi+health+mhchf+standardized_bac,data=extreme_age)
summary(fit.coxph)
```


'85+ COMMUNITY DWELLING ADJUSTED FOR ALL COVARIATES'



    Call:
    coxph(formula = surv_object_extreme_age ~ age + site + firstcohort + 
        hwbmi + health + mhchf + standardized_bray, data = extreme_age)
    
      n= 257, number of events= 41 
    
                          coef exp(coef) se(coef)      z Pr(>|z|)  
    age                0.10636   1.11222  0.05515  1.928   0.0538 .
    siteIM             1.17930   3.25210  0.57569  2.049   0.0405 *
    siteJF             0.89464   2.44647  0.51354  1.742   0.0815 .
    siteQZ             0.80661   2.24031  0.57897  1.393   0.1636  
    siteRN             0.37261   1.45151  0.83928  0.444   0.6571  
    siteYA             0.92597   2.52431  0.60146  1.540   0.1237  
    firstcohort        0.02797   1.02836  0.40945  0.068   0.9455  
    hwbmi             -0.04600   0.95504  0.04859 -0.947   0.3438  
    health2good        0.35697   1.42900  0.37984  0.940   0.3473  
    health3<good       0.85868   2.36005  0.60713  1.414   0.1573  
    mhchf              1.06715   2.90707  0.45664  2.337   0.0194 *
    standardized_bray -0.35318   0.70245  0.17560 -2.011   0.0443 *
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
                      exp(coef) exp(-coef) lower .95 upper .95
    age                  1.1122     0.8991    0.9983     1.239
    siteIM               3.2521     0.3075    1.0523    10.050
    siteJF               2.4465     0.4088    0.8941     6.694
    siteQZ               2.2403     0.4464    0.7203     6.968
    siteRN               1.4515     0.6889    0.2802     7.520
    siteYA               2.5243     0.3961    0.7766     8.205
    firstcohort          1.0284     0.9724    0.4609     2.294
    hwbmi                0.9550     1.0471    0.8683     1.050
    health2good          1.4290     0.6998    0.6788     3.009
    health3<good         2.3600     0.4237    0.7180     7.757
    mhchf                2.9071     0.3440    1.1878     7.115
    standardized_bray    0.7025     1.4236    0.4979     0.991
    
    Concordance= 0.73  (se = 0.04 )
    Likelihood ratio test= 22.13  on 12 df,   p=0.04
    Wald test            = 21.14  on 12 df,   p=0.05
    Score (logrank) test = 23.74  on 12 df,   p=0.02




    Call:
    coxph(formula = surv_object_extreme_age ~ age + site + firstcohort + 
        hwbmi + health + mhchf + standardized_bac, data = extreme_age)
    
      n= 257, number of events= 41 
    
                          coef exp(coef)  se(coef)      z Pr(>|z|)    
    age               0.096047  1.100811  0.054615  1.759 0.078644 .  
    siteIM            1.321549  3.749226  0.570728  2.316 0.020583 *  
    siteJF            1.146666  3.147681  0.523039  2.192 0.028357 *  
    siteQZ            1.002786  2.725865  0.585292  1.713 0.086656 .  
    siteRN            0.495318  1.641019  0.835009  0.593 0.553055    
    siteYA            1.222937  3.397150  0.615244  1.988 0.046842 *  
    firstcohort      -0.004603  0.995408  0.415354 -0.011 0.991158    
    hwbmi            -0.063995  0.938010  0.050211 -1.275 0.202479    
    health2good       0.326084  1.385531  0.373677  0.873 0.382862    
    health3<good      0.869293  2.385223  0.607803  1.430 0.152654    
    mhchf             1.027144  2.793079  0.461744  2.224 0.026115 *  
    standardized_bac  0.642314  1.900875  0.168273  3.817 0.000135 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
                     exp(coef) exp(-coef) lower .95 upper .95
    age                 1.1008     0.9084    0.9891     1.225
    siteIM              3.7492     0.2667    1.2250    11.475
    siteJF              3.1477     0.3177    1.1292     8.774
    siteQZ              2.7259     0.3669    0.8656     8.584
    siteRN              1.6410     0.6094    0.3194     8.431
    siteYA              3.3972     0.2944    1.0172    11.345
    firstcohort         0.9954     1.0046    0.4410     2.247
    hwbmi               0.9380     1.0661    0.8501     1.035
    health2good         1.3855     0.7217    0.6661     2.882
    health3<good        2.3852     0.4192    0.7247     7.850
    mhchf               2.7931     0.3580    1.1299     6.904
    standardized_bac    1.9009     0.5261    1.3668     2.644
    
    Concordance= 0.768  (se = 0.038 )
    Likelihood ratio test= 32.05  on 12 df,   p=0.001
    Wald test            = 30.51  on 12 df,   p=0.002
    Score (logrank) test = 34.01  on 12 df,   p=7e-04




```R
#Bin participants into tertiles of bacteroides abundance
extreme_age$bac <- bin(extreme_age$standardized_bac, nbins =3, labels = NULL, method = "content", na.omit = FALSE)
```


```R
#Remove the individuals in the middle tertile to generate survival curve for t1 vs t3
b <-extreme_age[which(extreme_age$bac!='(-0.538,0.347]'),]
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


![png](output_11_0.png)



```R

```
