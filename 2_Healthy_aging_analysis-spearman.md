## This notebook takes the variables calculated in notebook #1 (MROS_Gut microbiome preprocessing) and performs analysis reported in Figure 4E and S1B


```python
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
```


```python
#Import data generated in notebook 1 that contains microbiome measures
df_unique = pd.read_csv('df_uniqueness_new.csv')
#check no. of samples
print(df_unique.shape)
#set index
df_unique.set_index(df_unique['Unnamed: 0'],inplace=True)
```


```python
#Identify the median number of medications in 85+ year old participants
#This corresponds to the reported stratification based on medication use
discovery_df=df_unique[df_unique['firstcohort']==1]
med_median=discovery_df['m1medsin'][discovery_df['age']>=85].median()
print('median no. of meds in 85+ year olds from the discovery cohort=',med_median)
print(np.percentile(df_unique['nfwlkspd'].dropna(),66))
walk_speed_cutoff=np.percentile(df_unique['nfwlkspd'].dropna(),66)
print(np.percentile(df_unique['lsc'].dropna(),66))
lsc_cutoff=np.percentile(df_unique['lsc'].dropna(),66)
```


```python
#health stratifications: here we generate new variables that stratify each participant based on their performance
#on health measures specified in the text
score=[]
for x in df_unique['qlhealth']:
    if x=='excellent':
        score.append(1)
    else:
        score.append(0)
print(np.sum(score))
med=[]
for x in df_unique['m1medsin']:
    if x<=med_median:
        med.append(1)
    else:
        med.append(0)
print(np.sum(med))
wlk=[]
#walking speed has 7 missing values that are either due to a participant not being able to come to the visit or not being
#able to perform the test. These participants were all classified into the less healthy (low) group.
for x in df_unique['nfwlkspd']:
    if x>=walk_speed_cutoff:
        wlk.append(1)
    else:
        wlk.append(0)
lsc=[]
for x in df_unique['lsc']:
    if x>=lsc_cutoff:
        lsc.append(1)
    else:
        lsc.append(0)
df_unique['percieved_health']=score
df_unique['med']=med
df_unique['wlk']=wlk
df_unique['lsc_quant']=lsc
#calculate total no. of times each participant was in the healthy group for composite healthy score
df_unique['total_health']=df_unique['percieved_health']+df_unique['wlk']+df_unique['lsc_quant']+df_unique['med']
comp=[]
for x in df_unique['total_health']:
    if x>=3:
        comp.append(1)
    else:
        comp.append(0)
#Generate stratfication variable (composite healthy vs. not composite healthy)
df_unique['comp_healthy']=comp
```


```python
df_unique['comp_healthy'].sum()
```


```python
#save dataframe with health stratifications for demographics table
df_unique.to_csv('demographics.csv')
df_unique.shape
```


```python
#square root transform uniqueness measures for regression
df_unique['sqrt_min_bray']=np.sqrt(df_unique['min_bray'])
df_unique['sqrt_min_bray_g']=np.sqrt(df_unique['min_bray_genus'])
df_unique['sqrt_min_wunifrac_g']=np.sqrt(df_unique['min_wunifrac_genus'])
df_unique['sqrt_min_wunifrac']=np.sqrt(df_unique['min_wunifrac'])
df_unique['sqrt_min_bray_g'].hist(bins=20)
```


```python
#Correlation between Bray-Curtis Uniqueness and age reported in the text.
discovery=df_unique[df_unique['firstcohort']==1]
print(scipy.stats.spearmanr(discovery['sqrt_min_bray_g'],discovery['age']))
scipy.stats.spearmanr(discovery['sqrt_min_bray'],discovery['age'])
```


```python
#Perform pearson correlation between age and uniqueness for MrOS participants across health stratifications and cohorts
cohorts=[0,1]
stratifications=['med','wlk','lsc_quant','percieved_health','comp_healthy']
health=[]
Metric=[]
Cohort=[]
Coefficient=[]
pvalue=[]
sample_size=[]
condition=[]
reg_coef=[]
p_reg=[]
lower=[]
upper=[]
for x in cohorts:
    for y in stratifications:    
        df=df_unique[df_unique['firstcohort']==x]
        df_healthy=df[df[y]==1]
        condition.append(1)
        sample_size.append(len(df_healthy))
        health.append(y)
        df_sick=df[df[y]!=1]
        sample_size.append(len(df_sick))
        health.append(y)
        condition.append(0)
        Metric.append('Bray-Curtis')
        Metric.append('Bray-Curtis')
        Cohort.append(x)
        Cohort.append(x)
        coef=scipy.stats.spearmanr(df_healthy['age'],df_healthy['sqrt_min_bray_g'])[0]
        p=scipy.stats.spearmanr(df_healthy['age'],df_healthy['sqrt_min_bray_g'])[1]
        regression=smf.ols('sqrt_min_bray_g~hwbmi+age',data=df_healthy).fit()
        p_reg.append(regression.pvalues[2])
        lower.append(regression.conf_int(alpha=0.05, cols=None)[2:3][0].tolist())
        upper.append(regression.conf_int(alpha=0.05, cols=None)[2:3][1].tolist())
        reg_coef.append(regression.params[2].tolist())
        Coefficient.append(coef)
        pvalue.append(p)
        coef=scipy.stats.spearmanr(df_sick['age'],df_sick['sqrt_min_bray_g'])[0]
        Coefficient.append(coef)
        p=scipy.stats.spearmanr(df_sick['age'],df_sick['sqrt_min_bray_g'])[1]
        pvalue.append(p)
        regression=smf.ols('sqrt_min_bray_g~hwbmi+age',data=df_sick).fit()
        p_reg.append(regression.pvalues[2])
        lower.append(regression.conf_int(alpha=0.05, cols=None)[2:3][0].tolist())
        upper.append(regression.conf_int(alpha=0.05, cols=None)[2:3][1].tolist())
        reg_coef.append(regression.params[2].tolist())
        #Weighted_unifrac
        df=df_unique[df_unique['firstcohort']==x]
        df_healthy=df[df[y]==1]
        condition.append(1)
        sample_size.append(len(df_healthy))
        health.append(y)
        df_sick=df[df[y]!=1]
        sample_size.append(len(df_sick))
        health.append(y)
        condition.append(0)
        Metric.append('Weighted_Unifrac')
        Metric.append('Weighted_Unifrac')
        Cohort.append(x)
        Cohort.append(x)
        coef=scipy.stats.spearmanr(df_healthy['age'],df_healthy['sqrt_min_wunifrac_g'])[0]
        p=scipy.stats.spearmanr(df_healthy['age'],df_healthy['sqrt_min_wunifrac_g'])[1]
        regression=smf.ols('sqrt_min_wunifrac_g~hwbmi+age',data=df_healthy).fit()
        p_reg.append(regression.pvalues[2])
        lower.append(regression.conf_int(alpha=0.05, cols=None)[2:3][0].tolist())
        upper.append(regression.conf_int(alpha=0.05, cols=None)[2:3][1].tolist())
        reg_coef.append(regression.params[2].tolist())
        Coefficient.append(coef)
        pvalue.append(p)
        coef=scipy.stats.spearmanr(df_sick['age'],df_sick['sqrt_min_wunifrac_g'])[0]
        Coefficient.append(coef)
        p=scipy.stats.spearmanr(df_sick['age'],df_sick['sqrt_min_wunifrac_g'])[1]
        pvalue.append(p)
        regression=smf.ols('sqrt_min_wunifrac_g~hwbmi+age',data=df_sick).fit()
        p_reg.append(regression.pvalues[2])
        lower.append(regression.conf_int(alpha=0.05, cols=None)[2:3][0].tolist())
        upper.append(regression.conf_int(alpha=0.05, cols=None)[2:3][1].tolist())
        reg_coef.append(regression.params[2].tolist())
results=pd.DataFrame()
lower=[ x[0] for x in  lower]
upper=[ x[0] for x in  upper]
results['Metric']=Metric
results['Health']=health
results['cohort']=Cohort
results['spearmanr']=Coefficient
results['pvalue']=pvalue
results['beta_coef']=reg_coef
results['coef_pvalue']=p_reg
results['sample_size']=sample_size
results['lower']=lower
results['upper']=upper
results['healthy(yes1/no0)']=condition
results=results.sort_values(by=['cohort','Health','healthy(yes1/no0)'],ascending=True)
```


```python
#CI for plotting coefficients
results['err']=results['beta_coef']-results['lower']
```


```python
#results and sample sizes reported in Figure 4E
results.index=results['Metric']
r_bray=results[results.index=='Bray-Curtis']
r_bray
```


```python
#Figure 4E based on the above analysis
sns.set(font_scale=0.5,context='poster',font='Arial',style='white')
plt.figure(figsize=[18,18], dpi=200)
# Reorder it following the values:
#ordered_df = df.sort_values(by='values')
my_range=range(len(r_bray.index))
# Create a color if the group is "B"
my_color=np.where(r_bray ['healthy(yes1/no0)']==1, 'darkblue', 'darkred')
my_size=np.where(r_bray ['healthy(yes1/no0)']<2, 100, 30)
plt.figure(figsize=[5,10], dpi=200)
plt.rcParams.update({'font.size': 24})
plt.rcParams['axes.facecolor'] = 'white'
#plt.rcParams['axes.facecolor'] = 'white'
plt.hlines(y=my_range, xmin=0, xmax=r_bray['spearmanr'], color=my_color, alpha=0.5)
plt.scatter(r_bray['spearmanr'], my_range, color=my_color, s=my_size, alpha=1)
# Add title and exis names
plt.xlim(-0.15,0.35)
plt.yticks(my_range, r_bray['healthy(yes1/no0)'])
plt.xlabel('Spearmanr')
plt.ylabel('Group')
```


```python
#Figure 4E beta-coefficients
reversed_df=r_bray.iloc[::-1]
my_color=np.where(r_bray ['healthy(yes1/no0)']==1, 'darkred', 'darkblue')
sns.set(font_scale=0.75,context='poster',font='Arial',style='white')
fig, ax = plt.subplots(figsize=(10, 3.0))
reversed_df.plot(x='healthy(yes1/no0)', y='beta_coef', kind='bar', 
             ax=ax, color='none', edgecolor = "none",
             yerr='err', legend=False)
ax.set_ylabel('')
ax.set_xlabel('')
ax.scatter(x=pd.np.arange(reversed_df.shape[0]), 
           marker='s', s=120, 
           y=reversed_df['beta_coef'], color=my_color)
ax.axhline(y=0, linestyle='--', color='black', linewidth=4)
ax.xaxis.set_ticks_position('none')
```


```python
#results and sample sizes reported in Figure S1a
results.index=results['Metric']
Weighted_Unifrac=results[results.index=='Weighted_Unifrac']
Weighted_Unifrac
```


```python
#Figure 4E based on the above analysis
sns.set(font_scale=1.0,context='poster',font='Arial',style='white')
plt.figure(figsize=[18,18], dpi=200)
# Reorder it following the values:
#ordered_df = df.sort_values(by='values')
my_range=range(len(Weighted_Unifrac.index))
# Create a color if the group is "B"
my_color=np.where(Weighted_Unifrac ['healthy(yes1/no0)']==1, 'darkblue', 'darkred')
my_size=np.where(Weighted_Unifrac ['healthy(yes1/no0)']<2, 100, 30)
plt.figure(figsize=[5,10], dpi=200)
plt.rcParams.update({'font.size': 24})
plt.rcParams['axes.facecolor'] = 'white'
#plt.rcParams['axes.facecolor'] = 'white'
plt.hlines(y=my_range, xmin=0, xmax=Weighted_Unifrac['spearmanr'], color=my_color, alpha=0.5)
plt.scatter(Weighted_Unifrac['spearmanr'], my_range, color=my_color, s=my_size, alpha=1)
# Add title and exis names
plt.xlim(-0.2,0.40)
plt.yticks(my_range, Weighted_Unifrac['healthy(yes1/no0)'])
plt.xlabel('spearmanr')
plt.ylabel('Group')
```


```python
results.to_csv('spearman_healthy_aging_corr.csv')
```


```python
#Same code for Alpha diversity
#Perform pearson correlation between age and alpha diversity for MrOS participants across health stratifications and cohorts
cohorts=[0,1]
stratifications=['med','wlk','lsc_quant','percieved_health','comp_healthy']
health=[]
Metric=[]
Cohort=[]
Coefficient=[]
pvalue=[]
sample_size=[]
condition=[]
for x in cohorts:
    for y in stratifications:    
        df=df_unique[df_unique['firstcohort']==x]
        df_healthy=df[df[y]==1]
        condition.append(1)
        sample_size.append(len(df_healthy))
        health.append(y)
        df_sick=df[df[y]!=1]
        sample_size.append(len(df_sick))
        health.append(y)
        condition.append(0)
        Metric.append('Shannon')
        Metric.append('Shannon')
        Cohort.append(x)
        Cohort.append(x)
        coef=scipy.stats.spearmanr(df_healthy['age'],df_healthy['Shannon'])[0]
        p=scipy.stats.spearmanr(df_healthy['age'],df_healthy['Shannon'])[1]
        Coefficient.append(coef)
        pvalue.append(p)
        coef=scipy.stats.spearmanr(df_sick['age'],df_sick['Shannon'])[0]
        Coefficient.append(coef)
        p=scipy.stats.spearmanr(df_sick['age'],df_sick['Shannon'])[1]
        pvalue.append(p)
        #Weighted_unifrac
        df=df_unique[df_unique['firstcohort']==x]
        df_healthy=df[df[y]==1]
        condition.append(1)
        sample_size.append(len(df_healthy))
        health.append(y)
        df_sick=df[df[y]!=1]
        sample_size.append(len(df_sick))
        health.append(y)
        condition.append(0)
        Metric.append('Observed')
        Metric.append('Observed')
        Cohort.append(x)
        Cohort.append(x)
        coef=scipy.stats.spearmanr(df_healthy['age'],df_healthy['Observed'])[0]
        p=scipy.stats.spearmanr(df_healthy['age'],df_healthy['Observed'])[1]
        Coefficient.append(coef)
        pvalue.append(p)
        coef=scipy.stats.spearmanr(df_sick['age'],df_sick['Observed'])[0]
        Coefficient.append(coef)
        p=scipy.stats.spearmanr(df_sick['age'],df_sick['Observed'])[1]
        pvalue.append(p)
results_alpha=pd.DataFrame()
results_alpha['Metric']=Metric
results_alpha['Health']=health
results_alpha['cohort']=Cohort
results_alpha['spearmanr']=Coefficient
results_alpha['pvalue']=pvalue
results_alpha['sample_size']=sample_size
results_alpha['healthy(yes1/no0)']=condition
results_alpha=results_alpha.sort_values(by=['cohort','Health','healthy(yes1/no0)'],ascending=True)
```


```python
#results and sample sizes reported in Figure 4E
results_alpha.index=results_alpha['Metric']
Shannon=results_alpha[results_alpha.index=='Shannon']
Shannon
```


```python
#Figure 4E based on the above analysis
sns.set(font_scale=1.0,context='poster',font='Arial',style='white')
plt.figure(figsize=[18,18], dpi=200)
# Reorder it following the values:
#ordered_df = df.sort_values(by='values')
my_range=range(len(Shannon.index))
# Create a color if the group is "B"
my_color=np.where(Shannon ['healthy(yes1/no0)']==1, 'darkgreen', 'grey')
my_size=np.where(Shannon ['healthy(yes1/no0)']<2, 100, 30)
plt.figure(figsize=[5,10], dpi=200)
plt.rcParams.update({'font.size': 24})
plt.rcParams['axes.facecolor'] = 'white'
#plt.rcParams['axes.facecolor'] = 'white'
plt.hlines(y=my_range, xmin=0, xmax=Shannon['spearmanr'], color=my_color, alpha=0.5)
plt.scatter(Shannon['spearmanr'], my_range, color=my_color, s=my_size, alpha=1)
# Add title and exis names
plt.xlim(-0.3,0.50)
plt.yticks(my_range, Shannon['healthy(yes1/no0)'])
plt.xlabel('Spearmanr')
plt.ylabel('Group')
```


```python
#results and sample sizes reported in Figure 4E
Observed=results_alpha[results_alpha.index=='Observed']
Observed
```


```python
#Figure 4E based on the above analysis
sns.set(font_scale=1.0,context='poster',font='Arial',style='white')
plt.figure(figsize=[18,18], dpi=200)
# Reorder it following the values:
#ordered_df = df.sort_values(by='values')
my_range=range(len(r_bray.index))
# Create a color if the group is "B"
my_color=np.where(Observed ['healthy(yes1/no0)']==1, 'gold', 'grey')
my_size=np.where(Observed ['healthy(yes1/no0)']<2, 100, 30)
plt.figure(figsize=[5,10], dpi=200)
plt.rcParams.update({'font.size': 24})
plt.rcParams['axes.facecolor'] = 'white'
#plt.rcParams['axes.facecolor'] = 'white'
plt.hlines(y=my_range, xmin=0, xmax=Observed['spearmanr'], color=my_color, alpha=0.5)
plt.scatter(Observed['spearmanr'], my_range, color=my_color, s=my_size, alpha=1)
# Add title and exis names
plt.xlim(-0.2,0.40)
plt.yticks(my_range, Observed['healthy(yes1/no0)'])
plt.xlabel('Pearsonr')
plt.ylabel('Group')
```


```python
#Same code for Alpha diversity
#Perform pearson correlation between age and alpha diversity for MrOS participants across health stratifications and cohorts
cohorts=[0,1]
stratifications=['med','wlk','lsc_quant','percieved_health','comp_healthy']
health=[]
Metric=[]
Cohort=[]
Coefficient=[]
pvalue=[]
sample_size=[]
condition=[]
for x in cohorts:
    for y in stratifications:    
        df=df_unique[df_unique['firstcohort']==x]
        df_healthy=df[df[y]==1]
        condition.append(1)
        sample_size.append(len(df_healthy))
        health.append(y)
        df_sick=df[df[y]!=1]
        sample_size.append(len(df_sick))
        health.append(y)
        condition.append(0)
        Metric.append('Shannon')
        Metric.append('Shannon')
        Cohort.append(x)
        Cohort.append(x)
        coef=scipy.stats.spearmanr(df_healthy['age'],df_healthy['Shannon_genus'])[0]
        p=scipy.stats.spearmanr(df_healthy['age'],df_healthy['Shannon_genus'])[1]
        Coefficient.append(coef)
        pvalue.append(p)
        coef=scipy.stats.spearmanr(df_sick['age'],df_sick['Shannon_genus'])[0]
        Coefficient.append(coef)
        p=scipy.stats.spearmanr(df_sick['age'],df_sick['Shannon_genus'])[1]
        pvalue.append(p)
        #Weighted_unifrac
        df=df_unique[df_unique['firstcohort']==x]
        df_healthy=df[df[y]==1]
        condition.append(1)
        sample_size.append(len(df_healthy))
        health.append(y)
        df_sick=df[df[y]!=1]
        sample_size.append(len(df_sick))
        health.append(y)
        condition.append(0)
        Metric.append('Observed')
        Metric.append('Observed')
        Cohort.append(x)
        Cohort.append(x)
        coef=scipy.stats.spearmanr(df_healthy['age'],df_healthy['Observed_genus'])[0]
        p=scipy.stats.spearmanr(df_healthy['age'],df_healthy['Observed_genus'])[1]
        Coefficient.append(coef)
        pvalue.append(p)
        coef=scipy.stats.spearmanr(df_sick['age'],df_sick['Observed_genus'])[0]
        Coefficient.append(coef)
        p=scipy.stats.spearmanr(df_sick['age'],df_sick['Observed_genus'])[1]
        pvalue.append(p)
results_alpha=pd.DataFrame()
results_alpha['Metric']=Metric
results_alpha['Health']=health
results_alpha['cohort']=Cohort
results_alpha['spearmanr']=Coefficient
results_alpha['pvalue']=pvalue
results_alpha['sample_size']=sample_size
results_alpha['healthy(yes1/no0)']=condition
results_alpha=results_alpha.sort_values(by=['cohort','Health','healthy(yes1/no0)'],ascending=True)
```


```python
#results and sample sizes reported in Figure 4E
results_alpha.index=results_alpha['Metric']
Shannon=results_alpha[results_alpha.index=='Shannon']
Shannon
```


```python
#Figure 4E based on the above analysis
sns.set(font_scale=1.0,context='poster',font='Arial',style='white')
plt.figure(figsize=[18,18], dpi=200)
# Reorder it following the values:
#ordered_df = df.sort_values(by='values')
my_range=range(len(Shannon.index))
# Create a color if the group is "B"
my_color=np.where(Shannon ['healthy(yes1/no0)']==1, 'darkgreen', 'grey')
my_size=np.where(Shannon ['healthy(yes1/no0)']<2, 100, 30)
plt.figure(figsize=[5,10], dpi=200)
plt.rcParams.update({'font.size': 24})
plt.rcParams['axes.facecolor'] = 'white'
#plt.rcParams['axes.facecolor'] = 'white'
plt.hlines(y=my_range, xmin=0, xmax=Shannon['spearmanr'], color=my_color, alpha=0.5)
plt.scatter(Shannon['spearmanr'], my_range, color=my_color, s=my_size, alpha=1)
# Add title and exis names
plt.xlim(-0.3,0.50)
plt.yticks(my_range, Shannon['healthy(yes1/no0)'])
plt.xlabel('Spearmanr')
plt.ylabel('Group')
```


```python
#results and sample sizes reported in Figure 4E
Observed=results_alpha[results_alpha.index=='Observed']
Observed
```


```python
#Figure 4E based on the above analysis
sns.set(font_scale=1.0,context='poster',font='Arial',style='white')
plt.figure(figsize=[18,18], dpi=200)
# Reorder it following the values:
#ordered_df = df.sort_values(by='values')
my_range=range(len(r_bray.index))
# Create a color if the group is "B"
my_color=np.where(Observed ['healthy(yes1/no0)']==1, 'gold', 'grey')
my_size=np.where(Observed ['healthy(yes1/no0)']<2, 100, 30)
plt.figure(figsize=[5,10], dpi=200)
plt.rcParams.update({'font.size': 24})
plt.rcParams['axes.facecolor'] = 'white'
#plt.rcParams['axes.facecolor'] = 'white'
plt.hlines(y=my_range, xmin=0, xmax=Observed['spearmanr'], color=my_color, alpha=0.5)
plt.scatter(Observed['spearmanr'], my_range, color=my_color, s=my_size, alpha=1)
# Add title and exis names
plt.xlim(-0.3,0.50)
plt.yticks(my_range, Observed['healthy(yes1/no0)'])
plt.xlabel('Pearsonr')
plt.ylabel('Group')
```


```python
#preparing figure 5a
community=df_unique[df_unique['mhhsp']!=1]
community=community[community['giliveh']!=1]
community=community[community['giliven']!=1]
community.shape
community['age_bin']=pd.qcut(community['age'],3,labels=False)
len(community['bacteroides'][community['comp_healthy']==1])
```


```python
sns.set(font_scale=4.00,context='poster',font='Arial',style='white')
plt.figure(figsize=[18,25], dpi=100)
ax=sns.boxplot(y=community['bacteroides'][community['comp_healthy']==1],x=community['age_bin'][community['comp_healthy']==1],notch=True, order=[0,1,2],fliersize=0.0,palette='Blues',showfliers=True,linewidth=4, meanline=False,showmeans=False)
#ax = sns.regplot(x="age_bin", y="bacteroides", data=discovery,color='k')
ax = sns.swarmplot(x='age_bin', y='bacteroides', data=community[community['comp_healthy']==1], color="black",size=15)
ax.set_xlabel('Quartiles of Age')
ax.set_ylabel('Bacteroides (Relative Abundance)')
ax.set_ylim(0,1)
plt.show()
```


```python
sns.set(font_scale=4.00,context='poster',font='Arial',style='white')
plt.figure(figsize=[18,25], dpi=100)
ax=sns.boxplot(y=community['bacteroides'][community['comp_healthy']==0],x=community['age_bin'][community['comp_healthy']==0],notch=True, order=[0,1,2],fliersize=0.0,palette='Reds',showfliers=True,linewidth=4, meanline=False,showmeans=False)
#ax = sns.regplot(x="age_bin", y="bacteroides", data=discovery,color='k')
ax = sns.swarmplot(x='age_bin', y='bacteroides', data=community[community['comp_healthy']==0], color="black",size=15)
ax.set_xlabel('Quartiles of Age')
ax.set_ylabel('Bacteroides (Relative Abundance)')
ax.set_ylim(0,1)
plt.show()
```


```python

```
