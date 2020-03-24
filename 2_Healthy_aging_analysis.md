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
df_unique = pd.read_csv('df_uniqueness.csv')
#check no. of samples
print(df_unique.shape)
#set index
df_unique.set_index(df_unique['Unnamed: 0'],inplace=True)
```

    (907, 143)



```python
#Identify the median number of medications in 85+ year old participants
#This corresponds to the reported stratification based on medication use
discovery_df=df_unique[df_unique['firstcohort']==1]
med_median=discovery_df['m1medsin'][discovery_df['age']>=85].median()
print('median no. of meds in 85+ year olds from the discovery cohort=',med_median)
```

    median no. of meds in 85+ year olds from the discovery cohort= 8.0



```python
#health stratifications: here we generate new variables that stratify each participant based on their performance
#on health measures specified in the text
score=[]
for x in df_unique['qlhealth']:
    if x=='excellent':
        score.append(1)
    else:
        score.append(0)
med=[]
for x in df_unique['m1medsin']:
    if x<=med_median:
        med.append(1)
    else:
        med.append(0)
wlk=[]
#walking speed has 7 missing values that are either due to a participant not being able to come to the visit or not being
#able to perform the test. These participants were all classified into the less healthy (low) group.
for x in df_unique['nfwlkspd']:
    if x>=np.percentile(df_unique['nfwlkspd'].dropna(),66):
        wlk.append(1)
    else:
        wlk.append(0)
lsc=[]
for x in df_unique['lsc']:
    if x>=np.percentile(df_unique['lsc'],66):
        lsc.append(1)
    else:
        lsc.append(0)
df_unique['percieved_health']=score
df_unique['med']=med
df_unique['wlk']=wlk
df_unique['lsc']=lsc
#calculate total no. of times each participant was in the healthy group for composite healthy score
df_unique['total_health']=df_unique['percieved_health']+df_unique['wlk']+df_unique['lsc']+df_unique['med']
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
#save dataframe with health stratifications for demographics table
df_unique.to_csv('demographics.csv')
```


```python
#square root transform uniqueness measures for regression
df_unique['sqrt_min_bray']=np.sqrt(df_unique['min_bray'])
df_unique['sqrt_min_wunifrac']=np.sqrt(df_unique['min_wunifrac'])
df_unique['sqrt_min_bray'].hist(bins=20)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3eff879278>




![png](output_6_1.png)



```python
#Correlation between Bray-Curtis Uniqueness and age reported in the text.
discovery=df_unique[df_unique['firstcohort']==1]
scipy.stats.pearsonr(discovery['sqrt_min_bray'],discovery['age'])
```




    (0.07539880292303881, 0.06516604634127642)




```python
#Perform pearson correlation between age and uniqueness for MrOS participants across health stratifications and cohorts
cohorts=[0,1]
stratifications=['med','wlk','lsc','percieved_health','comp_healthy']
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
        Metric.append('Bray-Curtis')
        Metric.append('Bray-Curtis')
        Cohort.append(x)
        Cohort.append(x)
        coef=scipy.stats.pearsonr(df_healthy['age'],df_healthy['sqrt_min_bray'])[0]
        p=scipy.stats.pearsonr(df_healthy['age'],df_healthy['sqrt_min_bray'])[1]
        Coefficient.append(coef)
        pvalue.append(p)
        coef=scipy.stats.pearsonr(df_sick['age'],df_sick['sqrt_min_bray'])[0]
        Coefficient.append(coef)
        p=scipy.stats.pearsonr(df_sick['age'],df_sick['sqrt_min_bray'])[1]
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
        Metric.append('Weighted_Unifrac')
        Metric.append('Weighted_Unifrac')
        Cohort.append(x)
        Cohort.append(x)
        coef=scipy.stats.pearsonr(df_healthy['age'],df_healthy['min_wunifrac'])[0]
        p=scipy.stats.pearsonr(df_healthy['age'],df_healthy['min_wunifrac'])[1]
        Coefficient.append(coef)
        pvalue.append(p)
        coef=scipy.stats.pearsonr(df_sick['age'],df_sick['min_wunifrac'])[0]
        Coefficient.append(coef)
        p=scipy.stats.pearsonr(df_sick['age'],df_sick['min_wunifrac'])[1]
        pvalue.append(p)
results=pd.DataFrame()
results['Metric']=Metric
results['Health']=health
results['cohort']=Cohort
results['pearsonr']=Coefficient
results['pvalue']=pvalue
results['sample_size']=sample_size
results['healthy(yes1/no0)']=condition
results=results.sort_values(by=['cohort','Health','healthy(yes1/no0)'],ascending=True)
```


```python
#results and sample sizes reported in Figure 4E
results.index=results['Metric']
r_bray=results[results.index=='Bray-Curtis']
r_bray
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Health</th>
      <th>cohort</th>
      <th>pearsonr</th>
      <th>pvalue</th>
      <th>sample_size</th>
      <th>healthy(yes1/no0)</th>
    </tr>
    <tr>
      <th>Metric</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>0.002521</td>
      <td>0.968617</td>
      <td>246</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>0.387973</td>
      <td>0.001835</td>
      <td>62</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>lsc</td>
      <td>0</td>
      <td>0.072521</td>
      <td>0.312435</td>
      <td>196</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>lsc</td>
      <td>0</td>
      <td>0.102401</td>
      <td>0.282651</td>
      <td>112</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>med</td>
      <td>0</td>
      <td>-0.069426</td>
      <td>0.374106</td>
      <td>166</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>med</td>
      <td>0</td>
      <td>0.268525</td>
      <td>0.001233</td>
      <td>142</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>0.045737</td>
      <td>0.519106</td>
      <td>201</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>0.169287</td>
      <td>0.081307</td>
      <td>107</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>wlk</td>
      <td>0</td>
      <td>0.046258</td>
      <td>0.513294</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>wlk</td>
      <td>0</td>
      <td>0.120876</td>
      <td>0.217104</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.022171</td>
      <td>0.633088</td>
      <td>466</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.285378</td>
      <td>0.000870</td>
      <td>133</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>lsc</td>
      <td>1</td>
      <td>0.023981</td>
      <td>0.635091</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>lsc</td>
      <td>1</td>
      <td>0.155422</td>
      <td>0.026065</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>med</td>
      <td>1</td>
      <td>0.007042</td>
      <td>0.902195</td>
      <td>307</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>med</td>
      <td>1</td>
      <td>0.138428</td>
      <td>0.017947</td>
      <td>292</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.039388</td>
      <td>0.435594</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.148728</td>
      <td>0.033313</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.018330</td>
      <td>0.715098</td>
      <td>399</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.221161</td>
      <td>0.001649</td>
      <td>200</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Figure 4E based on the above analysis
sns.set(font_scale=1.0,context='poster',font='Arial',style='white')
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
plt.hlines(y=my_range, xmin=0, xmax=r_bray['pearsonr'], color=my_color, alpha=0.5)
plt.scatter(r_bray['pearsonr'], my_range, color=my_color, s=my_size, alpha=1)
# Add title and exis names
plt.xlim(-0.3,0.50)
plt.yticks(my_range, r_bray['healthy(yes1/no0)'])
plt.xlabel('Pearsonr')
plt.ylabel('Group')
```




    Text(0, 0.5, 'Group')




    <Figure size 3600x3600 with 0 Axes>


    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.
    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.



![png](output_10_3.png)



```python
#Same code for Alpha diversity
#Perform pearson correlation between age and alpha diversity for MrOS participants across health stratifications and cohorts
cohorts=[0,1]
stratifications=['med','wlk','lsc','percieved_health','comp_healthy']
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
        coef=scipy.stats.pearsonr(df_healthy['age'],df_healthy['Shannon'])[0]
        p=scipy.stats.pearsonr(df_healthy['age'],df_healthy['Shannon'])[1]
        Coefficient.append(coef)
        pvalue.append(p)
        coef=scipy.stats.pearsonr(df_sick['age'],df_sick['Shannon'])[0]
        Coefficient.append(coef)
        p=scipy.stats.pearsonr(df_sick['age'],df_sick['Shannon'])[1]
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
        coef=scipy.stats.pearsonr(df_healthy['age'],df_healthy['Observed'])[0]
        p=scipy.stats.pearsonr(df_healthy['age'],df_healthy['Observed'])[1]
        Coefficient.append(coef)
        pvalue.append(p)
        coef=scipy.stats.pearsonr(df_sick['age'],df_sick['Observed'])[0]
        Coefficient.append(coef)
        p=scipy.stats.pearsonr(df_sick['age'],df_sick['Observed'])[1]
        pvalue.append(p)
results_alpha=pd.DataFrame()
results_alpha['Metric']=Metric
results_alpha['Health']=health
results_alpha['cohort']=Cohort
results_alpha['pearsonr']=Coefficient
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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Health</th>
      <th>cohort</th>
      <th>pearsonr</th>
      <th>pvalue</th>
      <th>sample_size</th>
      <th>healthy(yes1/no0)</th>
    </tr>
    <tr>
      <th>Metric</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>-0.063973</td>
      <td>0.317651</td>
      <td>246</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>0.168446</td>
      <td>0.190628</td>
      <td>62</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>lsc</td>
      <td>0</td>
      <td>-0.030844</td>
      <td>0.667807</td>
      <td>196</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>lsc</td>
      <td>0</td>
      <td>0.002086</td>
      <td>0.982588</td>
      <td>112</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>med</td>
      <td>0</td>
      <td>-0.070751</td>
      <td>0.365033</td>
      <td>166</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>med</td>
      <td>0</td>
      <td>0.039110</td>
      <td>0.644003</td>
      <td>142</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>-0.073901</td>
      <td>0.297125</td>
      <td>201</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>0.109536</td>
      <td>0.261383</td>
      <td>107</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>wlk</td>
      <td>0</td>
      <td>-0.047427</td>
      <td>0.502696</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>wlk</td>
      <td>0</td>
      <td>0.077399</td>
      <td>0.430344</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.001385</td>
      <td>0.976209</td>
      <td>466</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.021466</td>
      <td>0.806261</td>
      <td>133</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>lsc</td>
      <td>1</td>
      <td>-0.020144</td>
      <td>0.690174</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>lsc</td>
      <td>1</td>
      <td>0.039307</td>
      <td>0.575772</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>med</td>
      <td>1</td>
      <td>-0.017032</td>
      <td>0.766295</td>
      <td>307</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>med</td>
      <td>1</td>
      <td>0.015546</td>
      <td>0.791379</td>
      <td>292</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.009764</td>
      <td>0.846801</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>-0.015296</td>
      <td>0.827676</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>wlk</td>
      <td>1</td>
      <td>-0.003192</td>
      <td>0.949314</td>
      <td>399</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.059788</td>
      <td>0.400361</td>
      <td>200</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




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
plt.hlines(y=my_range, xmin=0, xmax=Shannon['pearsonr'], color=my_color, alpha=0.5)
plt.scatter(Shannon['pearsonr'], my_range, color=my_color, s=my_size, alpha=1)
# Add title and exis names
plt.xlim(-0.3,0.50)
plt.yticks(my_range, Shannon['healthy(yes1/no0)'])
plt.xlabel('Pearsonr')
plt.ylabel('Group')
```




    Text(0, 0.5, 'Group')




    <Figure size 3600x3600 with 0 Axes>



![png](output_13_2.png)



```python
#results and sample sizes reported in Figure 4E
Observed=results_alpha[results_alpha.index=='Observed']
Observed
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Health</th>
      <th>cohort</th>
      <th>pearsonr</th>
      <th>pvalue</th>
      <th>sample_size</th>
      <th>healthy(yes1/no0)</th>
    </tr>
    <tr>
      <th>Metric</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>-0.019318</td>
      <td>0.763050</td>
      <td>246</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>0.071916</td>
      <td>0.578578</td>
      <td>62</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>lsc</td>
      <td>0</td>
      <td>0.036063</td>
      <td>0.615799</td>
      <td>196</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>lsc</td>
      <td>0</td>
      <td>-0.097325</td>
      <td>0.307321</td>
      <td>112</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>med</td>
      <td>0</td>
      <td>-0.045000</td>
      <td>0.564822</td>
      <td>166</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>med</td>
      <td>0</td>
      <td>0.031613</td>
      <td>0.708796</td>
      <td>142</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>-0.023471</td>
      <td>0.740849</td>
      <td>201</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>0.044350</td>
      <td>0.650121</td>
      <td>107</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>wlk</td>
      <td>0</td>
      <td>-0.030345</td>
      <td>0.668136</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>wlk</td>
      <td>0</td>
      <td>0.073596</td>
      <td>0.453406</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.043429</td>
      <td>0.349567</td>
      <td>466</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.073832</td>
      <td>0.398339</td>
      <td>133</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>lsc</td>
      <td>1</td>
      <td>0.013814</td>
      <td>0.784597</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>lsc</td>
      <td>1</td>
      <td>0.104918</td>
      <td>0.134354</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>med</td>
      <td>1</td>
      <td>0.021944</td>
      <td>0.701748</td>
      <td>307</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>med</td>
      <td>1</td>
      <td>0.057626</td>
      <td>0.326446</td>
      <td>292</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.044276</td>
      <td>0.380762</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.040452</td>
      <td>0.564694</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.053936</td>
      <td>0.282476</td>
      <td>399</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.074526</td>
      <td>0.294263</td>
      <td>200</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




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
plt.hlines(y=my_range, xmin=0, xmax=Observed['pearsonr'], color=my_color, alpha=0.5)
plt.scatter(Observed['pearsonr'], my_range, color=my_color, s=my_size, alpha=1)
# Add title and exis names
plt.xlim(-0.3,0.50)
plt.yticks(my_range, Observed['healthy(yes1/no0)'])
plt.xlabel('Pearsonr')
plt.ylabel('Group')
```




    Text(0, 0.5, 'Group')




    <Figure size 3600x3600 with 0 Axes>



![png](output_15_2.png)

