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
df_unique.head()
```

    (907, 148)





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
      <th>Unnamed: 0</th>
      <th>batch</th>
      <th>id</th>
      <th>site</th>
      <th>firstcohort</th>
      <th>fastingblood</th>
      <th>status</th>
      <th>dead</th>
      <th>age</th>
      <th>survival</th>
      <th>...</th>
      <th>Shannon</th>
      <th>Observed</th>
      <th>min_bray_genus</th>
      <th>min_wunifrac_genus</th>
      <th>Shannon_genus</th>
      <th>Observed_genus</th>
      <th>bacteroides</th>
      <th>prevotella</th>
      <th>P_B</th>
      <th>reads</th>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>RN0023</th>
      <td>RN0023</td>
      <td>1</td>
      <td>Orwoll.BI0023.BI</td>
      <td>RN</td>
      <td>1</td>
      <td>1.0</td>
      <td>active</td>
      <td>0.0</td>
      <td>83</td>
      <td>2.472279</td>
      <td>...</td>
      <td>3.984556</td>
      <td>237</td>
      <td>0.282877</td>
      <td>0.117900</td>
      <td>3.037238</td>
      <td>105</td>
      <td>0.2748</td>
      <td>0.0</td>
      <td>0.2748</td>
      <td>9991</td>
    </tr>
    <tr>
      <th>RN0056</th>
      <td>RN0056</td>
      <td>1</td>
      <td>Orwoll.BI0056.BI</td>
      <td>RN</td>
      <td>1</td>
      <td>1.0</td>
      <td>active</td>
      <td>0.0</td>
      <td>81</td>
      <td>2.759754</td>
      <td>...</td>
      <td>2.701232</td>
      <td>131</td>
      <td>0.159316</td>
      <td>0.056270</td>
      <td>1.512966</td>
      <td>68</td>
      <td>0.7242</td>
      <td>0.0</td>
      <td>0.7242</td>
      <td>10000</td>
    </tr>
    <tr>
      <th>RN0131</th>
      <td>RN0131</td>
      <td>1</td>
      <td>Orwoll.BI0131.BI</td>
      <td>RN</td>
      <td>1</td>
      <td>1.0</td>
      <td>active</td>
      <td>0.0</td>
      <td>83</td>
      <td>2.880219</td>
      <td>...</td>
      <td>3.102646</td>
      <td>165</td>
      <td>0.238248</td>
      <td>0.115928</td>
      <td>2.336418</td>
      <td>74</td>
      <td>0.2433</td>
      <td>0.0</td>
      <td>0.2433</td>
      <td>9996</td>
    </tr>
    <tr>
      <th>RN0153</th>
      <td>RN0153</td>
      <td>1</td>
      <td>Orwoll.BI0153.BI</td>
      <td>RN</td>
      <td>1</td>
      <td>1.0</td>
      <td>active</td>
      <td>0.0</td>
      <td>79</td>
      <td>2.899384</td>
      <td>...</td>
      <td>3.559490</td>
      <td>154</td>
      <td>0.185400</td>
      <td>0.059060</td>
      <td>2.295188</td>
      <td>74</td>
      <td>0.4191</td>
      <td>0.0</td>
      <td>0.4191</td>
      <td>10000</td>
    </tr>
    <tr>
      <th>RN0215</th>
      <td>RN0215</td>
      <td>1</td>
      <td>Orwoll.BI0215.BI</td>
      <td>RN</td>
      <td>1</td>
      <td>1.0</td>
      <td>active</td>
      <td>0.0</td>
      <td>81</td>
      <td>2.362765</td>
      <td>...</td>
      <td>3.717956</td>
      <td>161</td>
      <td>0.214700</td>
      <td>0.073221</td>
      <td>2.683730</td>
      <td>76</td>
      <td>0.3767</td>
      <td>0.0</td>
      <td>0.3767</td>
      <td>10000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 148 columns</p>
</div>




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

    median no. of meds in 85+ year olds from the discovery cohort= 8.0
    1.1665664598246186
    96.0



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

    311
    433



```python
df_unique['comp_healthy'].sum()
```




    194




```python
#save dataframe with health stratifications for demographics table
df_unique.to_csv('demographics.csv')
df_unique.shape
```




    (907, 154)




```python
#square root transform uniqueness measures for regression
df_unique['sqrt_min_bray']=np.sqrt(df_unique['min_bray'])
df_unique['sqrt_min_bray_g']=np.sqrt(df_unique['min_bray_genus'])
df_unique['sqrt_min_wunifrac_g']=np.sqrt(df_unique['min_wunifrac_genus'])
df_unique['sqrt_min_wunifrac']=np.sqrt(df_unique['min_wunifrac'])
df_unique['sqrt_min_bray_g'].hist(bins=20)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fbbb9d27f60>




![png](output_7_1.png)



```python
#Correlation between Bray-Curtis Uniqueness and age reported in the text.
discovery=df_unique[df_unique['firstcohort']==1]
print(scipy.stats.spearmanr(discovery['sqrt_min_bray_g'],discovery['age']))
scipy.stats.spearmanr(discovery['sqrt_min_bray'],discovery['age'])
```

    SpearmanrResult(correlation=0.10550480943731853, pvalue=0.009766276656475675)





    SpearmanrResult(correlation=0.07366893376623675, pvalue=0.0715940114119013)




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
        regression=smf.ols('sqrt_min_bray_g~age',data=df_healthy).fit()
        p_reg.append(regression.pvalues[1])
        lower.append(regression.conf_int(alpha=0.05, cols=None)[1:2][0].tolist())
        upper.append(regression.conf_int(alpha=0.05, cols=None)[1:2][1].tolist())
        reg_coef.append(regression.params[1].tolist())
        Coefficient.append(coef)
        pvalue.append(p)
        coef=scipy.stats.spearmanr(df_sick['age'],df_sick['sqrt_min_bray_g'])[0]
        Coefficient.append(coef)
        p=scipy.stats.spearmanr(df_sick['age'],df_sick['sqrt_min_bray_g'])[1]
        pvalue.append(p)
        regression=smf.ols('sqrt_min_bray_g~age',data=df_sick).fit()
        p_reg.append(regression.pvalues[1])
        lower.append(regression.conf_int(alpha=0.05, cols=None)[1:2][0].tolist())
        upper.append(regression.conf_int(alpha=0.05, cols=None)[1:2][1].tolist())
        reg_coef.append(regression.params[1].tolist())
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
        regression=smf.ols('sqrt_min_wunifrac_g~age',data=df_healthy).fit()
        p_reg.append(regression.pvalues[1])
        lower.append(regression.conf_int(alpha=0.05, cols=None)[1:2][0].tolist())
        upper.append(regression.conf_int(alpha=0.05, cols=None)[1:2][1].tolist())
        reg_coef.append(regression.params[1].tolist())
        Coefficient.append(coef)
        pvalue.append(p)
        coef=scipy.stats.spearmanr(df_sick['age'],df_sick['sqrt_min_wunifrac_g'])[0]
        Coefficient.append(coef)
        p=scipy.stats.spearmanr(df_sick['age'],df_sick['sqrt_min_wunifrac_g'])[1]
        pvalue.append(p)
        regression=smf.ols('sqrt_min_wunifrac_g~age',data=df_sick).fit()
        p_reg.append(regression.pvalues[1])
        lower.append(regression.conf_int(alpha=0.05, cols=None)[1:2][0].tolist())
        upper.append(regression.conf_int(alpha=0.05, cols=None)[1:2][1].tolist())
        reg_coef.append(regression.params[1].tolist())
results=pd.DataFrame()
lower=[ x[0] for x in  lower]
upper=[ x[0] for x in  upper]
results['Metric']=Metric
results['Health']=health
results['cohort']=Cohort
results['spearmanr']=Coefficient
results['pvalue']=pvalue
results['coef_pvalue']=p_reg
results['beta_coef']=reg_coef
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
      <th>spearmanr</th>
      <th>pvalue</th>
      <th>coef_pvalue</th>
      <th>beta_coef</th>
      <th>sample_size</th>
      <th>lower</th>
      <th>upper</th>
      <th>healthy(yes1/no0)</th>
      <th>err</th>
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
      <td>-0.056412</td>
      <td>0.377351</td>
      <td>0.923278</td>
      <td>0.000109</td>
      <td>247</td>
      <td>-0.002124</td>
      <td>0.002342</td>
      <td>0</td>
      <td>0.002233</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>0.298740</td>
      <td>0.019354</td>
      <td>0.001279</td>
      <td>0.007317</td>
      <td>61</td>
      <td>0.002989</td>
      <td>0.011646</td>
      <td>1</td>
      <td>0.004328</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>0.000906</td>
      <td>0.989914</td>
      <td>0.304085</td>
      <td>0.001262</td>
      <td>197</td>
      <td>-0.001154</td>
      <td>0.003678</td>
      <td>0</td>
      <td>0.002416</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>0.039264</td>
      <td>0.682431</td>
      <td>0.205507</td>
      <td>0.002280</td>
      <td>111</td>
      <td>-0.001268</td>
      <td>0.005827</td>
      <td>1</td>
      <td>0.003548</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>med</td>
      <td>0</td>
      <td>-0.087736</td>
      <td>0.259548</td>
      <td>0.382384</td>
      <td>-0.001222</td>
      <td>167</td>
      <td>-0.003978</td>
      <td>0.001533</td>
      <td>0</td>
      <td>0.002755</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>med</td>
      <td>0</td>
      <td>0.180223</td>
      <td>0.032474</td>
      <td>0.000730</td>
      <td>0.004821</td>
      <td>141</td>
      <td>0.002062</td>
      <td>0.007579</td>
      <td>1</td>
      <td>0.002759</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>-0.004467</td>
      <td>0.949689</td>
      <td>0.528914</td>
      <td>0.000832</td>
      <td>202</td>
      <td>-0.001769</td>
      <td>0.003433</td>
      <td>0</td>
      <td>0.002601</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>0.111312</td>
      <td>0.255966</td>
      <td>0.052395</td>
      <td>0.002988</td>
      <td>106</td>
      <td>-0.000032</td>
      <td>0.006008</td>
      <td>1</td>
      <td>0.003020</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>wlk</td>
      <td>0</td>
      <td>-0.035136</td>
      <td>0.619591</td>
      <td>0.424624</td>
      <td>0.001024</td>
      <td>202</td>
      <td>-0.001500</td>
      <td>0.003549</td>
      <td>0</td>
      <td>0.002524</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>wlk</td>
      <td>0</td>
      <td>0.113989</td>
      <td>0.244639</td>
      <td>0.256184</td>
      <td>0.002005</td>
      <td>106</td>
      <td>-0.001478</td>
      <td>0.005488</td>
      <td>1</td>
      <td>0.003483</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.050010</td>
      <td>0.281323</td>
      <td>0.540974</td>
      <td>0.000529</td>
      <td>466</td>
      <td>-0.001171</td>
      <td>0.002230</td>
      <td>0</td>
      <td>0.001700</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.279296</td>
      <td>0.001131</td>
      <td>0.000781</td>
      <td>0.006139</td>
      <td>133</td>
      <td>0.002609</td>
      <td>0.009670</td>
      <td>1</td>
      <td>0.003530</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.055721</td>
      <td>0.269864</td>
      <td>0.542758</td>
      <td>0.000558</td>
      <td>394</td>
      <td>-0.001242</td>
      <td>0.002357</td>
      <td>0</td>
      <td>0.001800</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.169176</td>
      <td>0.015312</td>
      <td>0.024264</td>
      <td>0.003457</td>
      <td>205</td>
      <td>0.000454</td>
      <td>0.006460</td>
      <td>1</td>
      <td>0.003003</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>med</td>
      <td>1</td>
      <td>0.084288</td>
      <td>0.140629</td>
      <td>0.775945</td>
      <td>0.000316</td>
      <td>307</td>
      <td>-0.001868</td>
      <td>0.002501</td>
      <td>0</td>
      <td>0.002185</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>med</td>
      <td>1</td>
      <td>0.125611</td>
      <td>0.031893</td>
      <td>0.016454</td>
      <td>0.002626</td>
      <td>292</td>
      <td>0.000484</td>
      <td>0.004769</td>
      <td>1</td>
      <td>0.002142</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.074522</td>
      <td>0.139789</td>
      <td>0.343236</td>
      <td>0.000955</td>
      <td>394</td>
      <td>-0.001023</td>
      <td>0.002933</td>
      <td>0</td>
      <td>0.001978</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.167131</td>
      <td>0.016612</td>
      <td>0.032590</td>
      <td>0.002608</td>
      <td>205</td>
      <td>0.000218</td>
      <td>0.004998</td>
      <td>1</td>
      <td>0.002390</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.037485</td>
      <td>0.455258</td>
      <td>0.621007</td>
      <td>0.000463</td>
      <td>399</td>
      <td>-0.001376</td>
      <td>0.002302</td>
      <td>0</td>
      <td>0.001839</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.239485</td>
      <td>0.000637</td>
      <td>0.001543</td>
      <td>0.005040</td>
      <td>200</td>
      <td>0.001945</td>
      <td>0.008135</td>
      <td>1</td>
      <td>0.003095</td>
    </tr>
  </tbody>
</table>
</div>




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




    Text(0, 0.5, 'Group')




    <Figure size 3600x3600 with 0 Axes>


    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.
    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.



![png](output_12_3.png)



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

    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.



![png](output_13_1.png)



```python
#results and sample sizes reported in Figure S1a
results.index=results['Metric']
Weighted_Unifrac=results[results.index=='Weighted_Unifrac']
Weighted_Unifrac
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
      <th>spearmanr</th>
      <th>pvalue</th>
      <th>coef_pvalue</th>
      <th>beta_coef</th>
      <th>sample_size</th>
      <th>lower</th>
      <th>upper</th>
      <th>healthy(yes1/no0)</th>
      <th>err</th>
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
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>-0.073845</td>
      <td>0.247573</td>
      <td>0.759093</td>
      <td>-0.000350</td>
      <td>247</td>
      <td>-0.002597</td>
      <td>0.001896</td>
      <td>0</td>
      <td>0.002247</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>0.272888</td>
      <td>0.033354</td>
      <td>0.000690</td>
      <td>0.007494</td>
      <td>61</td>
      <td>0.003308</td>
      <td>0.011679</td>
      <td>1</td>
      <td>0.004185</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>-0.039510</td>
      <td>0.581471</td>
      <td>0.801157</td>
      <td>0.000310</td>
      <td>197</td>
      <td>-0.002113</td>
      <td>0.002733</td>
      <td>0</td>
      <td>0.002423</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>0.048636</td>
      <td>0.612220</td>
      <td>0.065309</td>
      <td>0.003306</td>
      <td>111</td>
      <td>-0.000213</td>
      <td>0.006826</td>
      <td>1</td>
      <td>0.003519</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>med</td>
      <td>0</td>
      <td>-0.087558</td>
      <td>0.260518</td>
      <td>0.212578</td>
      <td>-0.001668</td>
      <td>167</td>
      <td>-0.004299</td>
      <td>0.000964</td>
      <td>0</td>
      <td>0.002631</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>med</td>
      <td>0</td>
      <td>0.098812</td>
      <td>0.243720</td>
      <td>0.003222</td>
      <td>0.004388</td>
      <td>141</td>
      <td>0.001494</td>
      <td>0.007282</td>
      <td>1</td>
      <td>0.002894</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>-0.024884</td>
      <td>0.725189</td>
      <td>0.703729</td>
      <td>0.000495</td>
      <td>202</td>
      <td>-0.002070</td>
      <td>0.003061</td>
      <td>0</td>
      <td>0.002565</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>0.057594</td>
      <td>0.557592</td>
      <td>0.114321</td>
      <td>0.002508</td>
      <td>106</td>
      <td>-0.000615</td>
      <td>0.005630</td>
      <td>1</td>
      <td>0.003123</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>wlk</td>
      <td>0</td>
      <td>-0.071047</td>
      <td>0.315008</td>
      <td>0.702637</td>
      <td>0.000501</td>
      <td>202</td>
      <td>-0.002084</td>
      <td>0.003087</td>
      <td>0</td>
      <td>0.002585</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>wlk</td>
      <td>0</td>
      <td>0.112922</td>
      <td>0.249113</td>
      <td>0.350575</td>
      <td>0.001509</td>
      <td>106</td>
      <td>-0.001683</td>
      <td>0.004701</td>
      <td>1</td>
      <td>0.003192</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.032647</td>
      <td>0.482023</td>
      <td>0.665819</td>
      <td>0.000366</td>
      <td>466</td>
      <td>-0.001298</td>
      <td>0.002031</td>
      <td>0</td>
      <td>0.001664</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.307563</td>
      <td>0.000317</td>
      <td>0.000018</td>
      <td>0.007120</td>
      <td>133</td>
      <td>0.003956</td>
      <td>0.010284</td>
      <td>1</td>
      <td>0.003164</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.032968</td>
      <td>0.514079</td>
      <td>0.605400</td>
      <td>0.000459</td>
      <td>394</td>
      <td>-0.001287</td>
      <td>0.002205</td>
      <td>0</td>
      <td>0.001746</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.200214</td>
      <td>0.003998</td>
      <td>0.003892</td>
      <td>0.004260</td>
      <td>205</td>
      <td>0.001384</td>
      <td>0.007137</td>
      <td>1</td>
      <td>0.002876</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>med</td>
      <td>1</td>
      <td>0.062278</td>
      <td>0.276678</td>
      <td>0.779964</td>
      <td>0.000293</td>
      <td>307</td>
      <td>-0.001767</td>
      <td>0.002353</td>
      <td>0</td>
      <td>0.002060</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>med</td>
      <td>1</td>
      <td>0.127620</td>
      <td>0.029232</td>
      <td>0.012707</td>
      <td>0.002691</td>
      <td>292</td>
      <td>0.000579</td>
      <td>0.004803</td>
      <td>1</td>
      <td>0.002112</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.070870</td>
      <td>0.160314</td>
      <td>0.366566</td>
      <td>0.000885</td>
      <td>394</td>
      <td>-0.001040</td>
      <td>0.002810</td>
      <td>0</td>
      <td>0.001925</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.148375</td>
      <td>0.033739</td>
      <td>0.014128</td>
      <td>0.002844</td>
      <td>205</td>
      <td>0.000579</td>
      <td>0.005110</td>
      <td>1</td>
      <td>0.002265</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.017678</td>
      <td>0.724810</td>
      <td>0.636342</td>
      <td>0.000438</td>
      <td>399</td>
      <td>-0.001383</td>
      <td>0.002259</td>
      <td>0</td>
      <td>0.001821</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.265350</td>
      <td>0.000146</td>
      <td>0.000229</td>
      <td>0.005295</td>
      <td>200</td>
      <td>0.002514</td>
      <td>0.008077</td>
      <td>1</td>
      <td>0.002782</td>
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




    Text(0, 0.5, 'Group')




    <Figure size 3600x3600 with 0 Axes>


    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.
    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.



![png](output_15_3.png)



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
      <th>spearmanr</th>
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
      <td>-0.048526</td>
      <td>0.447718</td>
      <td>247</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>0.205532</td>
      <td>0.112046</td>
      <td>61</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>0.001549</td>
      <td>0.982765</td>
      <td>197</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>-0.015260</td>
      <td>0.873701</td>
      <td>111</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>med</td>
      <td>0</td>
      <td>-0.069226</td>
      <td>0.374034</td>
      <td>167</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>med</td>
      <td>0</td>
      <td>0.072986</td>
      <td>0.389737</td>
      <td>141</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>-0.048996</td>
      <td>0.488648</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>0.127584</td>
      <td>0.192466</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>wlk</td>
      <td>0</td>
      <td>-0.029997</td>
      <td>0.671718</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>wlk</td>
      <td>0</td>
      <td>0.094245</td>
      <td>0.336578</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>-0.006806</td>
      <td>0.883512</td>
      <td>466</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.056954</td>
      <td>0.514948</td>
      <td>133</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>-0.030070</td>
      <td>0.551767</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.069291</td>
      <td>0.323535</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>med</td>
      <td>1</td>
      <td>-0.025828</td>
      <td>0.652152</td>
      <td>307</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>med</td>
      <td>1</td>
      <td>0.024931</td>
      <td>0.671376</td>
      <td>292</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>-0.013426</td>
      <td>0.790496</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.021608</td>
      <td>0.758445</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>wlk</td>
      <td>1</td>
      <td>-0.011448</td>
      <td>0.819675</td>
      <td>399</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.081593</td>
      <td>0.250729</td>
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
plt.hlines(y=my_range, xmin=0, xmax=Shannon['spearmanr'], color=my_color, alpha=0.5)
plt.scatter(Shannon['spearmanr'], my_range, color=my_color, s=my_size, alpha=1)
# Add title and exis names
plt.xlim(-0.3,0.50)
plt.yticks(my_range, Shannon['healthy(yes1/no0)'])
plt.xlabel('Spearmanr')
plt.ylabel('Group')
```




    Text(0, 0.5, 'Group')




    <Figure size 3600x3600 with 0 Axes>



![png](output_19_2.png)



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
      <th>spearmanr</th>
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
      <td>-0.027856</td>
      <td>0.663090</td>
      <td>247</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>0.158211</td>
      <td>0.223302</td>
      <td>61</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>0.053282</td>
      <td>0.457111</td>
      <td>197</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>-0.106413</td>
      <td>0.266312</td>
      <td>111</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>med</td>
      <td>0</td>
      <td>-0.069055</td>
      <td>0.375217</td>
      <td>167</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>med</td>
      <td>0</td>
      <td>0.079248</td>
      <td>0.350245</td>
      <td>141</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>-0.018955</td>
      <td>0.788884</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>0.096278</td>
      <td>0.326210</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>wlk</td>
      <td>0</td>
      <td>-0.018885</td>
      <td>0.789655</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>wlk</td>
      <td>0</td>
      <td>0.084163</td>
      <td>0.391024</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.048460</td>
      <td>0.296529</td>
      <td>466</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.099318</td>
      <td>0.255373</td>
      <td>133</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.026603</td>
      <td>0.598562</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.104189</td>
      <td>0.137097</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>med</td>
      <td>1</td>
      <td>0.023786</td>
      <td>0.678055</td>
      <td>307</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>med</td>
      <td>1</td>
      <td>0.069654</td>
      <td>0.235394</td>
      <td>292</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.039579</td>
      <td>0.433370</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.072395</td>
      <td>0.302280</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.053965</td>
      <td>0.282216</td>
      <td>399</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.093824</td>
      <td>0.186345</td>
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
plt.hlines(y=my_range, xmin=0, xmax=Observed['spearmanr'], color=my_color, alpha=0.5)
plt.scatter(Observed['spearmanr'], my_range, color=my_color, s=my_size, alpha=1)
# Add title and exis names
plt.xlim(-0.2,0.40)
plt.yticks(my_range, Observed['healthy(yes1/no0)'])
plt.xlabel('Pearsonr')
plt.ylabel('Group')
```




    Text(0, 0.5, 'Group')




    <Figure size 3600x3600 with 0 Axes>



![png](output_21_2.png)



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
      <th>spearmanr</th>
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
      <td>-0.064557</td>
      <td>0.312259</td>
      <td>247</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>0.230940</td>
      <td>0.073345</td>
      <td>61</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>0.015976</td>
      <td>0.823679</td>
      <td>197</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>-0.044997</td>
      <td>0.639109</td>
      <td>111</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>med</td>
      <td>0</td>
      <td>-0.109305</td>
      <td>0.159681</td>
      <td>167</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>med</td>
      <td>0</td>
      <td>0.113329</td>
      <td>0.180883</td>
      <td>141</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>-0.050960</td>
      <td>0.471373</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>0.117116</td>
      <td>0.231849</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>wlk</td>
      <td>0</td>
      <td>-0.027039</td>
      <td>0.702473</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>wlk</td>
      <td>0</td>
      <td>0.065946</td>
      <td>0.501814</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.018140</td>
      <td>0.696118</td>
      <td>466</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.148795</td>
      <td>0.087393</td>
      <td>133</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.003341</td>
      <td>0.947292</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.107591</td>
      <td>0.124658</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>med</td>
      <td>1</td>
      <td>0.032568</td>
      <td>0.569727</td>
      <td>307</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>med</td>
      <td>1</td>
      <td>0.055595</td>
      <td>0.343808</td>
      <td>292</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.016437</td>
      <td>0.744988</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.092422</td>
      <td>0.187498</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.015391</td>
      <td>0.759231</td>
      <td>399</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.129346</td>
      <td>0.067936</td>
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
plt.hlines(y=my_range, xmin=0, xmax=Shannon['spearmanr'], color=my_color, alpha=0.5)
plt.scatter(Shannon['spearmanr'], my_range, color=my_color, s=my_size, alpha=1)
# Add title and exis names
plt.xlim(-0.3,0.50)
plt.yticks(my_range, Shannon['healthy(yes1/no0)'])
plt.xlabel('Spearmanr')
plt.ylabel('Group')
```




    Text(0, 0.5, 'Group')




    <Figure size 3600x3600 with 0 Axes>



![png](output_24_2.png)



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
      <th>spearmanr</th>
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
      <td>-0.032072</td>
      <td>0.615937</td>
      <td>247</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>0.128647</td>
      <td>0.323108</td>
      <td>61</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>0.053906</td>
      <td>0.451846</td>
      <td>197</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>-0.137437</td>
      <td>0.150311</td>
      <td>111</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>med</td>
      <td>0</td>
      <td>-0.083338</td>
      <td>0.284286</td>
      <td>167</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>med</td>
      <td>0</td>
      <td>0.080006</td>
      <td>0.345640</td>
      <td>141</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>-0.035828</td>
      <td>0.612703</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>0.088674</td>
      <td>0.366040</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>wlk</td>
      <td>0</td>
      <td>-0.008288</td>
      <td>0.906804</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>wlk</td>
      <td>0</td>
      <td>0.030474</td>
      <td>0.756483</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.055834</td>
      <td>0.228980</td>
      <td>466</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.114277</td>
      <td>0.190278</td>
      <td>133</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.043032</td>
      <td>0.394298</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.107099</td>
      <td>0.126402</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>med</td>
      <td>1</td>
      <td>0.038186</td>
      <td>0.505037</td>
      <td>307</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>med</td>
      <td>1</td>
      <td>0.080317</td>
      <td>0.171065</td>
      <td>292</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.050375</td>
      <td>0.318583</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.072607</td>
      <td>0.300861</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.065448</td>
      <td>0.192020</td>
      <td>399</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.092594</td>
      <td>0.192209</td>
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
plt.hlines(y=my_range, xmin=0, xmax=Observed['spearmanr'], color=my_color, alpha=0.5)
plt.scatter(Observed['spearmanr'], my_range, color=my_color, s=my_size, alpha=1)
# Add title and exis names
plt.xlim(-0.3,0.50)
plt.yticks(my_range, Observed['healthy(yes1/no0)'])
plt.xlabel('Pearsonr')
plt.ylabel('Group')
```




    Text(0, 0.5, 'Group')




    <Figure size 3600x3600 with 0 Axes>



![png](output_26_2.png)



```python
#preparing figure 5a
community=df_unique[df_unique['mhhsp']!=1]
community=community[community['giliveh']!=1]
community=community[community['giliven']!=1]
community.shape
community['age_bin']=pd.qcut(community['age'],3,labels=False)
len(community['bacteroides'][community['comp_healthy']==1])
```




    172




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

    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.
    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.



![png](output_28_1.png)



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


![png](output_29_0.png)



```python

```
