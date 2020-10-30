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
print(np.percentile(df_unique['lsc'].dropna(),66))
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
    if x>=1.1665664598246186:
        wlk.append(1)
    else:
        wlk.append(0)
lsc=[]
for x in df_unique['lsc']:
    if x>=np.percentile(df_unique['lsc'].dropna(),66.6):
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




    <matplotlib.axes._subplots.AxesSubplot at 0x7f83549ae0f0>




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
        coef=scipy.stats.pearsonr(df_healthy['age'],df_healthy['sqrt_min_bray_g'])[0]
        p=scipy.stats.pearsonr(df_healthy['age'],df_healthy['sqrt_min_bray_g'])[1]
        Coefficient.append(coef)
        pvalue.append(p)
        coef=scipy.stats.pearsonr(df_sick['age'],df_sick['sqrt_min_bray_g'])[0]
        Coefficient.append(coef)
        p=scipy.stats.pearsonr(df_sick['age'],df_sick['sqrt_min_bray_g'])[1]
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
        coef=scipy.stats.pearsonr(df_healthy['age'],df_healthy['sqrt_min_wunifrac_g'])[0]
        p=scipy.stats.pearsonr(df_healthy['age'],df_healthy['sqrt_min_wunifrac_g'])[1]
        Coefficient.append(coef)
        pvalue.append(p)
        coef=scipy.stats.pearsonr(df_sick['age'],df_sick['sqrt_min_wunifrac_g'])[0]
        Coefficient.append(coef)
        p=scipy.stats.pearsonr(df_sick['age'],df_sick['sqrt_min_wunifrac_g'])[1]
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
      <td>0.006159</td>
      <td>0.923278</td>
      <td>247</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>0.403055</td>
      <td>0.001279</td>
      <td>61</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>0.073590</td>
      <td>0.304085</td>
      <td>197</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>0.121093</td>
      <td>0.205507</td>
      <td>111</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>med</td>
      <td>0</td>
      <td>-0.068027</td>
      <td>0.382384</td>
      <td>167</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>med</td>
      <td>0</td>
      <td>0.281223</td>
      <td>0.000730</td>
      <td>141</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>0.044558</td>
      <td>0.528914</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>0.188956</td>
      <td>0.052395</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>wlk</td>
      <td>0</td>
      <td>0.056483</td>
      <td>0.424624</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>wlk</td>
      <td>0</td>
      <td>0.111262</td>
      <td>0.256184</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.028390</td>
      <td>0.540974</td>
      <td>466</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.287838</td>
      <td>0.000781</td>
      <td>133</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.030754</td>
      <td>0.542758</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.157332</td>
      <td>0.024264</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>med</td>
      <td>1</td>
      <td>0.016309</td>
      <td>0.775945</td>
      <td>307</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>med</td>
      <td>1</td>
      <td>0.140279</td>
      <td>0.016454</td>
      <td>292</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.047874</td>
      <td>0.343236</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.149337</td>
      <td>0.032590</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.024826</td>
      <td>0.621007</td>
      <td>399</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bray-Curtis</th>
      <td>Bray-Curtis</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.222482</td>
      <td>0.001543</td>
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



![png](output_11_3.png)



```python
#results and sample sizes reported in Figure 4E
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
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>-0.019611</td>
      <td>0.759093</td>
      <td>247</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>0.422694</td>
      <td>0.000690</td>
      <td>61</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>0.018057</td>
      <td>0.801157</td>
      <td>197</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>0.175569</td>
      <td>0.065309</td>
      <td>111</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>med</td>
      <td>0</td>
      <td>-0.096958</td>
      <td>0.212578</td>
      <td>167</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>med</td>
      <td>0</td>
      <td>0.246434</td>
      <td>0.003222</td>
      <td>141</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>0.026919</td>
      <td>0.703729</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>0.154282</td>
      <td>0.114321</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>wlk</td>
      <td>0</td>
      <td>0.027024</td>
      <td>0.702637</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>wlk</td>
      <td>0</td>
      <td>0.091562</td>
      <td>0.350575</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.020059</td>
      <td>0.665819</td>
      <td>466</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.362534</td>
      <td>0.000018</td>
      <td>133</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.026107</td>
      <td>0.605400</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.200795</td>
      <td>0.003892</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>med</td>
      <td>1</td>
      <td>0.016009</td>
      <td>0.779964</td>
      <td>307</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>med</td>
      <td>1</td>
      <td>0.145672</td>
      <td>0.012707</td>
      <td>292</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.045610</td>
      <td>0.366566</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.171174</td>
      <td>0.014128</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.023742</td>
      <td>0.636342</td>
      <td>399</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Weighted_Unifrac</th>
      <td>Weighted_Unifrac</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.257775</td>
      <td>0.000229</td>
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
my_range=range(len(Weighted_Unifrac.index))
# Create a color if the group is "B"
my_color=np.where(Weighted_Unifrac ['healthy(yes1/no0)']==1, 'darkblue', 'darkred')
my_size=np.where(Weighted_Unifrac ['healthy(yes1/no0)']<2, 100, 30)
plt.figure(figsize=[5,10], dpi=200)
plt.rcParams.update({'font.size': 24})
plt.rcParams['axes.facecolor'] = 'white'
#plt.rcParams['axes.facecolor'] = 'white'
plt.hlines(y=my_range, xmin=0, xmax=Weighted_Unifrac['pearsonr'], color=my_color, alpha=0.5)
plt.scatter(Weighted_Unifrac['pearsonr'], my_range, color=my_color, s=my_size, alpha=1)
# Add title and exis names
plt.xlim(-0.3,0.50)
plt.yticks(my_range, Weighted_Unifrac['healthy(yes1/no0)'])
plt.xlabel('Pearsonr')
plt.ylabel('Group')
```




    Text(0, 0.5, 'Group')




    <Figure size 3600x3600 with 0 Axes>



![png](output_13_2.png)



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
      <td>-0.061235</td>
      <td>0.337861</td>
      <td>247</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>0.174788</td>
      <td>0.177886</td>
      <td>61</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>-0.027889</td>
      <td>0.697258</td>
      <td>197</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>0.006612</td>
      <td>0.945090</td>
      <td>111</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>med</td>
      <td>0</td>
      <td>-0.063724</td>
      <td>0.413275</td>
      <td>167</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>med</td>
      <td>0</td>
      <td>0.039408</td>
      <td>0.642678</td>
      <td>141</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>-0.070443</td>
      <td>0.319147</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>0.114870</td>
      <td>0.240986</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>wlk</td>
      <td>0</td>
      <td>-0.041702</td>
      <td>0.555676</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>wlk</td>
      <td>0</td>
      <td>0.076980</td>
      <td>0.432849</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.003057</td>
      <td>0.947530</td>
      <td>466</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.022208</td>
      <td>0.799705</td>
      <td>133</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>-0.016230</td>
      <td>0.748089</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.035814</td>
      <td>0.610189</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>med</td>
      <td>1</td>
      <td>-0.020113</td>
      <td>0.725579</td>
      <td>307</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>med</td>
      <td>1</td>
      <td>0.021613</td>
      <td>0.713032</td>
      <td>292</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.010698</td>
      <td>0.832353</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>-0.012524</td>
      <td>0.858549</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>wlk</td>
      <td>1</td>
      <td>-0.000952</td>
      <td>0.984870</td>
      <td>399</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Shannon</th>
      <td>Shannon</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.060371</td>
      <td>0.395771</td>
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



![png](output_16_2.png)



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
      <td>-0.030432</td>
      <td>0.634099</td>
      <td>247</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>comp_healthy</td>
      <td>0</td>
      <td>0.094906</td>
      <td>0.466889</td>
      <td>61</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>0.024602</td>
      <td>0.731477</td>
      <td>197</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>lsc_quant</td>
      <td>0</td>
      <td>-0.092836</td>
      <td>0.332488</td>
      <td>111</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>med</td>
      <td>0</td>
      <td>-0.061313</td>
      <td>0.431210</td>
      <td>167</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>med</td>
      <td>0</td>
      <td>0.042141</td>
      <td>0.619777</td>
      <td>141</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>-0.031447</td>
      <td>0.656838</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>percieved_health</td>
      <td>0</td>
      <td>0.046770</td>
      <td>0.634016</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>wlk</td>
      <td>0</td>
      <td>-0.036362</td>
      <td>0.607416</td>
      <td>202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>wlk</td>
      <td>0</td>
      <td>0.082664</td>
      <td>0.399546</td>
      <td>106</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.043972</td>
      <td>0.343567</td>
      <td>466</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>comp_healthy</td>
      <td>1</td>
      <td>0.062893</td>
      <td>0.472026</td>
      <td>133</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.015536</td>
      <td>0.758522</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>lsc_quant</td>
      <td>1</td>
      <td>0.094779</td>
      <td>0.176446</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>med</td>
      <td>1</td>
      <td>0.022721</td>
      <td>0.691708</td>
      <td>307</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>med</td>
      <td>1</td>
      <td>0.054429</td>
      <td>0.354036</td>
      <td>292</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.045617</td>
      <td>0.366490</td>
      <td>394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>percieved_health</td>
      <td>1</td>
      <td>0.034815</td>
      <td>0.620190</td>
      <td>205</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.052730</td>
      <td>0.293393</td>
      <td>399</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Observed</th>
      <td>Observed</td>
      <td>wlk</td>
      <td>1</td>
      <td>0.070668</td>
      <td>0.320040</td>
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



![png](output_18_2.png)



```python
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



![png](output_20_1.png)



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


![png](output_21_0.png)



```python

```
