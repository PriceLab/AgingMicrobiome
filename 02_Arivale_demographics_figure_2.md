```python
from statsmodels.stats import multitest as multi
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
%matplotlib inline
warnings.filterwarnings("ignore")
```


```python
#Import data for Vendor A
genotek=pd.read_csv('genotek_complete.csv')
#set index
genotek.set_index('public_client_id',inplace=True)
#generate new variables sqrt min
genotek['min_bray_sqrt']=np.sqrt(genotek['min_bray'])
genotek['min_bray_g_sqrt']=np.sqrt(genotek['min_bray_genus'])
```


```python
print(genotek['age'].min())
print(genotek['age'].max())
print(genotek['age'].mean())
print(genotek['age'].std())
print(genotek.groupby(by='sex').size())
```

    21
    82
    47.45844820795589
    12.054126130560013
    sex
    F    1588
    M     951
    dtype: int64



```python
#correlation between age and Bray-Curtis Vendor A Fig. 2B
print("Vendor A")
print('ASV level Bray-Curtis',scipy.stats.spearmanr(genotek['min_bray'],genotek['age']))
print('Genus level Bray-Curtis',scipy.stats.spearmanr(genotek['min_bray_genus'],genotek['age']))
```

    Vendor A
    ASV level Bray-Curtis SpearmanrResult(correlation=0.1572768840256516, pvalue=1.577842568588764e-15)
    Genus level Bray-Curtis SpearmanrResult(correlation=0.18208894446581123, pvalue=2.2855154835446022e-20)



```python
# Import data from Vendor B
second_genome=pd.read_csv('second_genome_complete.csv')
#set index
second_genome.set_index('public_client_id',inplace=True)
#generate new variables sqrt min
second_genome['min_bray_sqrt']=np.sqrt(second_genome['min_bray'])
second_genome['min_bray_g_sqrt']=np.sqrt(second_genome['min_bray_genus'])
```


```python
print(second_genome['age'].min())
print(second_genome['age'].max())
print(second_genome['age'].mean())
print(second_genome['age'].std())
print(second_genome.groupby(by='sex').size())
```

    18
    87
    49.37881508078995
    12.52067975055304
    sex
    F    590
    M    524
    dtype: int64



```python
#correlation between age and Bray-Curtis Vendor A Fig. 2B
print("Vendor B")
print('ASV level Bray-Curtis',scipy.stats.spearmanr(second_genome['min_bray'],second_genome['age']))
print('Genus level Bray-Curtis',scipy.stats.spearmanr(second_genome['min_bray_genus'],second_genome['age']))
```

    Vendor B
    ASV level Bray-Curtis SpearmanrResult(correlation=0.18196533005049664, pvalue=9.49260731657714e-10)
    Genus level Bray-Curtis SpearmanrResult(correlation=0.18897948927799696, pvalue=2.046764790654971e-10)



```python
#Generate Figure 2A ASV
sns.set(font_scale=2.00,context='poster',font='Arial',style='white')
plt.figure(figsize=[15,22.5], dpi=100)
#combine data
frames=[genotek,second_genome]
combined=pd.concat(frames,axis=0,sort=True)
#adjust for vendor
results=smf.ols('min_bray~vendor',data=combined).fit()
combined['corr']=results.resid+combined['min_bray'].mean()
#generate figure
ax=sns.boxplot(y=(combined['corr']),x=combined['age_1'],notch=False,fliersize=10.0, order=['<30','30-39','40-49','50-59','60-69','70-79','80+'],palette='Greys',showfliers=True,linewidth=4, meanline=False,showmeans=False)
ax.set_xlabel('Age')
ax.set_ylabel('Uniqueness (Bray-Curtis)')
plt.show()
```

    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.
    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.



![png](output_7_1.png)



```python
#Generate Figure 2A Genus
sns.set(font_scale=2.00,context='poster',font='Arial',style='white')
plt.figure(figsize=[15,22.5], dpi=100)
#adjust for vendor
results=smf.ols('min_bray_genus~vendor',data=combined).fit()
combined['corr_genus']=results.resid+combined['min_bray_genus'].mean()
#generate boxplot
ax=sns.boxplot(y=(combined['corr_genus']),x=combined['age_1'],notch=False,fliersize=10.0, order=['<30','30-39','40-49','50-59','60-69','70-79','80+'],palette='Blues',showfliers=True,linewidth=4, meanline=False,showmeans=False)
ax.set_xlabel('Age')
ax.set_ylabel('Uniqueness (Bray-Curtis)')
plt.show()
```


![png](output_8_0.png)



```python
#Kernel density plots Fig.2
sns.set(font_scale=2.0,context='poster',font='Arial',style='white')
plt.figure(figsize=[15,10], dpi=100)
sns.distplot([combined['min_bray_genus']], hist = False, kde = True,color='lightblue',
                 kde_kws = {'shade': True, 'linewidth': 2})
sns.distplot(combined['min_bray'], hist = False, kde = True,color='black',
                 kde_kws = {'shade': True, 'linewidth': 2})
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f37a096d748>




![png](output_9_1.png)



```python
weight=pd.read_csv('weight_arivale.csv')
weight=weight.set_index('public_client_id')
combined['BMI']=weight['BMI_CALC']
```


```python
sex=[]
for x in combined['sex']:
    if x=='F':
        sex.append(1)
    elif x=='M':
        sex.append(0)
    else:
        sex.append(9)
combined['sex_num']=sex
```


```python
#Statistical analysis figure 1A
combined['min_bray_sqrt_genus']=np.sqrt(combined['min_bray_genus'])
results = smf.ols('min_bray_sqrt_genus ~ vendor+C(age_1, Treatment("<30"))+sex+BMI+Shannon_genus', data=combined).fit()
results.pvalues
```




    Intercept                              6.349282e-227
    C(age_1, Treatment("<30"))[T.30-39]     9.087249e-01
    C(age_1, Treatment("<30"))[T.40-49]     7.150687e-02
    C(age_1, Treatment("<30"))[T.50-59]     3.570698e-03
    C(age_1, Treatment("<30"))[T.60-69]     4.331771e-07
    C(age_1, Treatment("<30"))[T.70-79]     8.157767e-09
    C(age_1, Treatment("<30"))[T.80+]       7.903359e-03
    sex[T.M]                                6.851984e-05
    vendor                                  2.306016e-31
    BMI                                     6.676141e-04
    Shannon_genus                          7.394479e-119
    dtype: float64




```python
combined['min_bray_sqrt']=np.sqrt(combined['min_bray'])
results = smf.ols('min_bray_sqrt ~ vendor+C(age_1, Treatment("<30"))+sex+BMI+Shannon', data=combined).fit()
results.pvalues
```




    Intercept                               0.000000e+00
    C(age_1, Treatment("<30"))[T.30-39]     4.018527e-01
    C(age_1, Treatment("<30"))[T.40-49]     5.719993e-01
    C(age_1, Treatment("<30"))[T.50-59]     3.517325e-02
    C(age_1, Treatment("<30"))[T.60-69]     1.877557e-05
    C(age_1, Treatment("<30"))[T.70-79]     1.470618e-09
    C(age_1, Treatment("<30"))[T.80+]       1.123212e-02
    sex[T.M]                                1.277220e-11
    vendor                                 5.504748e-106
    BMI                                     4.594970e-01
    Shannon                                 2.320058e-04
    dtype: float64




```python

```


```python
#save df for metabolomics analysis
combined.to_csv('results_merged.csv')
```


```python
#import demographic data on Arivale participants
whole_df=pd.read_csv('demographics_df.csv')
whole_df.set_index('public_client_id',inplace=True)
```


```python
#missing responses are coded as 9 in the df, these will be converted to nan in the next cell
whole_df.groupby(by='Antibiotics').size()
```




    Antibiotics
    0.0    2193
    1.0     360
    9.0      15
    dtype: int64




```python
whole_df['sex']=combined['sex_num']
```


```python
for x in ['Prescription Med','Sweets','Fruits','Vegetables','Grains','Alcohol','Tobacco','Sleep','Diarrhea','Antibiotics']:
    whole_df[x].fillna(9,inplace=True)
```


```python
#test associations between clinical, demographic, lifestyle factors and bray-curtis uniqueness
p=[]
analyte=[]
test_value=[]
r_squared=[]
coef=[]
missing=[]
variable=[]
results_ols=pd.DataFrame()
#exclude outliers
whole_df['min_bray_sqrt_genus']=np.sqrt(combined['min_bray_genus'])
no_outliers=whole_df[whole_df['min_bray_sqrt_genus']<(whole_df['min_bray_sqrt_genus'].mean()+3*whole_df['min_bray_sqrt_genus'].std())]
no_outliers=no_outliers[no_outliers['min_bray_sqrt_genus']>(whole_df['min_bray_sqrt_genus'].mean()-3*whole_df['min_bray_sqrt_genus'].std())]
print('size after outlier removal',no_outliers.shape)
demographics=['age','sex','Race(ref.white)','BMI']
for x in demographics:
    variable.append('demographics')
    no_outliers['response']=no_outliers[x]
    missing.append(no_outliers['response'].isnull().sum())
    results = smf.ols('min_bray_sqrt_genus ~ vendor+response', data=no_outliers).fit()
    just_vendor = smf.ols('min_bray_sqrt_genus ~ vendor', data=no_outliers[no_outliers['response'].isnull()==False]).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[2]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[2]
    coef.append(param)
    r_squared.append((results.rsquared-just_vendor.rsquared)*100)
    test_value.append(param)
clinical_tests=['Globulin','Sodium','Homocysteine','GGT','ALAT','ALP','HDL','CRP','LDL','Triglycerides','n6/n3','Insulin','Glucose','HOMA-IR','HbA1c','Creatinine','Vitamin D']
diet_list=['Fruits','Vegetables','Grains','Alcohol','Tobacco','Sleep','Diarrhea','Antibiotics']
for x in diet_list:
    variable.append('diet/lifestyle')
    no_outliers['response']=no_outliers[x]
    df=no_outliers[no_outliers['response']<9]
    missing.append(len(no_outliers['response'])-len(df))
    results = smf.ols('min_bray_sqrt_genus ~ vendor+response', data=df).fit()
    just_vendor = smf.ols('min_bray_sqrt_genus ~ vendor', data=df).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[2]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[2]
    coef.append(param)
    r_squared.append((results.rsquared-just_vendor.rsquared)*100)
    test_value.append(param)
for x in ['Prescription Med','Sweets']:
    variable.append('diet/lifestyle')
    no_outliers['response']=no_outliers[x]
    df=no_outliers[no_outliers['response']<9]
    missing.append(len(no_outliers['response'])-len(df))
    results = smf.ols('min_bray_sqrt_genus ~ response', data=df).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[1]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[1]
    coef.append(param)
    r_squared.append(results.rsquared*100)
    test_value.append(param)
for x in clinical_tests:
    variable.append('clinical_labs')
    no_outliers['response']=no_outliers[x]
    missing.append(no_outliers['response'].isnull().sum())
    results = smf.ols('min_bray_sqrt_genus ~ vendor+response', data=no_outliers).fit()
    just_vendor = smf.ols('min_bray_sqrt_genus ~ vendor', data=no_outliers[no_outliers['response'].isnull()==False]).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[2]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[2]
    coef.append(param)
    r_squared.append((results.rsquared-just_vendor.rsquared)*100)
    test_value.append(param)
results_ols=pd.DataFrame()
results_ols['analyte']=analyte
results_ols['pvalue']=p
results_ols.set_index('analyte')
results_ols['r_squared']=r_squared
results_ols['coefficient']=coef
results_ols['missing']=missing
results_ols['variable']=variable
#multiple hypothesis correction
results_ols['corr_pval']=multi.multipletests(results_ols['pvalue'], alpha=0.05, method='bonferroni', is_sorted=False,returnsorted=False)[1]
#confirm no. of metaoblites tested
results_ols.sort_values(by='r_squared',ascending=False,inplace=True)
```

    size after outlier removal (3618, 70)



```python
direction=[]
for x in results_ols['coefficient']:
    if x<0:
        direction.append(-1)
    else:
        direction.append(1)
results_ols['direction']=direction
results_ols['r_squared_dir']=results_ols['r_squared']*results_ols['direction']
results_ols.sort_values(by='corr_pval')
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
      <th>analyte</th>
      <th>pvalue</th>
      <th>r_squared</th>
      <th>coefficient</th>
      <th>missing</th>
      <th>variable</th>
      <th>corr_pval</th>
      <th>direction</th>
      <th>r_squared_dir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>age</td>
      <td>2.357540e-28</td>
      <td>3.046453</td>
      <td>0.000869</td>
      <td>0</td>
      <td>demographics</td>
      <td>7.308375e-27</td>
      <td>1</td>
      <td>3.046453</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Vitamin D</td>
      <td>2.701616e-08</td>
      <td>0.802681</td>
      <td>0.005422</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>8.375009e-07</td>
      <td>1</td>
      <td>0.802681</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Alcohol</td>
      <td>8.987587e-07</td>
      <td>0.658101</td>
      <td>-0.007863</td>
      <td>269</td>
      <td>diet/lifestyle</td>
      <td>2.786152e-05</td>
      <td>-1</td>
      <td>-0.658101</td>
    </tr>
    <tr>
      <th>20</th>
      <td>HDL</td>
      <td>2.030378e-05</td>
      <td>0.472444</td>
      <td>0.004181</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>6.294172e-04</td>
      <td>1</td>
      <td>0.472444</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sex</td>
      <td>1.324238e-04</td>
      <td>0.370316</td>
      <td>0.007561</td>
      <td>0</td>
      <td>demographics</td>
      <td>4.105137e-03</td>
      <td>1</td>
      <td>0.370316</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Prescription Med</td>
      <td>4.076586e-04</td>
      <td>1.404503</td>
      <td>0.014805</td>
      <td>2732</td>
      <td>diet/lifestyle</td>
      <td>1.263742e-02</td>
      <td>1</td>
      <td>1.404503</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Triglycerides</td>
      <td>7.894477e-04</td>
      <td>0.293367</td>
      <td>-0.003296</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>2.447288e-02</td>
      <td>-1</td>
      <td>-0.293367</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Diarrhea</td>
      <td>1.739344e-03</td>
      <td>0.263366</td>
      <td>-0.004061</td>
      <td>204</td>
      <td>diet/lifestyle</td>
      <td>5.391967e-02</td>
      <td>-1</td>
      <td>-0.263366</td>
    </tr>
    <tr>
      <th>28</th>
      <td>HbA1c</td>
      <td>8.454809e-03</td>
      <td>0.180688</td>
      <td>0.002618</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>2.620991e-01</td>
      <td>1</td>
      <td>0.180688</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Grains</td>
      <td>1.234112e-02</td>
      <td>0.169699</td>
      <td>-0.003138</td>
      <td>237</td>
      <td>diet/lifestyle</td>
      <td>3.825746e-01</td>
      <td>-1</td>
      <td>-0.169699</td>
    </tr>
    <tr>
      <th>24</th>
      <td>n6/n3</td>
      <td>1.567014e-02</td>
      <td>0.152180</td>
      <td>-0.002428</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>4.857742e-01</td>
      <td>-1</td>
      <td>-0.152180</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BMI</td>
      <td>1.751712e-02</td>
      <td>0.150074</td>
      <td>-0.000390</td>
      <td>170</td>
      <td>demographics</td>
      <td>5.430308e-01</td>
      <td>-1</td>
      <td>-0.150074</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Antibiotics</td>
      <td>2.031116e-02</td>
      <td>0.195616</td>
      <td>0.007551</td>
      <td>1092</td>
      <td>diet/lifestyle</td>
      <td>6.296458e-01</td>
      <td>1</td>
      <td>0.195616</td>
    </tr>
    <tr>
      <th>22</th>
      <td>LDL</td>
      <td>2.534290e-02</td>
      <td>0.130326</td>
      <td>-0.002186</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>7.856300e-01</td>
      <td>-1</td>
      <td>-0.130326</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GGT</td>
      <td>2.980023e-02</td>
      <td>0.123045</td>
      <td>-0.002116</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>9.238072e-01</td>
      <td>-1</td>
      <td>-0.123045</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Globulin</td>
      <td>3.009083e-02</td>
      <td>0.122611</td>
      <td>-0.002121</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>9.328157e-01</td>
      <td>-1</td>
      <td>-0.122611</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tobacco</td>
      <td>3.130034e-02</td>
      <td>0.129973</td>
      <td>-0.010575</td>
      <td>359</td>
      <td>diet/lifestyle</td>
      <td>9.703106e-01</td>
      <td>-1</td>
      <td>-0.129973</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Creatinine</td>
      <td>8.225649e-01</td>
      <td>0.001312</td>
      <td>-0.000219</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.001312</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Homocysteine</td>
      <td>5.360479e-02</td>
      <td>0.097103</td>
      <td>0.001899</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>1</td>
      <td>0.097103</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ALAT</td>
      <td>6.189030e-02</td>
      <td>0.090877</td>
      <td>-0.001825</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.090877</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Sweets</td>
      <td>4.698986e-01</td>
      <td>0.059903</td>
      <td>-0.000889</td>
      <td>2744</td>
      <td>diet/lifestyle</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.059903</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ALP</td>
      <td>1.816952e-01</td>
      <td>0.046511</td>
      <td>0.001319</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>1</td>
      <td>0.046511</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Insulin</td>
      <td>2.018528e-01</td>
      <td>0.042476</td>
      <td>-0.001248</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.042476</td>
    </tr>
    <tr>
      <th>21</th>
      <td>CRP</td>
      <td>2.249516e-01</td>
      <td>0.038398</td>
      <td>-0.001186</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.038398</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Glucose</td>
      <td>2.873025e-01</td>
      <td>0.029526</td>
      <td>0.001049</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>1</td>
      <td>0.029526</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fruits</td>
      <td>3.120022e-01</td>
      <td>0.027416</td>
      <td>0.001187</td>
      <td>198</td>
      <td>diet/lifestyle</td>
      <td>1.000000e+00</td>
      <td>1</td>
      <td>0.027416</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Sleep</td>
      <td>4.018094e-01</td>
      <td>0.027333</td>
      <td>0.002036</td>
      <td>1257</td>
      <td>diet/lifestyle</td>
      <td>1.000000e+00</td>
      <td>1</td>
      <td>0.027333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Race(ref.white)</td>
      <td>4.696943e-01</td>
      <td>0.013621</td>
      <td>0.001749</td>
      <td>88</td>
      <td>demographics</td>
      <td>1.000000e+00</td>
      <td>1</td>
      <td>0.013621</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Sodium</td>
      <td>4.710159e-01</td>
      <td>0.013551</td>
      <td>0.000708</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>1</td>
      <td>0.013551</td>
    </tr>
    <tr>
      <th>27</th>
      <td>HOMA-IR</td>
      <td>5.391534e-01</td>
      <td>0.009835</td>
      <td>-0.000599</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.009835</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Vegetables</td>
      <td>8.625670e-01</td>
      <td>0.000804</td>
      <td>-0.000197</td>
      <td>198</td>
      <td>diet/lifestyle</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.000804</td>
    </tr>
  </tbody>
</table>
</div>




```python
#significant results after multiple hypothesis correction (0.05 two-sided)
results_ols[results_ols['corr_pval']<0.05]
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
      <th>analyte</th>
      <th>pvalue</th>
      <th>r_squared</th>
      <th>coefficient</th>
      <th>missing</th>
      <th>variable</th>
      <th>corr_pval</th>
      <th>direction</th>
      <th>r_squared_dir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>age</td>
      <td>2.357540e-28</td>
      <td>3.046453</td>
      <td>0.000869</td>
      <td>0</td>
      <td>demographics</td>
      <td>7.308375e-27</td>
      <td>1</td>
      <td>3.046453</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Prescription Med</td>
      <td>4.076586e-04</td>
      <td>1.404503</td>
      <td>0.014805</td>
      <td>2732</td>
      <td>diet/lifestyle</td>
      <td>1.263742e-02</td>
      <td>1</td>
      <td>1.404503</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Vitamin D</td>
      <td>2.701616e-08</td>
      <td>0.802681</td>
      <td>0.005422</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>8.375009e-07</td>
      <td>1</td>
      <td>0.802681</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Alcohol</td>
      <td>8.987587e-07</td>
      <td>0.658101</td>
      <td>-0.007863</td>
      <td>269</td>
      <td>diet/lifestyle</td>
      <td>2.786152e-05</td>
      <td>-1</td>
      <td>-0.658101</td>
    </tr>
    <tr>
      <th>20</th>
      <td>HDL</td>
      <td>2.030378e-05</td>
      <td>0.472444</td>
      <td>0.004181</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>6.294172e-04</td>
      <td>1</td>
      <td>0.472444</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sex</td>
      <td>1.324238e-04</td>
      <td>0.370316</td>
      <td>0.007561</td>
      <td>0</td>
      <td>demographics</td>
      <td>4.105137e-03</td>
      <td>1</td>
      <td>0.370316</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Triglycerides</td>
      <td>7.894477e-04</td>
      <td>0.293367</td>
      <td>-0.003296</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>2.447288e-02</td>
      <td>-1</td>
      <td>-0.293367</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Generate figure 2d
sns.set(font_scale=1.95,context='poster',font='Arial',style='white')
sns.set_color_codes("dark")
plt.figure(figsize=[35,30], dpi=100)
plt.ylim(-2.5,4.0)
clr=[]
for x in results_ols['variable']:
    if x=='demographics':
        clr.append('lightgreen')
    elif x=='clinical_labs':
        clr.append('lightblue')
    elif x=='diet/lifestyle':
        clr.append('gold')
results_ols['clr']=clr
ax=sns.barplot(x='analyte', y='r_squared_dir',data=results_ols,palette=results_ols['clr'].tolist(),label="Total", edgecolor='k')
for item in ax.get_xticklabels():
    item.set_rotation(85)
ax.set(ylabel="Percent Variance Explained",
       xlabel="")
sns.despine(left=True, bottom=True)
```

    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.
    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.



![png](output_23_1.png)



```python
#run linear regression assessing relationship of each variable with uniqueness adjusting for age
demographics=['sex','Race(ref.white)','BMI']
p=[]
analyte=[]
test_value=[]
results=pd.DataFrame()
lower=[]
upper=[]
for x in demographics:
    variable.append('demographics')
    no_outliers['response']=no_outliers[x]
    no_outliers['response']=(no_outliers['response']-no_outliers['response'].mean())/no_outliers['response'].std()
    results = smf.ols('min_bray_sqrt_genus ~  vendor+age+response', data=no_outliers).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[3]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[3]
    test_value.append(param)
for x in clinical_tests:
    variable.append('clinical_labs')
    no_outliers['response']=no_outliers[x]
    no_outliers['response']=(no_outliers['response']-no_outliers['response'].mean())/no_outliers['response'].std()
    missing.append(no_outliers['response'].isnull().sum())
    results = smf.ols('min_bray_sqrt_genus ~  vendor+age+response', data=no_outliers).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[3]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[3]
    test_value.append(param)
for x in diet_list:
    variable.append('diet/lifestyle')
    no_outliers['response']=no_outliers[x]
    df=no_outliers[no_outliers['response']<9]
    df['response']=(df['response']-df['response'].mean())/df['response'].std()
    results = smf.ols('min_bray_sqrt_genus ~  vendor+Age+response', data=df).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[3]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[3]
    test_value.append(param)
for x in ['Prescription Med','Sweets']:
    variable.append('diet/lifestyle')
    no_outliers['response']=no_outliers[x]
    missing.append(len(no_outliers['response'][no_outliers['response']<9]))
    df=no_outliers[no_outliers['response']<9]
    df['response']=(df['response']-df['response'].mean())/df['response'].std()
    results = smf.ols('min_bray_sqrt_genus ~  Age+response', data=df).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[2]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[2]
    test_value.append(param)
results_coef=pd.DataFrame()
lower=[ x[0] for x in  lower]
upper=[ x[0] for x in  upper]
results_coef['analyte']=analyte
results_coef['pvalue']=p
results_coef['Beta_coeff']=test_value
results_coef.set_index('analyte')
results_coef['corr_pval']=multi.multipletests(results_coef['pvalue'], alpha=0.05, method='bonferroni', is_sorted=False,returnsorted=False)[1]
results_coef.sort_values(by='pvalue',inplace=True)
results_coef.sort_values(by='analyte',ascending=True,inplace=True)
results_coef=results_coef.set_index('analyte')
results_age_adj=results_coef
results_age_adj['adjusted']=1
```


```python
#run linear regression assessing relationship of each variable with ASV-level uniqueness adjusting for age
demographics=['sex','Race(ref.white)','BMI']
p=[]
analyte=[]
test_value=[]
results_age_ASV=pd.DataFrame()
for x in demographics:
    variable.append('demographics')
    no_outliers['response']=no_outliers[x]
    #scale/standardize
    no_outliers['response']=(no_outliers['response']-no_outliers['response'].mean())/no_outliers['response'].std()
    results = smf.ols('min_bray_sqrt ~  vendor+age+response', data=no_outliers).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[3]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[3]
    test_value.append(param)
for x in clinical_tests:
    variable.append('clinical_labs')
    no_outliers['response']=no_outliers[x]
    no_outliers['response']=(no_outliers['response']-no_outliers['response'].mean())/no_outliers['response'].std()
    results = smf.ols('min_bray_sqrt ~  vendor+age+response', data=no_outliers).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[3]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[3]
    test_value.append(param)
for x in diet_list:
    variable.append('diet/lifestyle')
    no_outliers['response']=no_outliers[x]
    df=no_outliers[no_outliers['response']<9]
    df['response']=(df['response']-df['response'].mean())/df['response'].std()
    results = smf.ols('min_bray_sqrt ~  vendor+Age+response', data=df).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[3]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[3]
    test_value.append(param)
for x in ['Prescription Med','Sweets']:
    variable.append('diet/lifestyle')
    no_outliers['response']=no_outliers[x]
    df=no_outliers[no_outliers['response']<9]
    df['response']=(df['response']-df['response'].mean())/df['response'].std()
    results = smf.ols('min_bray_sqrt ~  Age+response', data=df).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[2]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[2]
    test_value.append(param)
results_age_ASV['analyte']=analyte
results_age_ASV['pvalue']=p
results_age_ASV['Beta_coeff']=test_value
results_age_ASV.set_index('analyte')
results_age_ASV['corr_pval']=multi.multipletests(results_age_ASV['pvalue'], alpha=0.05, method='bonferroni', is_sorted=False,returnsorted=False)[1]
results_age_ASV.sort_values(by='analyte',ascending=True,inplace=True)
results_age_ASV['adjusted']=1
results_age_ASV
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
      <th>analyte</th>
      <th>pvalue</th>
      <th>Beta_coeff</th>
      <th>corr_pval</th>
      <th>adjusted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>ALAT</td>
      <td>3.108881e-01</td>
      <td>-0.000684</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ALP</td>
      <td>3.947624e-01</td>
      <td>-0.000582</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Alcohol</td>
      <td>9.378565e-02</td>
      <td>-0.001167</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Antibiotics</td>
      <td>5.016965e-05</td>
      <td>0.003300</td>
      <td>1.505089e-03</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BMI</td>
      <td>1.416517e-02</td>
      <td>-0.001678</td>
      <td>4.249551e-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CRP</td>
      <td>7.447765e-02</td>
      <td>-0.001205</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Creatinine</td>
      <td>4.216653e-04</td>
      <td>-0.002378</td>
      <td>1.264996e-02</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Diarrhea</td>
      <td>4.187928e-02</td>
      <td>0.001398</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Fruits</td>
      <td>6.198069e-01</td>
      <td>0.000340</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GGT</td>
      <td>6.573135e-02</td>
      <td>-0.001248</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Globulin</td>
      <td>1.261844e-02</td>
      <td>-0.001691</td>
      <td>3.785532e-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Glucose</td>
      <td>7.700407e-01</td>
      <td>0.000200</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Grains</td>
      <td>2.420615e-03</td>
      <td>-0.002087</td>
      <td>7.261846e-02</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>HDL</td>
      <td>1.806700e-05</td>
      <td>0.002932</td>
      <td>5.420100e-04</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>HOMA-IR</td>
      <td>6.052503e-01</td>
      <td>-0.000349</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>HbA1c</td>
      <td>5.991392e-01</td>
      <td>0.000364</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Homocysteine</td>
      <td>5.301023e-03</td>
      <td>-0.001931</td>
      <td>1.590307e-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Insulin</td>
      <td>2.348506e-01</td>
      <td>-0.000802</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>LDL</td>
      <td>2.240127e-02</td>
      <td>-0.001548</td>
      <td>6.720381e-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Prescription Med</td>
      <td>2.855336e-02</td>
      <td>0.002816</td>
      <td>8.566009e-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Race(ref.white)</td>
      <td>2.157719e-01</td>
      <td>-0.000841</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Sleep</td>
      <td>1.294242e-01</td>
      <td>0.001276</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sodium</td>
      <td>2.067305e-02</td>
      <td>-0.001573</td>
      <td>6.201915e-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Sweets</td>
      <td>3.923745e-01</td>
      <td>0.001076</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Tobacco</td>
      <td>5.115015e-02</td>
      <td>-0.001362</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Triglycerides</td>
      <td>1.945830e-03</td>
      <td>-0.002095</td>
      <td>5.837490e-02</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Vegetables</td>
      <td>8.262994e-01</td>
      <td>0.000150</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Vitamin D</td>
      <td>7.846881e-07</td>
      <td>0.003380</td>
      <td>2.354064e-05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>n6/n3</td>
      <td>4.194554e-02</td>
      <td>-0.001438</td>
      <td>1.000000e+00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>sex</td>
      <td>3.005534e-12</td>
      <td>0.004647</td>
      <td>9.016602e-11</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
results_age_ASV.set_index('analyte',inplace=True)
results_age_ASV[results_age_ASV['corr_pval']<0.05]
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
      <th>pvalue</th>
      <th>Beta_coeff</th>
      <th>corr_pval</th>
      <th>adjusted</th>
    </tr>
    <tr>
      <th>analyte</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Antibiotics</th>
      <td>5.016965e-05</td>
      <td>0.003300</td>
      <td>1.505089e-03</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Creatinine</th>
      <td>4.216653e-04</td>
      <td>-0.002378</td>
      <td>1.264996e-02</td>
      <td>1</td>
    </tr>
    <tr>
      <th>HDL</th>
      <td>1.806700e-05</td>
      <td>0.002932</td>
      <td>5.420100e-04</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Vitamin D</th>
      <td>7.846881e-07</td>
      <td>0.003380</td>
      <td>2.354064e-05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>3.005534e-12</td>
      <td>0.004647</td>
      <td>9.016602e-11</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Generate supplementary table
results_final=results_ols[results_ols.columns[0:7]]
results_final.set_index('analyte',inplace=True)
results_final['age_adjusted_coeff']=results_age_adj['Beta_coeff']
results_final['age_adjusted_corr_pvalue']=results_age_adj['corr_pval']
results_final['age_adjusted_coeff_ASV_level']=results_age_ASV['Beta_coeff']
results_final['age_adjusted_corr_pvalue_ASV_level']=results_age_ASV['corr_pval']
```


```python
results_final.head()
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
      <th>pvalue</th>
      <th>r_squared</th>
      <th>coefficient</th>
      <th>missing</th>
      <th>variable</th>
      <th>corr_pval</th>
      <th>age_adjusted_coeff</th>
      <th>age_adjusted_corr_pvalue</th>
      <th>age_adjusted_coeff_ASV_level</th>
      <th>age_adjusted_corr_pvalue_ASV_level</th>
    </tr>
    <tr>
      <th>analyte</th>
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
      <th>age</th>
      <td>2.357540e-28</td>
      <td>3.046453</td>
      <td>0.000869</td>
      <td>0</td>
      <td>demographics</td>
      <td>7.308375e-27</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Prescription Med</th>
      <td>4.076586e-04</td>
      <td>1.404503</td>
      <td>0.014805</td>
      <td>2732</td>
      <td>diet/lifestyle</td>
      <td>1.263742e-02</td>
      <td>0.004069</td>
      <td>1.000000</td>
      <td>0.002816</td>
      <td>0.856601</td>
    </tr>
    <tr>
      <th>Vitamin D</th>
      <td>2.701616e-08</td>
      <td>0.802681</td>
      <td>0.005422</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>8.375009e-07</td>
      <td>0.003676</td>
      <td>0.004944</td>
      <td>0.003380</td>
      <td>0.000024</td>
    </tr>
    <tr>
      <th>Alcohol</th>
      <td>8.987587e-07</td>
      <td>0.658101</td>
      <td>-0.007863</td>
      <td>269</td>
      <td>diet/lifestyle</td>
      <td>2.786152e-05</td>
      <td>-0.003640</td>
      <td>0.007137</td>
      <td>-0.001167</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>HDL</th>
      <td>2.030378e-05</td>
      <td>0.472444</td>
      <td>0.004181</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>6.294172e-04</td>
      <td>0.002747</td>
      <td>0.145723</td>
      <td>0.002932</td>
      <td>0.000542</td>
    </tr>
  </tbody>
</table>
</div>




```python
#correlation in beta-coefficients for all variables between asv and genus level analysis
results_sig=results_final[results_final['pvalue']<0.05]
scipy.stats.spearmanr(results_sig['age_adjusted_coeff_ASV_level'].dropna(),results_sig['age_adjusted_coeff'].dropna())
```




    SpearmanrResult(correlation=0.711764705882353, pvalue=0.0019840432477699634)




```python
#save results
results_final.to_csv('demographics_results.csv')
```
