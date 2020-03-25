```python
import matplotlib.pyplot as plt
from statsmodels.stats import multitest as multi
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
from sklearn.preprocessing import StandardScaler
%matplotlib inline
```


```python
#Import data for Vendor A
genotek=pd.read_csv('new_silva_min.csv')
#set index
genotek.set_index('public_client_id',inplace=True)
#generate new variables sqrt min
genotek['min_bray_sqrt']=np.sqrt(genotek['min_bray'])
```


```python
#Generate Figure 2A
sns.set(font_scale=2.00,context='poster',font='Arial',style='white')
plt.figure(figsize=[17.5,25], dpi=100)
ax=sns.boxplot(y=np.sqrt(genotek['min_bray']),x=genotek['age_1'],notch=False,fliersize=10.0, order=['<30','30-39','40-49','50-59','60-69','70-79','80+'],palette='Blues',showfliers=True,linewidth=4, meanline=False,showmeans=False)
ax.set_xlabel('Age')
ax.set_ylabel('Uniqueness (Bray-Curtis)')
plt.show()
```

    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.
    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.



![png](output_2_1.png)



```python
# Import data from Vendor B
second_genome=pd.read_csv('new_silva_second_genome_complete.csv')
second_genome.set_index('public_client_id',inplace=True)
print (len(second_genome['min_bray']))
second_genome['min_bray_sqrt']=np.sqrt(second_genome['min_bray'])
#Generate Figure 2A
sns.set(font_scale=2.00,context='poster',font='Arial',style='white')
plt.figure(figsize=[18,25], dpi=100)
meanlineprops = dict(linestyle='-', linewidth=7, color='Red')
#plt.ylim(0.2,.95)
ax=sns.boxplot(y=np.sqrt(second_genome['min_bray']),x=second_genome['age_1'],notch=False,fliersize=12.0, order=['<30','30-39','40-49','50-59','60-69','70-79','80+'],meanprops=meanlineprops,palette='Reds',showfliers=True,linewidth=4, meanline=False,showmeans=False)
ax.set_xlabel('Age')
ax.set_ylabel('Uniqueness (Bray-Curtis)')
plt.show()
```

    1114



![png](output_3_1.png)



```python
#import demographic data on Arivale participants containing uniqueness columns
whole_df=pd.read_csv('demographics_df.csv')
whole_df.set_index('public_client_id',inplace=True)
```


```python
sex=[]
for x in whole_df['sex']:
    if x=='F':
        sex.append(1)
    elif x=='M':
        sex.append(0)
    else:
        sex.append(9)
whole_df['sex_num']=sex
```


```python
#investigate sex-dependent difference in uniqueness aging patterns
no_outliers=whole_df[whole_df['min_bray_sqrt']<(whole_df['min_bray_sqrt'].mean()+3*whole_df['min_bray_sqrt'].std())]
no_outliers=no_outliers[no_outliers['min_bray_sqrt']>(whole_df['min_bray_sqrt'].mean()-3*whole_df['min_bray_sqrt'].std())]
print(no_outliers.shape)
results = smf.ols('min_bray_sqrt ~ vendor+age*sex_num+BMI+Shannon', data=no_outliers).fit()
results.summary()
```

    (3618, 72)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>min_bray_sqrt</td>  <th>  R-squared:         </th> <td>   0.077</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.076</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   46.72</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 25 Mar 2020</td> <th>  Prob (F-statistic):</th> <td>2.95e-55</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:04:27</td>     <th>  Log-Likelihood:    </th> <td>  4873.4</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  3358</td>      <th>  AIC:               </th> <td>  -9733.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  3351</td>      <th>  BIC:               </th> <td>  -9690.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     6</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>   <td>    0.3864</td> <td>    0.012</td> <td>   31.062</td> <td> 0.000</td> <td>    0.362</td> <td>    0.411</td>
</tr>
<tr>
  <th>vendor</th>      <td>   -0.0102</td> <td>    0.002</td> <td>   -4.736</td> <td> 0.000</td> <td>   -0.014</td> <td>   -0.006</td>
</tr>
<tr>
  <th>age</th>         <td>    0.0007</td> <td>    0.000</td> <td>    5.359</td> <td> 0.000</td> <td>    0.000</td> <td>    0.001</td>
</tr>
<tr>
  <th>sex_num</th>     <td>   -0.0149</td> <td>    0.008</td> <td>   -1.864</td> <td> 0.062</td> <td>   -0.031</td> <td>    0.001</td>
</tr>
<tr>
  <th>age:sex_num</th> <td>    0.0004</td> <td>    0.000</td> <td>    2.535</td> <td> 0.011</td> <td> 9.19e-05</td> <td>    0.001</td>
</tr>
<tr>
  <th>BMI</th>         <td>   -0.0004</td> <td>    0.000</td> <td>   -2.256</td> <td> 0.024</td> <td>   -0.001</td> <td>-4.94e-05</td>
</tr>
<tr>
  <th>Shannon</th>     <td>    0.0210</td> <td>    0.002</td> <td>    9.427</td> <td> 0.000</td> <td>    0.017</td> <td>    0.025</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>164.332</td> <th>  Durbin-Watson:     </th> <td>   1.818</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 191.646</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.539</td>  <th>  Prob(JB):          </th> <td>2.42e-42</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.453</td>  <th>  Cond. No.          </th> <td>    861.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
#Estimate beta-coefficients for men and women seperately
Men=no_outliers[no_outliers['sex']=='M']
Women=no_outliers[no_outliers['sex']=='F']
Men['age_standard']=(Men['age']-Men['age'].mean())/Men['age'].std()
Women['age_standard']=(Women['age']-Women['age'].mean())/Women['age'].std()
results = smf.ols('min_bray_sqrt ~ age_standard+Shannon+BMI', data=Men).fit()
print('MEN',results.summary())
results = smf.ols('min_bray_sqrt ~ age_standard+Shannon+BMI', data=Women).fit()
print('WOMEN',results.summary())
```

    MEN                             OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          min_bray_sqrt   R-squared:                       0.030
    Model:                            OLS   Adj. R-squared:                  0.028
    Method:                 Least Squares   F-statistic:                     14.35
    Date:                Wed, 25 Mar 2020   Prob (F-statistic):           3.31e-09
    Time:                        15:04:27   Log-Likelihood:                 1982.0
    No. Observations:                1374   AIC:                            -3956.
    Df Residuals:                    1370   BIC:                            -3935.
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------
    Intercept        0.4592      0.018     25.234      0.000       0.423       0.495
    age_standard     0.0088      0.002      5.580      0.000       0.006       0.012
    Shannon          0.0088      0.003      2.534      0.011       0.002       0.016
    BMI             -0.0002      0.000     -0.710      0.478      -0.001       0.000
    ==============================================================================
    Omnibus:                       70.914   Durbin-Watson:                   1.774
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               82.388
    Skew:                           0.549   Prob(JB):                     1.29e-18
    Kurtosis:                       3.483   Cond. No.                         331.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    WOMEN                             OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          min_bray_sqrt   R-squared:                       0.105
    Model:                            OLS   Adj. R-squared:                  0.104
    Method:                 Least Squares   F-statistic:                     77.65
    Date:                Wed, 25 Mar 2020   Prob (F-statistic):           1.74e-47
    Time:                        15:04:27   Log-Likelihood:                 2891.2
    No. Observations:                1984   AIC:                            -5774.
    Df Residuals:                    1980   BIC:                            -5752.
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------
    Intercept        0.3787      0.015     25.916      0.000       0.350       0.407
    age_standard     0.0129      0.001     10.151      0.000       0.010       0.015
    Shannon          0.0293      0.003     10.104      0.000       0.024       0.035
    BMI             -0.0003      0.000     -1.741      0.082      -0.001    4.28e-05
    ==============================================================================
    Omnibus:                       86.970   Durbin-Watson:                   1.850
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               97.901
    Skew:                           0.520   Prob(JB):                     5.51e-22
    Kurtosis:                       3.318   Cond. No.                         333.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


    /opt/conda/envs/arivale-py3/lib/python3.7/site-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    /opt/conda/envs/arivale-py3/lib/python3.7/site-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """



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
no_outliers=whole_df[whole_df['min_bray_sqrt']<(whole_df['min_bray_sqrt'].mean()+3*whole_df['min_bray_sqrt'].std())]
no_outliers=no_outliers[no_outliers['min_bray_sqrt']>(whole_df['min_bray_sqrt'].mean()-3*whole_df['min_bray_sqrt'].std())]
print('size after outlier removal',no_outliers.shape)

demographics=['Age','Sex','Race(ref.white)','BMI']
for x in demographics:
    variable.append('demographics')
    no_outliers['response']=no_outliers[x]
    missing.append(no_outliers['response'].isnull().sum())
    results = smf.ols('min_bray_sqrt ~ vendor+response', data=no_outliers).fit()
    just_vendor = smf.ols('min_bray_sqrt ~ vendor', data=no_outliers[no_outliers['response'].isnull()==False]).fit()
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
diet_list=['Fruits','Vegetables','Grains','Alcohol','Sweets','Tobacco','Sleep','Prescription Med','Diarrhea','Antibiotics']
for x in diet_list:
    variable.append('diet/lifestyle')
    no_outliers['response']=no_outliers[x]
    missing.append(len(no_outliers['response'][no_outliers['response']<9]))
    df=no_outliers[no_outliers['response']<9]
    results = smf.ols('min_bray_sqrt ~ vendor+response', data=df).fit()
    just_vendor = smf.ols('min_bray_sqrt ~ vendor', data=df).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[2]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[2]
    coef.append(param)
    r_squared.append((results.rsquared-just_vendor.rsquared)*100)
    test_value.append(param)
for x in clinical_tests:
    variable.append('clinical_labs')
    no_outliers['response']=no_outliers[x]
    missing.append(no_outliers['response'].isnull().sum())
    results = smf.ols('min_bray_sqrt ~ vendor+response', data=no_outliers).fit()
    just_vendor = smf.ols('min_bray_sqrt ~ vendor', data=no_outliers[no_outliers['response'].isnull()==False]).fit()
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

    size after outlier removal (3618, 72)



```python
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>5.887447e-33</td>
      <td>3.859015</td>
      <td>0.000933</td>
      <td>0</td>
      <td>demographics</td>
      <td>1.825109e-31</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Prescription Med</td>
      <td>1.748134e-04</td>
      <td>1.532198</td>
      <td>0.015106</td>
      <td>914</td>
      <td>diet/lifestyle</td>
      <td>5.419216e-03</td>
    </tr>
    <tr>
      <th>20</th>
      <td>HDL</td>
      <td>3.031418e-08</td>
      <td>0.865993</td>
      <td>0.005495</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>9.397394e-07</td>
    </tr>
    <tr>
      <th>24</th>
      <td>n6/n3</td>
      <td>3.656834e-08</td>
      <td>0.855769</td>
      <td>-0.005465</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.133619e-06</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Vitamin D</td>
      <td>1.075193e-07</td>
      <td>0.797029</td>
      <td>0.005275</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>3.333097e-06</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Alcohol</td>
      <td>1.360110e-05</td>
      <td>0.560815</td>
      <td>-0.007082</td>
      <td>3349</td>
      <td>diet/lifestyle</td>
      <td>4.216342e-04</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Homocysteine</td>
      <td>1.439458e-04</td>
      <td>0.408715</td>
      <td>0.003771</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>4.462319e-03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BMI</td>
      <td>4.476630e-04</td>
      <td>0.364574</td>
      <td>-0.000597</td>
      <td>260</td>
      <td>demographics</td>
      <td>1.387755e-02</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Triglycerides</td>
      <td>7.707511e-04</td>
      <td>0.320059</td>
      <td>-0.003355</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>2.389328e-02</td>
    </tr>
  </tbody>
</table>
</div>




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
      <td>Age</td>
      <td>5.887447e-33</td>
      <td>3.859015</td>
      <td>0.000933</td>
      <td>0</td>
      <td>demographics</td>
      <td>1.825109e-31</td>
      <td>1</td>
      <td>3.859015</td>
    </tr>
    <tr>
      <th>20</th>
      <td>HDL</td>
      <td>3.031418e-08</td>
      <td>0.865993</td>
      <td>0.005495</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>9.397394e-07</td>
      <td>1</td>
      <td>0.865993</td>
    </tr>
    <tr>
      <th>24</th>
      <td>n6/n3</td>
      <td>3.656834e-08</td>
      <td>0.855769</td>
      <td>-0.005465</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.133619e-06</td>
      <td>-1</td>
      <td>-0.855769</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Vitamin D</td>
      <td>1.075193e-07</td>
      <td>0.797029</td>
      <td>0.005275</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>3.333097e-06</td>
      <td>1</td>
      <td>0.797029</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Alcohol</td>
      <td>1.360110e-05</td>
      <td>0.560815</td>
      <td>-0.007082</td>
      <td>3349</td>
      <td>diet/lifestyle</td>
      <td>4.216342e-04</td>
      <td>-1</td>
      <td>-0.560815</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Homocysteine</td>
      <td>1.439458e-04</td>
      <td>0.408715</td>
      <td>0.003771</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>4.462319e-03</td>
      <td>1</td>
      <td>0.408715</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Prescription Med</td>
      <td>1.748134e-04</td>
      <td>1.532198</td>
      <td>0.015106</td>
      <td>914</td>
      <td>diet/lifestyle</td>
      <td>5.419216e-03</td>
      <td>1</td>
      <td>1.532198</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BMI</td>
      <td>4.476630e-04</td>
      <td>0.364574</td>
      <td>-0.000597</td>
      <td>260</td>
      <td>demographics</td>
      <td>1.387755e-02</td>
      <td>-1</td>
      <td>-0.364574</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Triglycerides</td>
      <td>7.707511e-04</td>
      <td>0.320059</td>
      <td>-0.003355</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>2.389328e-02</td>
      <td>-1</td>
      <td>-0.320059</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Diarrhea</td>
      <td>2.385872e-03</td>
      <td>0.268675</td>
      <td>-0.004008</td>
      <td>3415</td>
      <td>diet/lifestyle</td>
      <td>7.396205e-02</td>
      <td>-1</td>
      <td>-0.268675</td>
    </tr>
    <tr>
      <th>28</th>
      <td>HbA1c</td>
      <td>7.156158e-03</td>
      <td>0.204807</td>
      <td>0.002725</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>2.218409e-01</td>
      <td>1</td>
      <td>0.204807</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Race(ref.white)</td>
      <td>1.700183e-02</td>
      <td>0.160611</td>
      <td>0.005839</td>
      <td>88</td>
      <td>demographics</td>
      <td>5.270568e-01</td>
      <td>1</td>
      <td>0.160611</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sex</td>
      <td>2.355609e-02</td>
      <td>0.141017</td>
      <td>0.004521</td>
      <td>0</td>
      <td>demographics</td>
      <td>7.302387e-01</td>
      <td>1</td>
      <td>0.141017</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Sleep</td>
      <td>9.313900e-01</td>
      <td>0.000316</td>
      <td>0.000214</td>
      <td>2335</td>
      <td>diet/lifestyle</td>
      <td>1.000000e+00</td>
      <td>1</td>
      <td>0.000316</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Sodium</td>
      <td>8.964380e-01</td>
      <td>0.000480</td>
      <td>0.000129</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>1</td>
      <td>0.000480</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Vegetables</td>
      <td>8.461063e-01</td>
      <td>0.001096</td>
      <td>0.000224</td>
      <td>3422</td>
      <td>diet/lifestyle</td>
      <td>1.000000e+00</td>
      <td>1</td>
      <td>0.001096</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Insulin</td>
      <td>4.881959e-01</td>
      <td>0.013618</td>
      <td>-0.000694</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.013618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fruits</td>
      <td>3.416195e-01</td>
      <td>0.026307</td>
      <td>0.001134</td>
      <td>3422</td>
      <td>diet/lifestyle</td>
      <td>1.000000e+00</td>
      <td>1</td>
      <td>0.026307</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Grains</td>
      <td>3.202129e-01</td>
      <td>0.029105</td>
      <td>-0.001263</td>
      <td>3379</td>
      <td>diet/lifestyle</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.029105</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GGT</td>
      <td>2.534556e-01</td>
      <td>0.036957</td>
      <td>-0.001132</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.036957</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ALP</td>
      <td>1.841812e-01</td>
      <td>0.049973</td>
      <td>0.001332</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>1</td>
      <td>0.049973</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ALAT</td>
      <td>4.087102e-02</td>
      <td>0.118443</td>
      <td>-0.002050</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.118443</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Tobacco</td>
      <td>1.177141e-01</td>
      <td>0.074603</td>
      <td>-0.007721</td>
      <td>3264</td>
      <td>diet/lifestyle</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.074603</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Glucose</td>
      <td>1.041622e-01</td>
      <td>0.074816</td>
      <td>0.001632</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>1</td>
      <td>0.074816</td>
    </tr>
    <tr>
      <th>22</th>
      <td>LDL</td>
      <td>4.967172e-02</td>
      <td>0.109133</td>
      <td>-0.001950</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.109133</td>
    </tr>
    <tr>
      <th>21</th>
      <td>CRP</td>
      <td>4.604621e-02</td>
      <td>0.112740</td>
      <td>-0.001979</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.112740</td>
    </tr>
    <tr>
      <th>27</th>
      <td>HOMA-IR</td>
      <td>9.531837e-01</td>
      <td>0.000098</td>
      <td>-0.000058</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.000098</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sweets</td>
      <td>2.716145e-01</td>
      <td>0.134149</td>
      <td>-0.001290</td>
      <td>902</td>
      <td>diet/lifestyle</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.134149</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Antibiotics</td>
      <td>5.402505e-02</td>
      <td>0.147731</td>
      <td>0.006427</td>
      <td>2500</td>
      <td>diet/lifestyle</td>
      <td>1.000000e+00</td>
      <td>1</td>
      <td>0.147731</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Globulin</td>
      <td>1.306099e-01</td>
      <td>0.064751</td>
      <td>-0.001503</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.064751</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Creatinine</td>
      <td>9.890062e-01</td>
      <td>0.000005</td>
      <td>-0.000014</td>
      <td>104</td>
      <td>clinical_labs</td>
      <td>1.000000e+00</td>
      <td>-1</td>
      <td>-0.000005</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set(font_scale=1.95,context='poster',font='Arial',style='white')
sns.set_color_codes("dark")
plt.figure(figsize=[35,30], dpi=100)
plt.ylim(-2.5,4.5)
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



![png](output_11_1.png)



```python
#run linear regression assessing relationship of each variable with uniqueness adjusting for age
demographics=['Race(ref.white)','BMI','Sex']
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
    missing.append(no_outliers['response'].isnull().sum())
    #print name of condition and shape of dataframe
    print (x)
    results = smf.ols('min_bray_sqrt ~  vendor+Age+response', data=no_outliers).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[3]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[3]
    test_value.append(param)
    lower.append(results.conf_int(alpha=0.05/len(results_ols['analyte'][results_ols['corr_pval']<0.05]), cols=None)[3:4][0].tolist())
    upper.append(results.conf_int(alpha=0.05/len(results_ols['analyte'][results_ols['corr_pval']<0.05]), cols=None)[3:4][1].tolist())
for x in clinical_tests:
    variable.append('clinical_labs')
    no_outliers['response']=no_outliers[x]
    no_outliers['response']=(no_outliers['response']-no_outliers['response'].mean())/no_outliers['response'].std()
    missing.append(no_outliers['response'].isnull().sum())
    #print name of condition and shape of dataframe
    print (x)
    results = smf.ols('min_bray_sqrt ~  vendor+Age+response', data=no_outliers).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[3]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[3]
    test_value.append(param)
    lower.append(results.conf_int(alpha=0.05/len(results_ols['analyte']), cols=None)[3:4][0].tolist())
    upper.append(results.conf_int(alpha=0.05/len(results_ols['analyte']), cols=None)[3:4][1].tolist())
for x in diet_list:
    variable.append('diet/lifestyle')
    no_outliers['response']=no_outliers[x]
    missing.append(len(no_outliers['response'][no_outliers['response']<9]))
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
    lower.append(results.conf_int(alpha=0.05/len(results_ols['analyte']), cols=None)[3:4][0].tolist())
    upper.append(results.conf_int(alpha=0.05/len(results_ols['analyte']), cols=None)[3:4][1].tolist())   
results_coef=pd.DataFrame()
lower=[ x[0] for x in  lower]
upper=[ x[0] for x in  upper]
results_coef['analyte']=analyte
results_coef['pvalue']=p
results_coef['Beta_coeff']=test_value
results_coef['lower_limit']=lower
results_coef['upper_limit']=upper
results_coef.set_index('analyte')
results_coef['corr_pval']=multi.multipletests(results_coef['pvalue'], alpha=0.05, method='bonferroni', is_sorted=False,returnsorted=False)[1]
results_coef.sort_values(by='pvalue',inplace=True)
results_coef.sort_values(by='analyte',ascending=True,inplace=True)
results_coef=results_coef.set_index('analyte')
results_pred=results_coef
results_pred['adjusted']=1
results_pred
```

    Race(ref.white)
    BMI
    Sex
    Globulin
    Sodium
    Homocysteine
    GGT
    ALAT
    ALP
    HDL
    CRP
    LDL
    Triglycerides
    n6/n3
    Insulin
    Glucose
    HOMA-IR
    HbA1c
    Creatinine
    Vitamin D


    /opt/conda/envs/arivale-py3/lib/python3.7/site-packages/ipykernel_launcher.py:48: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy





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
      <th>lower_limit</th>
      <th>upper_limit</th>
      <th>corr_pval</th>
      <th>adjusted</th>
    </tr>
    <tr>
      <th>analyte</th>
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
      <th>ALAT</th>
      <td>0.026205</td>
      <td>-0.002166</td>
      <td>-0.005239</td>
      <td>0.000908</td>
      <td>0.786164</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ALP</th>
      <td>0.550247</td>
      <td>-0.000590</td>
      <td>-0.003707</td>
      <td>0.002527</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Alcohol</th>
      <td>0.002860</td>
      <td>-0.002983</td>
      <td>-0.006137</td>
      <td>0.000171</td>
      <td>0.085803</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Antibiotics</th>
      <td>0.103350</td>
      <td>0.001844</td>
      <td>-0.001729</td>
      <td>0.005418</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>0.000101</td>
      <td>-0.003869</td>
      <td>-0.006627</td>
      <td>-0.001112</td>
      <td>0.003022</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CRP</th>
      <td>0.043543</td>
      <td>-0.001966</td>
      <td>-0.005040</td>
      <td>0.001107</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Creatinine</th>
      <td>0.558882</td>
      <td>-0.000571</td>
      <td>-0.003651</td>
      <td>0.002510</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Diarrhea</th>
      <td>0.086906</td>
      <td>-0.001702</td>
      <td>-0.004838</td>
      <td>0.001435</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fruits</th>
      <td>0.861430</td>
      <td>-0.000172</td>
      <td>-0.003291</td>
      <td>0.002946</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>GGT</th>
      <td>0.016431</td>
      <td>-0.002349</td>
      <td>-0.005438</td>
      <td>0.000740</td>
      <td>0.492916</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Globulin</th>
      <td>0.740228</td>
      <td>-0.000325</td>
      <td>-0.003416</td>
      <td>0.002766</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Glucose</th>
      <td>0.656269</td>
      <td>-0.000441</td>
      <td>-0.003567</td>
      <td>0.002685</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Grains</th>
      <td>0.867210</td>
      <td>-0.000166</td>
      <td>-0.003293</td>
      <td>0.002961</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>HDL</th>
      <td>0.000055</td>
      <td>0.003964</td>
      <td>0.000867</td>
      <td>0.007062</td>
      <td>0.001641</td>
      <td>1</td>
    </tr>
    <tr>
      <th>HOMA-IR</th>
      <td>0.389496</td>
      <td>-0.000841</td>
      <td>-0.003923</td>
      <td>0.002242</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>HbA1c</th>
      <td>0.944554</td>
      <td>0.000070</td>
      <td>-0.003092</td>
      <td>0.003231</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Homocysteine</th>
      <td>0.109418</td>
      <td>0.001589</td>
      <td>-0.001543</td>
      <td>0.004722</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Insulin</th>
      <td>0.214460</td>
      <td>-0.001211</td>
      <td>-0.004289</td>
      <td>0.001867</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LDL</th>
      <td>0.001378</td>
      <td>-0.003130</td>
      <td>-0.006216</td>
      <td>-0.000045</td>
      <td>0.041349</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Prescription Med</th>
      <td>0.085453</td>
      <td>0.003414</td>
      <td>-0.002858</td>
      <td>0.009687</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Race(ref.white)</th>
      <td>0.484777</td>
      <td>0.000688</td>
      <td>-0.002043</td>
      <td>0.003418</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0.013793</td>
      <td>0.002369</td>
      <td>-0.000299</td>
      <td>0.005037</td>
      <td>0.413792</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Sleep</th>
      <td>0.713502</td>
      <td>0.000427</td>
      <td>-0.003247</td>
      <td>0.004102</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Sodium</th>
      <td>0.339590</td>
      <td>-0.000934</td>
      <td>-0.004022</td>
      <td>0.002153</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Sweets</th>
      <td>0.982582</td>
      <td>-0.000042</td>
      <td>-0.006141</td>
      <td>0.006057</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Tobacco</th>
      <td>0.250475</td>
      <td>-0.001157</td>
      <td>-0.004336</td>
      <td>0.002021</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Triglycerides</th>
      <td>0.000020</td>
      <td>-0.004164</td>
      <td>-0.007239</td>
      <td>-0.001090</td>
      <td>0.000590</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Vegetables</th>
      <td>0.844503</td>
      <td>-0.000193</td>
      <td>-0.003301</td>
      <td>0.002914</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Vitamin D</th>
      <td>0.001445</td>
      <td>0.003160</td>
      <td>0.000032</td>
      <td>0.006288</td>
      <td>0.043353</td>
      <td>1</td>
    </tr>
    <tr>
      <th>n6/n3</th>
      <td>0.001570</td>
      <td>-0.003148</td>
      <td>-0.006289</td>
      <td>-0.000008</td>
      <td>0.047098</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
results_ols.set_index('analyte',inplace=True)
```


```python
#models adjusted for age
results_pred[results_pred['corr_pval']<0.05]
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
      <th>lower_limit</th>
      <th>upper_limit</th>
      <th>corr_pval</th>
      <th>adjusted</th>
    </tr>
    <tr>
      <th>analyte</th>
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
      <th>BMI</th>
      <td>0.000101</td>
      <td>-0.003869</td>
      <td>-0.006627</td>
      <td>-0.001112</td>
      <td>0.003022</td>
      <td>1</td>
    </tr>
    <tr>
      <th>HDL</th>
      <td>0.000055</td>
      <td>0.003964</td>
      <td>0.000867</td>
      <td>0.007062</td>
      <td>0.001641</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LDL</th>
      <td>0.001378</td>
      <td>-0.003130</td>
      <td>-0.006216</td>
      <td>-0.000045</td>
      <td>0.041349</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Triglycerides</th>
      <td>0.000020</td>
      <td>-0.004164</td>
      <td>-0.007239</td>
      <td>-0.001090</td>
      <td>0.000590</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Vitamin D</th>
      <td>0.001445</td>
      <td>0.003160</td>
      <td>0.000032</td>
      <td>0.006288</td>
      <td>0.043353</td>
      <td>1</td>
    </tr>
    <tr>
      <th>n6/n3</th>
      <td>0.001570</td>
      <td>-0.003148</td>
      <td>-0.006289</td>
      <td>-0.000008</td>
      <td>0.047098</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
results_ols['age_adj_Beta']=results_pred['Beta_coeff']
results_ols['age_adj_pvalue']=results_pred['pvalue']
results_ols['age_adj_corr_pvalue']=results_pred['corr_pval']
#Save table S3
results_ols.to_csv('demo_clinical_health.csv')
```


```python
#Generate Spearman correlations between different gut microbiome measures reported in Figure 1B
alpha_uniqueness=pd.DataFrame()
cohort=[1,0,1,0,1,0,1,0,1,0,1,0]
metric=[]
coefficient=[]
pvalues=[]
measures=['min_bray','min_wunifrac','min_jaccard','min_unifrac','Shannon','Observed']
cohorts=[second_genome,genotek]
for x in measures:
    for y in cohorts:
        metric.append(x)
        coefficient.append(scipy.stats.spearmanr(y[x],y['age'])[0])
        pvalues.append(scipy.stats.spearmanr(y[x],y['age'])[1])
alpha_uniqueness['cohort']=cohort
alpha_uniqueness['metric']=metric
alpha_uniqueness['spearmanr']=coefficient
alpha_uniqueness['pvalue']=pvalues
```


```python
alpha_uniqueness
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
      <th>cohort</th>
      <th>metric</th>
      <th>spearmanr</th>
      <th>pvalue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>min_bray</td>
      <td>0.232494</td>
      <td>3.869477e-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>min_bray</td>
      <td>0.179939</td>
      <td>6.416813e-20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>min_wunifrac</td>
      <td>0.147400</td>
      <td>7.764145e-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>min_wunifrac</td>
      <td>0.132241</td>
      <td>2.238542e-11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>min_jaccard</td>
      <td>0.112567</td>
      <td>1.666384e-04</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>min_jaccard</td>
      <td>0.027083</td>
      <td>1.724932e-01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>min_unifrac</td>
      <td>0.118743</td>
      <td>7.101565e-05</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>min_unifrac</td>
      <td>0.018370</td>
      <td>3.548231e-01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>Shannon</td>
      <td>0.054083</td>
      <td>7.116560e-02</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>Shannon</td>
      <td>0.088277</td>
      <td>8.401979e-06</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>Observed</td>
      <td>0.069157</td>
      <td>2.097621e-02</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>Observed</td>
      <td>0.092031</td>
      <td>3.402486e-06</td>
    </tr>
  </tbody>
</table>
</div>


