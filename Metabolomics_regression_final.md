```python
import arivale_data_interface as adi
from arivale_data_interface import *
frozen_ss_path='/shared-data/snapshots/arivale_snapshot_ISB_2019-05-19_1330'
sn=list_snapshot_contents()
def get_frozen_snapshot(ss_name, ss_path=frozen_ss_path):
    return get_snapshot(ss_name, path=ss_path)
sn=list_snapshot_contents()
import matplotlib.pyplot as plt
from statsmodels.stats import multitest as multi
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn
# if you want plots to show up in the notebook you need to run this
%matplotlib inline
```

    /opt/conda/envs/arivale-py3/lib/python3.7/site-packages/patsy/constraint.py:13: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Mapping



```python
#Import demographic information on Arivale participants
whole_df=pd.read_csv('demographics_df.csv')
whole_df.set_index('public_client_id',inplace=True)
```


```python
#Import metabolomics data
metabolomics=pd.read_csv('whole_mets.csv')
print(metabolomics.shape)
mets=metabolomics[metabolomics.columns[0:652]].columns.tolist()
print(len(mets))
ind=[]
for x in metabolomics['public_client_id']:
    ind.append('0'+str(x))
metabolomics.index=ind
#keep all identified metabolites (exclude unidentified mets)
X = metabolomics[metabolomics.columns[0:652]]
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
Xcolumns=X.columns
X = scaler.fit_transform(X)
X=pd.DataFrame(data=X,columns=Xcolumns,index=metabolomics.index)
print (X.shape)
sns.set(font_scale=1.00)
X['BMI']=whole_df['BMI']
X['age']=whole_df['age']
X['sex']=whole_df['sex']
X['min']=whole_df['min_bray']
X['vendor']=whole_df['vendor']
X['Shannon']=whole_df['Shannon']
X['Observed']=whole_df['Observed']
X.shape
```

    (1476, 761)
    652
    (1476, 652)





    (1476, 659)




```python
#Square root transform Bray-Curtis Uniqueness
X['min']=np.sqrt(X['min'])
X.shape
```




    (1476, 659)




```python
#Perform linear regression on each metabolite individually predicting square root transformed Bray-Curtis Uniqueness,
#adjusting for covariates
import numpy as np
import statsmodels.formula.api as smf
p=[]
analyte=[]
test_value=[]
reg_df=pd.DataFrame(index=X.index,data=X)
print (reg_df.shape)
reg_df['age_sq']=reg_df['age']**2
for x in mets:
    reg_df['response']=X[x]
    results = smf.ols(formula = 'min ~ vendor+age_sq+age*sex+BMI+Shannon+response', data=reg_df).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[8]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[8]
    test_value.append(param)
results_prot=pd.DataFrame()
results_prot['analyte']=analyte
results_prot['pvalue']=p
results_prot.set_index('analyte')
results_prot['corr_pval']=multi.multipletests(results_prot['pvalue'], alpha=0.05, method='bonferroni', is_sorted=False,returnsorted=False)[1]
results_prot['covariate_adjusted_Beta_coeff_validation']=test_value
print (len(results_prot))
results_prot=results_prot.set_index('analyte')
print (len(results_prot[results_prot['corr_pval']<0.05]))
```

    (1476, 659)
    652
    8



```python
#Show significant mets
significant=results_prot[results_prot['corr_pval']<0.05]
significant.sort_values(by='corr_pval')
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
      <th>corr_pval</th>
      <th>covariate_adjusted_Beta_coeff_validation</th>
    </tr>
    <tr>
      <th>analyte</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>phenylacetylglutamine</th>
      <td>3.651374e-20</td>
      <td>2.380696e-17</td>
      <td>0.015775</td>
    </tr>
    <tr>
      <th>p-cresol glucuronide*</th>
      <td>6.755021e-13</td>
      <td>4.404274e-10</td>
      <td>0.011978</td>
    </tr>
    <tr>
      <th>6-hydroxyindole sulfate</th>
      <td>7.061798e-09</td>
      <td>4.604292e-06</td>
      <td>0.009598</td>
    </tr>
    <tr>
      <th>3-indoxyl sulfate</th>
      <td>5.238089e-07</td>
      <td>3.415234e-04</td>
      <td>0.008270</td>
    </tr>
    <tr>
      <th>lithocholate sulfate (1)</th>
      <td>7.391230e-07</td>
      <td>4.819082e-04</td>
      <td>0.008154</td>
    </tr>
    <tr>
      <th>p-cresol sulfate</th>
      <td>5.055170e-06</td>
      <td>3.295971e-03</td>
      <td>0.007949</td>
    </tr>
    <tr>
      <th>indoleacetate</th>
      <td>9.548799e-06</td>
      <td>6.225817e-03</td>
      <td>0.007364</td>
    </tr>
    <tr>
      <th>taurolithocholate 3-sulfate</th>
      <td>6.668268e-05</td>
      <td>4.347711e-02</td>
      <td>0.006751</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Plot phenylacetylglutamine vs BRay-Curtis Uniqueness
sns.set(font_scale=.5,context='poster',font='Arial',style='white')
plt.figure(figsize=[40,40], dpi=200)
reg_df.dropna(inplace=True)
ax=sns.lmplot('phenylacetylglutamine','min', reg_df,fit_reg=True,scatter_kws={"s": 20,'color':'darkred'},line_kws={'color': 'black'})
ax.set_axis_labels('phenylacetylglutamine', 'Square Root Minimum Bray-Curtis')
print(scipy.stats.pearsonr(reg_df['phenylacetylglutamine'],reg_df['min']))
scipy.stats.spearmanr(reg_df['phenylacetylglutamine'],reg_df['min'])
```

    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.
    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.


    (0.28905907419865545, 4.288469198615304e-29)





    SpearmanrResult(correlation=0.3087422991229005, pvalue=3.730334920861438e-33)




    <Figure size 8000x8000 with 0 Axes>



![png](output_6_4.png)



```python
#Import metabolite metadata
metabolomics_metadata = get_snapshot('metabolomics_metadata')
metabolomics_metadata.set_index('BIOCHEMICAL_NAME',inplace=True)
metabolomics_metadata=metabolomics_metadata[metabolomics_metadata.index.isin(results_prot.index.tolist())]
```


```python
results_prot['SUPER_PATHWAY']=metabolomics_metadata['SUPER_PATHWAY']
results_prot['SUB_PATHWAY']=metabolomics_metadata['SUB_PATHWAY']
results_prot['log_p']=-np.log10(results_prot['pvalue'])
pathway=[]
for x in results_prot['SUB_PATHWAY']:
    if x=='Phenylalanine and Tyrosine Metabolism':
        pathway.append('Phenyl/Tyrosine')
    elif x=='Tryptophan Metabolism':
        pathway.append('Tryptophan')
    elif x=='Leucine, Isoleucine and Valine Metabolism':
        pathway.append('BCAA')
    else:
        pathway.append('other')
results_prot['path_prot']=pathway      
significant=results_prot[results_prot['corr_pval']<0.05]
significant.sort_values(by='pvalue')
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
      <th>corr_pval</th>
      <th>covariate_adjusted_Beta_coeff_validation</th>
      <th>SUPER_PATHWAY</th>
      <th>SUB_PATHWAY</th>
      <th>log_p</th>
      <th>path_prot</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>phenylacetylglutamine</th>
      <td>3.651374e-20</td>
      <td>2.380696e-17</td>
      <td>0.015775</td>
      <td>Amino Acid</td>
      <td>Phenylalanine and Tyrosine Metabolism</td>
      <td>19.437544</td>
      <td>Phenyl/Tyrosine</td>
    </tr>
    <tr>
      <th>p-cresol glucuronide*</th>
      <td>6.755021e-13</td>
      <td>4.404274e-10</td>
      <td>0.011978</td>
      <td>Amino Acid</td>
      <td>Phenylalanine and Tyrosine Metabolism</td>
      <td>12.170373</td>
      <td>Phenyl/Tyrosine</td>
    </tr>
    <tr>
      <th>6-hydroxyindole sulfate</th>
      <td>7.061798e-09</td>
      <td>4.604292e-06</td>
      <td>0.009598</td>
      <td>Xenobiotics</td>
      <td>Chemical</td>
      <td>8.151085</td>
      <td>other</td>
    </tr>
    <tr>
      <th>3-indoxyl sulfate</th>
      <td>5.238089e-07</td>
      <td>3.415234e-04</td>
      <td>0.008270</td>
      <td>Amino Acid</td>
      <td>Tryptophan Metabolism</td>
      <td>6.280827</td>
      <td>Tryptophan</td>
    </tr>
    <tr>
      <th>lithocholate sulfate (1)</th>
      <td>7.391230e-07</td>
      <td>4.819082e-04</td>
      <td>0.008154</td>
      <td>Lipid</td>
      <td>Secondary Bile Acid Metabolism</td>
      <td>6.131283</td>
      <td>other</td>
    </tr>
    <tr>
      <th>p-cresol sulfate</th>
      <td>5.055170e-06</td>
      <td>3.295971e-03</td>
      <td>0.007949</td>
      <td>Amino Acid</td>
      <td>Phenylalanine and Tyrosine Metabolism</td>
      <td>5.296264</td>
      <td>Phenyl/Tyrosine</td>
    </tr>
    <tr>
      <th>indoleacetate</th>
      <td>9.548799e-06</td>
      <td>6.225817e-03</td>
      <td>0.007364</td>
      <td>Amino Acid</td>
      <td>Tryptophan Metabolism</td>
      <td>5.020051</td>
      <td>Tryptophan</td>
    </tr>
    <tr>
      <th>taurolithocholate 3-sulfate</th>
      <td>6.668268e-05</td>
      <td>4.347711e-02</td>
      <td>0.006751</td>
      <td>Lipid</td>
      <td>Secondary Bile Acid Metabolism</td>
      <td>4.175987</td>
      <td>other</td>
    </tr>
  </tbody>
</table>
</div>




```python
results_prot['SUPER_PATHWAY']=results_prot['SUPER_PATHWAY'].fillna('Unknown')
pathway=[]
color=[]
for x in results_prot['SUPER_PATHWAY']:
    if x=='Amino Acid':
        pathway.append(1)
        color.append('#FF8C00')
    elif x=='Carbohydrate':
        pathway.append(2)
        color.append('#FF8C00')
    elif x=='Cofactors and Vitamins':
        pathway.append(3)
        color.append('#DAA520')
    elif x=='Energy':
        pathway.append(4)
        color.append('#9ACD32')
    elif x=='Lipid':
        pathway.append(5)
        color.append('#008000')
    elif x=='Nucleotide':
        pathway.append(6)
        color.append('#2F4F4F')
    elif x=='Partially Characterized Molecules':
        pathway.append(7)
        color.append('#00FFFF')
    elif x=='Peptide':
        pathway.append(8)
        color.append('#00008B')
    elif x=='Xenobiotics':
        pathway.append(9)
        color.append('#800080')
    else:
        pathway.append(10)
        color.append('#808080')
results_prot['path']=pathway
results_prot['color']=color
results_prot.groupby(by='path').size()
```




    path
    1     138
    2      15
    3      18
    4       8
    5     357
    6      23
    7       3
    8      14
    9      53
    10     23
    dtype: int64




```python
#Save results for figure generation
results_prot.to_csv('mets_min_reg.csv')
```


```python
#Save results for supplementary table S4
results=results_prot[results_prot['pvalue']<0.01].sort_values(by='corr_pval',ascending=True)
results.to_csv('metabolomics_results.csv')
```


```python
#Spearman correlations shown in Figure 3B
spearman=pd.DataFrame()
analyte=[]
correlation=[]
for x in significant.index.tolist():
    df=reg_df.dropna()
    df['response']=df[x]
    print(scipy.stats.spearmanr(df['response'],df['min'])[0])
    correlation.append(scipy.stats.spearmanr(df['response'],df['min'])[0])
    analyte.append(x)
spearman['analyte']=analyte
spearman['spearman']=correlation
```

    0.1404343239725758
    0.13815494340305998
    0.27184948706313866
    0.3087422991229005
    0.1137966341460758
    0.23902266656443544
    0.1700485097858178
    0.1214370150431665



```python
spearman.index=spearman['analyte']
spearman['SUPER_PATHWAY']=results_prot['SUPER_PATHWAY']
spearman
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
      <th>spearman</th>
      <th>SUPER_PATHWAY</th>
    </tr>
    <tr>
      <th>analyte</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3-indoxyl sulfate</th>
      <td>3-indoxyl sulfate</td>
      <td>0.140434</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>indoleacetate</th>
      <td>indoleacetate</td>
      <td>0.138155</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>p-cresol sulfate</th>
      <td>p-cresol sulfate</td>
      <td>0.271849</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>phenylacetylglutamine</th>
      <td>phenylacetylglutamine</td>
      <td>0.308742</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>taurolithocholate 3-sulfate</th>
      <td>taurolithocholate 3-sulfate</td>
      <td>0.113797</td>
      <td>Lipid</td>
    </tr>
    <tr>
      <th>p-cresol glucuronide*</th>
      <td>p-cresol glucuronide*</td>
      <td>0.239023</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>6-hydroxyindole sulfate</th>
      <td>6-hydroxyindole sulfate</td>
      <td>0.170049</td>
      <td>Xenobiotics</td>
    </tr>
    <tr>
      <th>lithocholate sulfate (1)</th>
      <td>lithocholate sulfate (1)</td>
      <td>0.121437</td>
      <td>Lipid</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Generate figure 3C
spearman.sort_values(by='spearman',ascending=False,inplace=True)
plt.figure(figsize=[10,10], dpi=200)
sns.set(font_scale=1.75,context='poster',font='Arial',style='white')
colors = {'Amino Acid': 'darkred', 'Xenobiotics': 'darkblue','Lipid':'gold','Cofactors and Vitamins':'orange'}
#plt.ylim(0,0.40)
spearman['spearman'].plot(kind='bar', color=[colors[i] for i in spearman['SUPER_PATHWAY']],edgecolor='k')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f8ab7d3dac8>



    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.
    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.



![png](output_14_2.png)



```python
#Percent variance explained by phenylacetylglutamine reported in the results
results = smf.ols(formula = 'min ~ phenylacetylglutamine', data=reg_df).fit()
results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>min</td>       <th>  R-squared:         </th> <td>   0.084</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.083</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   131.0</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 25 Mar 2020</td> <th>  Prob (F-statistic):</th> <td>4.29e-29</td>
</tr>
<tr>
  <th>Time:</th>                 <td>14:45:12</td>     <th>  Log-Likelihood:    </th> <td>  1975.0</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1439</td>      <th>  AIC:               </th> <td>  -3946.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1437</td>      <th>  BIC:               </th> <td>  -3936.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>             <td>    0.5016</td> <td>    0.002</td> <td>  310.051</td> <td> 0.000</td> <td>    0.498</td> <td>    0.505</td>
</tr>
<tr>
  <th>phenylacetylglutamine</th> <td>    0.0186</td> <td>    0.002</td> <td>   11.446</td> <td> 0.000</td> <td>    0.015</td> <td>    0.022</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>168.580</td> <th>  Durbin-Watson:     </th> <td>   2.024</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 467.475</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 0.621</td>  <th>  Prob(JB):          </th> <td>3.08e-102</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.501</td>  <th>  Cond. No.          </th> <td>    1.00</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
#covariate adjusted beta-coefficient for phenylacetylglutamine reported in the results
results = smf.ols(formula = 'min ~ vendor+age_sq+age*sex+BMI+Observed+phenylacetylglutamine', data=reg_df).fit()
results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>min</td>       <th>  R-squared:         </th> <td>   0.124</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.119</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   25.39</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 25 Mar 2020</td> <th>  Prob (F-statistic):</th> <td>6.88e-37</td>
</tr>
<tr>
  <th>Time:</th>                 <td>14:45:12</td>     <th>  Log-Likelihood:    </th> <td>  2007.8</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1439</td>      <th>  AIC:               </th> <td>  -3998.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1430</td>      <th>  BIC:               </th> <td>  -3950.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     8</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>             <td>    0.5259</td> <td>    0.024</td> <td>   21.672</td> <td> 0.000</td> <td>    0.478</td> <td>    0.574</td>
</tr>
<tr>
  <th>sex[T.M]</th>              <td>    0.0035</td> <td>    0.014</td> <td>    0.243</td> <td> 0.808</td> <td>   -0.024</td> <td>    0.031</td>
</tr>
<tr>
  <th>vendor</th>                <td>   -0.0157</td> <td>    0.003</td> <td>   -4.533</td> <td> 0.000</td> <td>   -0.022</td> <td>   -0.009</td>
</tr>
<tr>
  <th>age_sq</th>                <td> 2.037e-05</td> <td> 9.49e-06</td> <td>    2.147</td> <td> 0.032</td> <td> 1.76e-06</td> <td>  3.9e-05</td>
</tr>
<tr>
  <th>age</th>                   <td>   -0.0011</td> <td>    0.001</td> <td>   -1.213</td> <td> 0.225</td> <td>   -0.003</td> <td>    0.001</td>
</tr>
<tr>
  <th>age:sex[T.M]</th>          <td>-4.659e-05</td> <td>    0.000</td> <td>   -0.165</td> <td> 0.869</td> <td>   -0.001</td> <td>    0.001</td>
</tr>
<tr>
  <th>BMI</th>                   <td>   -0.0006</td> <td>    0.000</td> <td>   -2.457</td> <td> 0.014</td> <td>   -0.001</td> <td>   -0.000</td>
</tr>
<tr>
  <th>Observed</th>              <td> 3.068e-05</td> <td> 1.64e-05</td> <td>    1.867</td> <td> 0.062</td> <td>-1.55e-06</td> <td> 6.29e-05</td>
</tr>
<tr>
  <th>phenylacetylglutamine</th> <td>    0.0151</td> <td>    0.002</td> <td>    8.854</td> <td> 0.000</td> <td>    0.012</td> <td>    0.018</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>191.388</td> <th>  Durbin-Watson:     </th> <td>   2.042</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 546.248</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 0.692</td>  <th>  Prob(JB):          </th> <td>2.42e-119</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.682</td>  <th>  Cond. No.          </th> <td>4.33e+04</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 4.33e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.


