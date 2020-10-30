```python
from statsmodels.stats import multitest as multi
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import pandas as pd
# if you want plots to show up in the notebook you need to run this
%matplotlib inline
```


```python
#Import demographic information on Arivale participants
whole_df=pd.read_csv('results_merged.csv')
whole_df.set_index('public_client_id',inplace=True)
whole_df.shape
```




    (3653, 38)




```python
#import metabolomics
metabolomics=pd.read_csv('whole_mets.csv')
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
X['min_genus']=whole_df['min_bray_genus']
X['min']=whole_df['min_bray']
X['vendor']=whole_df['vendor']
X['Shannon']=whole_df['Shannon']
X['days_since_first_draw']=whole_df['days_since_first_draw']
X.shape
```

    (1476, 761)
    652
    (1476, 652)





    (1476, 660)




```python
#Square root transform Bray-Curtis Uniqueness
X['min_genus_sqrt']=np.sqrt(X['min_genus'])
X['min_sqrt']=np.sqrt(X['min'])
X.shape
```




    (1476, 662)




```python
X=X[X['days_since_first_draw']>=-21]
print(X.shape)
X=X[X['days_since_first_draw']<=21]
no_outliers=X[X['min_genus_sqrt']<(X['min_genus_sqrt'].mean()+3*X['min_genus_sqrt'].std())]
no_outliers=no_outliers[no_outliers['min_genus_sqrt']>(X['min_genus_sqrt'].mean()-3*X['min_genus_sqrt'].std())]
no_outliers.shape
```

    (1475, 662)





    (1459, 662)




```python
print('IQR',np.percentile(no_outliers['days_since_first_draw'],[25,75]))
no_outliers['days_since_first_draw'].median()
```

    IQR [-2.  4.]





    0.0




```python
no_outliers['min_genus_sqrt'].hist(bins=25)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc8bf0fff98>




![png](output_7_1.png)



```python
#Perform linear regression on each metabolite individually predicting square root transformed Bray-Curtis Uniqueness,
#adjusting for covariates
import numpy as np
import statsmodels.formula.api as smf
p=[]
analyte=[]
test_value=[]
reg_df=pd.DataFrame(index=no_outliers.index,data=no_outliers)
print (reg_df.shape)
reg_df['age_sq']=reg_df['age']**2
for x in mets:
    reg_df['response']=X[x]
    results = smf.ols(formula = 'min_genus_sqrt ~ vendor+age_sq+age*sex+BMI+Shannon+response', data=reg_df).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[8]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[8]
    test_value.append(param)
results_met=pd.DataFrame()
results_met['analyte']=analyte
results_met['pvalue']=p
results_met.set_index('analyte')
results_met['corr_pval']=multi.multipletests(results_met['pvalue'], alpha=0.05, method='bonferroni', is_sorted=False,returnsorted=False)[1]
results_met['covariate_adjusted_Beta_coeff']=test_value
print (len(results_met))
results_met=results_met.set_index('analyte')
print (len(results_met[results_met['corr_pval']<0.025]))
```

    (1459, 662)
    652
    6



```python
#Show significant mets
significant=results_met[results_met['corr_pval']<0.025]
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
      <th>covariate_adjusted_Beta_coeff</th>
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
      <td>1.329990e-18</td>
      <td>8.671537e-16</td>
      <td>0.013569</td>
    </tr>
    <tr>
      <th>p-cresol glucuronide*</th>
      <td>2.338069e-12</td>
      <td>1.524421e-09</td>
      <td>0.010512</td>
    </tr>
    <tr>
      <th>p-cresol sulfate</th>
      <td>8.660483e-08</td>
      <td>5.646635e-05</td>
      <td>0.008643</td>
    </tr>
    <tr>
      <th>6-hydroxyindole sulfate</th>
      <td>3.284570e-07</td>
      <td>2.141540e-04</td>
      <td>0.007517</td>
    </tr>
    <tr>
      <th>indoleacetate</th>
      <td>7.017242e-06</td>
      <td>4.575242e-03</td>
      <td>0.006706</td>
    </tr>
    <tr>
      <th>3-indoxyl sulfate</th>
      <td>8.977596e-06</td>
      <td>5.853393e-03</td>
      <td>0.006528</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Perform linear regression on each metabolite individually predicting square root transformed ASV Bray-Curtis Uniqueness,
#adjusting for covariates
import numpy as np
import statsmodels.formula.api as smf
p=[]
analyte=[]
test_value=[]
reg_df=pd.DataFrame(index=no_outliers.index,data=no_outliers)
print (reg_df.shape)
reg_df['age_sq']=reg_df['age']**2
for x in mets:
    reg_df['response']=X[x]
    results = smf.ols(formula = 'min_sqrt ~ vendor+age_sq+age*sex+BMI+Shannon+response', data=reg_df).fit()
    p_extract=results.pvalues.tolist()
    p_test=p_extract[8]
    analyte.append(x)
    p.append(p_test)
    parameters=results.params.tolist()
    param=parameters[8]
    test_value.append(param)
results_met_ASV=pd.DataFrame()
results_met_ASV['analyte']=analyte
results_met_ASV['pvalue']=p
results_met_ASV.set_index('analyte')
results_met_ASV['corr_pval']=multi.multipletests(results_met_ASV['pvalue'], alpha=0.05, method='bonferroni', is_sorted=False,returnsorted=False)[1]
results_met_ASV['covariate_adjusted_Beta_coeff']=test_value
print (len(results_met_ASV))
results_met_ASV=results_met_ASV.set_index('analyte')
print (len(results_met_ASV[results_met_ASV['corr_pval']<0.025]))
```

    (1459, 664)
    652
    5



```python
#Show significant ASV-level mets
significant_ASV=results_met_ASV[results_met_ASV['corr_pval']<0.025]
significant_ASV.sort_values(by='corr_pval')
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
      <th>covariate_adjusted_Beta_coeff</th>
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
      <td>5.018332e-12</td>
      <td>3.271953e-09</td>
      <td>0.007999</td>
    </tr>
    <tr>
      <th>p-cresol glucuronide*</th>
      <td>6.890419e-11</td>
      <td>4.492553e-08</td>
      <td>0.007319</td>
    </tr>
    <tr>
      <th>p-cresol sulfate</th>
      <td>7.044743e-08</td>
      <td>4.593172e-05</td>
      <td>0.006506</td>
    </tr>
    <tr>
      <th>indoleacetate</th>
      <td>8.863066e-06</td>
      <td>5.778719e-03</td>
      <td>0.004958</td>
    </tr>
    <tr>
      <th>indolepropionate</th>
      <td>1.908190e-05</td>
      <td>1.244140e-02</td>
      <td>-0.004984</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Plot phenylacetylglutamine vs ASV Bray-Curtis Uniqueness
#adjust uniqueness for vendor
results = smf.ols(formula = 'min~ vendor', data=reg_df).fit()
#take residuals
reg_df['min_vend']=results.resid+reg_df['min'].mean()
#generate figure
sns.set(font_scale=.5,context='poster',font='Arial',style='white')
plt.figure(figsize=[40,40], dpi=200)
reg_df.dropna(inplace=True)
ax=sns.lmplot('phenylacetylglutamine','min_vend', reg_df,fit_reg=True,scatter_kws={"s": 20,'color':'darkred'},line_kws={'color': 'black'})
plt.ylim(0.3, 0.9)
ax.set_axis_labels('phenylacetylglutamine', 'Square Root Minimum Bray-Curtis')
print(scipy.stats.pearsonr(reg_df['phenylacetylglutamine'],reg_df['min_vend']))
scipy.stats.spearmanr(reg_df['phenylacetylglutamine'],reg_df['min_vend'])
```

    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.
    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.


    (0.2301649841822218, 2.02474350912233e-18)





    SpearmanrResult(correlation=0.24120365700763693, pvalue=3.988654356589762e-20)




    <Figure size 8000x8000 with 0 Axes>



![png](output_12_4.png)



```python
#Plot phenylacetylglutamine vs genus Bray-Curtis Uniqueness
#adjust uniqueness for vendor
results = smf.ols(formula = 'min_genus~ vendor', data=reg_df).fit()
#take residuals
reg_df['min_vend_genus']=results.resid+reg_df['min_genus'].mean()
#generate figure
sns.set(font_scale=.5,context='poster',font='Arial',style='white')
reg_df.dropna(inplace=True)
ax=sns.lmplot('phenylacetylglutamine','min_vend_genus', reg_df,fit_reg=True,scatter_kws={"s": 20,'color':'darkred'},line_kws={'color': 'black'})
plt.ylim(0, 0.6)
plt.figure(figsize=[40,40], dpi=200)
ax.set_axis_labels('phenylacetylglutamine', 'Square Root Minimum Bray-Curtis')
print(scipy.stats.pearsonr(reg_df['phenylacetylglutamine'],reg_df['min_vend_genus']))
scipy.stats.spearmanr(reg_df['phenylacetylglutamine'],reg_df['min_vend_genus'])
```

    (0.29001810838352116, 9.553047954541134e-29)





    SpearmanrResult(correlation=0.30297378123277513, pvalue=2.42211161587287e-31)




![png](output_13_2.png)



    <Figure size 8000x8000 with 0 Axes>



```python
#Import metabolite metadata
metabolomics_metadata = pd.read_csv('metabolomics_metadata.csv')
metabolomics_metadata.set_index('BIOCHEMICAL_NAME',inplace=True)
metabolomics_metadata=metabolomics_metadata[metabolomics_metadata.index.isin(results_met.index.tolist())]
```


```python
#Add metabolites pathway annotations
results_met['SUPER_PATHWAY']=metabolomics_metadata['SUPER_PATHWAY']
results_met['SUB_PATHWAY']=metabolomics_metadata['SUB_PATHWAY']
results_met['log_p']=-np.log10(results_met['pvalue'])
pathway=[]
for x in results_met['SUB_PATHWAY']:
    if x=='Phenylalanine and Tyrosine Metabolism':
        pathway.append('Phenyl/Tyrosine')
    elif x=='Tryptophan Metabolism':
        pathway.append('Tryptophan')
    elif x=='Leucine, Isoleucine and Valine Metabolism':
        pathway.append('BCAA')
    else:
        pathway.append('other')
results_met['path_met']=pathway      
significant=results_met[results_met['corr_pval']<0.05]
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
      <th>covariate_adjusted_Beta_coeff</th>
      <th>SUPER_PATHWAY</th>
      <th>SUB_PATHWAY</th>
      <th>log_p</th>
      <th>path_met</th>
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
      <td>1.329990e-18</td>
      <td>8.671537e-16</td>
      <td>0.013569</td>
      <td>Amino Acid</td>
      <td>Phenylalanine and Tyrosine Metabolism</td>
      <td>17.876151</td>
      <td>Phenyl/Tyrosine</td>
    </tr>
    <tr>
      <th>p-cresol glucuronide*</th>
      <td>2.338069e-12</td>
      <td>1.524421e-09</td>
      <td>0.010512</td>
      <td>Amino Acid</td>
      <td>Phenylalanine and Tyrosine Metabolism</td>
      <td>11.631143</td>
      <td>Phenyl/Tyrosine</td>
    </tr>
    <tr>
      <th>p-cresol sulfate</th>
      <td>8.660483e-08</td>
      <td>5.646635e-05</td>
      <td>0.008643</td>
      <td>Amino Acid</td>
      <td>Phenylalanine and Tyrosine Metabolism</td>
      <td>7.062458</td>
      <td>Phenyl/Tyrosine</td>
    </tr>
    <tr>
      <th>6-hydroxyindole sulfate</th>
      <td>3.284570e-07</td>
      <td>2.141540e-04</td>
      <td>0.007517</td>
      <td>Xenobiotics</td>
      <td>Chemical</td>
      <td>6.483521</td>
      <td>other</td>
    </tr>
    <tr>
      <th>indoleacetate</th>
      <td>7.017242e-06</td>
      <td>4.575242e-03</td>
      <td>0.006706</td>
      <td>Amino Acid</td>
      <td>Tryptophan Metabolism</td>
      <td>5.153834</td>
      <td>Tryptophan</td>
    </tr>
    <tr>
      <th>3-indoxyl sulfate</th>
      <td>8.977596e-06</td>
      <td>5.853393e-03</td>
      <td>0.006528</td>
      <td>Amino Acid</td>
      <td>Tryptophan Metabolism</td>
      <td>5.046840</td>
      <td>Tryptophan</td>
    </tr>
    <tr>
      <th>trimethylamine N-oxide</th>
      <td>7.259979e-05</td>
      <td>4.733506e-02</td>
      <td>0.006057</td>
      <td>Lipid</td>
      <td>Phospholipid Metabolism</td>
      <td>4.139065</td>
      <td>other</td>
    </tr>
  </tbody>
</table>
</div>




```python
results_met['SUPER_PATHWAY']=results_met['SUPER_PATHWAY'].fillna('Unknown')
pathway=[]
color=[]
for x in results_met['SUPER_PATHWAY']:
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
results_met['path']=pathway
results_met['color']=color
results_met.groupby(by='path').size()
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
results_met.to_csv('mets_min_reg.csv')
```


```python
#Spearman correlations shown in Figure 3B
results=results_met[results_met['corr_pval']<0.025].sort_values(by='corr_pval',ascending=True)
spearman=pd.DataFrame()
analyte=[]
correlation=[]
for x in results.index.tolist():
    df=reg_df.dropna()
    df['response']=df[x]
    print(scipy.stats.spearmanr(df['response'],df['min_vend_genus'])[0])
    correlation.append(scipy.stats.spearmanr(df['response'],df['min_vend_genus'])[0])
    analyte.append(x)
spearman['analyte']=analyte
spearman['spearman']=correlation
```

    0.30297378123277513
    0.23192424203976353
    0.26365330978018214
    0.1444823147252046
    0.12265365122276804
    0.11358627964067837



```python
spearman.index=spearman['analyte']
spearman['SUPER_PATHWAY']=results_met['SUPER_PATHWAY']
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
      <th>phenylacetylglutamine</th>
      <td>phenylacetylglutamine</td>
      <td>0.302974</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>p-cresol glucuronide*</th>
      <td>p-cresol glucuronide*</td>
      <td>0.231924</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>p-cresol sulfate</th>
      <td>p-cresol sulfate</td>
      <td>0.263653</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>6-hydroxyindole sulfate</th>
      <td>6-hydroxyindole sulfate</td>
      <td>0.144482</td>
      <td>Xenobiotics</td>
    </tr>
    <tr>
      <th>indoleacetate</th>
      <td>indoleacetate</td>
      <td>0.122654</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>3-indoxyl sulfate</th>
      <td>3-indoxyl sulfate</td>
      <td>0.113586</td>
      <td>Amino Acid</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Generate figure 3C
spearman.sort_values(by='spearman',ascending=False,inplace=True)
plt.figure(figsize=[10,10], dpi=200)
sns.set(font_scale=1.75,context='poster',font='Arial',style='white')
colors = {'Amino Acid': 'darkred', 'Xenobiotics': 'darkblue','Lipid':'gold','Cofactors and Vitamins':'orange',"Carbohydrate":"grey"}
#plt.ylim(0,0.40)
spearman['spearman'].plot(kind='bar', color=[colors[i] for i in spearman['SUPER_PATHWAY']],edgecolor='k')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc8bece1198>



    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.
    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.



![png](output_20_2.png)



```python
#Spearman correlations shown in Figure 3C
results_ASV=results_met_ASV[results_met_ASV['corr_pval']<0.025].sort_values(by='corr_pval',ascending=True)
spearman_ASV=pd.DataFrame()
analyte=[]
correlation=[]
for x in results_ASV.index.tolist():
    df=reg_df.dropna()
    df['response']=df[x]
    print(scipy.stats.spearmanr(df['response'],df['min_vend'])[0])
    correlation.append(scipy.stats.spearmanr(df['response'],df['min_vend'])[0])
    analyte.append(x)
spearman_ASV['analyte']=analyte
spearman_ASV['spearman']=correlation
```

    0.24120365700763693
    0.20306610670346886
    0.2227327362375886
    0.10078432953105557
    -0.1030012466740434



```python
spearman_ASV.index=spearman_ASV['analyte']
spearman_ASV['SUPER_PATHWAY']=results_met['SUPER_PATHWAY']
spearman_ASV
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
      <th>phenylacetylglutamine</th>
      <td>phenylacetylglutamine</td>
      <td>0.241204</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>p-cresol glucuronide*</th>
      <td>p-cresol glucuronide*</td>
      <td>0.203066</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>p-cresol sulfate</th>
      <td>p-cresol sulfate</td>
      <td>0.222733</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>indoleacetate</th>
      <td>indoleacetate</td>
      <td>0.100784</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>indolepropionate</th>
      <td>indolepropionate</td>
      <td>-0.103001</td>
      <td>Amino Acid</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Generate figure 3C
spearman_ASV.sort_values(by='spearman',ascending=False,inplace=True)
plt.figure(figsize=[10,10], dpi=200)
sns.set(font_scale=1.75,context='poster',font='Arial',style='white')
colors = {'Amino Acid': 'darkred', 'Xenobiotics': 'darkblue','Lipid':'gold','Cofactors and Vitamins':'orange',"Carbohydrate":"grey"}
plt.ylim(-0.15,0.30)
spearman_ASV['spearman'].plot(kind='bar', color=[colors[i] for i in spearman_ASV['SUPER_PATHWAY']],edgecolor='k')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc8bec4ada0>




![png](output_23_1.png)



```python
#Percent variance explained by phenylacetylglutamine reported in the results
results = smf.ols(formula = 'min_sqrt ~ vendor+age+age_sq+sex+Shannon+phenylacetylglutamine', data=reg_df).fit()
print(results.summary())
results.pvalues
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:               min_sqrt   R-squared:                       0.246
    Model:                            OLS   Adj. R-squared:                  0.243
    Method:                 Least Squares   F-statistic:                     76.51
    Date:                Fri, 30 Oct 2020   Prob (F-statistic):           8.52e-83
    Time:                        15:44:47   Log-Likelihood:                 2531.3
    No. Observations:                1411   AIC:                            -5049.
    Df Residuals:                    1404   BIC:                            -5012.
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    =========================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------------
    Intercept                 0.7196      0.018     39.412      0.000       0.684       0.755
    sex[T.M]                 -0.0105      0.002     -4.608      0.000      -0.015      -0.006
    vendor                    0.0400      0.002     17.262      0.000       0.035       0.045
    age                      -0.0008      0.001     -1.292      0.197      -0.002       0.000
    age_sq                 1.308e-05   6.47e-06      2.021      0.043    3.86e-07    2.58e-05
    Shannon                   0.0010      0.003      0.412      0.680      -0.004       0.006
    phenylacetylglutamine     0.0081      0.001      7.033      0.000       0.006       0.010
    ==============================================================================
    Omnibus:                       27.354   Durbin-Watson:                   1.971
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               30.018
    Skew:                           0.299   Prob(JB):                     3.03e-07
    Kurtosis:                       3.391   Cond. No.                     4.70e+04
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 4.7e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.





    Intercept                2.240367e-229
    sex[T.M]                  4.432441e-06
    vendor                    1.057765e-60
    age                       1.966011e-01
    age_sq                    4.343713e-02
    Shannon                   6.802603e-01
    phenylacetylglutamine     3.156710e-12
    dtype: float64




```python
#covariate adjusted beta-coefficient for ASV-level uniqueness by phenylacetylglutamine
results = smf.ols(formula = 'min_sqrt ~ vendor+age+sex+Shannon+phenylacetylglutamine', data=reg_df).fit()
print(results.summary())
results.pvalues
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:               min_sqrt   R-squared:                       0.244
    Model:                            OLS   Adj. R-squared:                  0.242
    Method:                 Least Squares   F-statistic:                     90.80
    Date:                Fri, 30 Oct 2020   Prob (F-statistic):           6.41e-83
    Time:                        15:44:48   Log-Likelihood:                 2529.3
    No. Observations:                1411   AIC:                            -5047.
    Df Residuals:                    1405   BIC:                            -5015.
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    =========================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------------
    Intercept                 0.6907      0.011     60.708      0.000       0.668       0.713
    sex[T.M]                 -0.0104      0.002     -4.553      0.000      -0.015      -0.006
    vendor                    0.0394      0.002     17.125      0.000       0.035       0.044
    age                       0.0004   9.88e-05      4.542      0.000       0.000       0.001
    Shannon                   0.0010      0.003      0.406      0.685      -0.004       0.006
    phenylacetylglutamine     0.0082      0.001      7.152      0.000       0.006       0.010
    ==============================================================================
    Omnibus:                       27.088   Durbin-Watson:                   1.967
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               29.458
    Skew:                           0.301   Prob(JB):                     4.01e-07
    Kurtosis:                       3.371   Cond. No.                         543.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.





    Intercept                0.000000e+00
    sex[T.M]                 5.741754e-06
    vendor                   7.422487e-60
    age                      6.044962e-06
    Shannon                  6.846597e-01
    phenylacetylglutamine    1.374419e-12
    dtype: float64




```python
#covariate adjusted beta-coefficient for phenylacetylglutamine with ASV-level uniqueness as the dependent variable 
results = smf.ols(formula = 'age ~ vendor+BMI+indoleacetate+sex', data=reg_df).fit()
print(results.summary())
results.pvalues
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                    age   R-squared:                       0.036
    Model:                            OLS   Adj. R-squared:                  0.033
    Method:                 Least Squares   F-statistic:                     13.21
    Date:                Fri, 30 Oct 2020   Prob (F-statistic):           1.44e-10
    Time:                        15:44:48   Log-Likelihood:                -5411.6
    No. Observations:                1411   AIC:                         1.083e+04
    Df Residuals:                    1406   BIC:                         1.086e+04
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    Intercept        47.7971      1.378     34.674      0.000      45.093      50.501
    sex[T.M]         -1.3717      0.639     -2.148      0.032      -2.624      -0.119
    vendor           -4.2440      0.625     -6.787      0.000      -5.471      -3.017
    BMI               0.1344      0.048      2.775      0.006       0.039       0.229
    indoleacetate     0.5697      0.305      1.865      0.062      -0.029       1.169
    ==============================================================================
    Omnibus:                       11.539   Durbin-Watson:                   1.873
    Prob(Omnibus):                  0.003   Jarque-Bera (JB):                8.379
    Skew:                          -0.064   Prob(JB):                       0.0152
    Kurtosis:                       2.645   Cond. No.                         132.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.





    Intercept        6.806597e-191
    sex[T.M]          3.187184e-02
    vendor            1.684991e-11
    BMI               5.587945e-03
    indoleacetate     6.232845e-02
    dtype: float64




```python
#covariate adjusted beta-coefficient for phenylacetylglutamine with genus-level uniqueness as the dependent variable
results = smf.ols(formula = 'min_genus_sqrt ~ vendor+age+age_sq+sex+Shannon+phenylacetylglutamine', data=reg_df).fit()
print(results.summary())
results.pvalues
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:         min_genus_sqrt   R-squared:                       0.199
    Model:                            OLS   Adj. R-squared:                  0.195
    Method:                 Least Squares   F-statistic:                     58.01
    Date:                Fri, 30 Oct 2020   Prob (F-statistic):           3.00e-64
    Time:                        15:44:48   Log-Likelihood:                 2137.5
    No. Observations:                1411   AIC:                            -4261.
    Df Residuals:                    1404   BIC:                            -4224.
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    =========================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------------
    Intercept                 0.4613      0.024     19.113      0.000       0.414       0.509
    sex[T.M]                 -0.0041      0.003     -1.350      0.177      -0.010       0.002
    vendor                   -0.0369      0.003    -12.034      0.000      -0.043      -0.031
    age                      -0.0006      0.001     -0.709      0.479      -0.002       0.001
    age_sq                 1.187e-05   8.55e-06      1.388      0.165   -4.91e-06    2.87e-05
    Shannon                   0.0129      0.003      3.872      0.000       0.006       0.019
    phenylacetylglutamine     0.0136      0.002      8.953      0.000       0.011       0.017
    ==============================================================================
    Omnibus:                       63.320   Durbin-Watson:                   2.036
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               71.911
    Skew:                           0.511   Prob(JB):                     2.43e-16
    Kurtosis:                       3.422   Cond. No.                     4.70e+04
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 4.7e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.





    Intercept                1.464993e-72
    sex[T.M]                 1.771392e-01
    vendor                   8.132960e-32
    age                      4.785698e-01
    age_sq                   1.653015e-01
    Shannon                  1.127345e-04
    phenylacetylglutamine    1.074150e-18
    dtype: float64




```python
#percent variance explained in ASV-level uniqueness by phenylacetylglutamine
results_1 = smf.ols(formula = 'min_sqrt ~ vendor+phenylacetylglutamine', data=reg_df).fit()
results_2 = smf.ols(formula = 'min_sqrt ~ vendor', data=reg_df).fit()
results_1.rsquared-results_2.rsquared
```




    0.04360624528462487




```python
#percent variance explained in genus-level uniqueness by phenylacetylglutamine
results_1 = smf.ols(formula = 'min_genus_sqrt ~ vendor+phenylacetylglutamine', data=reg_df).fit()
results_2 = smf.ols(formula = 'min_genus_sqrt ~ vendor', data=reg_df).fit()
results_1.rsquared-results_2.rsquared
```




    0.07677355261917995




```python
#generate supplementary tableS4
final_results=results_met[results_met.columns[0:4]].sort_values(by='corr_pval')
final_results['pvalue_ASV']=results_met_ASV['pvalue']
final_results['corr_pvalue_ASV']=results_met_ASV['corr_pval']
final_results['cov_adj_B_coeff_ASV']=results_met_ASV['covariate_adjusted_Beta_coeff']
```


```python
final_results[final_results['pvalue']<0.01].to_csv('metabolomics_results_supplementary_tableS4.csv')
```


```python

```
