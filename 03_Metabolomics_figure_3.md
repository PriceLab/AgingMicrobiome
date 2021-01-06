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
import warnings
%matplotlib inline
warnings.filterwarnings("ignore")
```


```python
#Import demographic information on Arivale participants
whole_df=pd.read_csv('results_merged.csv')
whole_df.set_index('public_client_id',inplace=True)
whole_df.shape
```




    (3653, 38)




```python
#Import metabolomics data
metabolomics=pd.read_csv('metabolite_median_impute.csv')
print(metabolomics.shape)
#subset identified metabolites for downstream statistical testing
mets=metabolomics[metabolomics.columns[0:653]].columns.tolist()
print(len(mets))
#correct public_client_id
ind=[]
for x in metabolomics['public_client_id']:
    ind.append('0'+str(x))
metabolomics.index=ind
#keep all identified metabolites (exclude unidentified mets)
X = metabolomics[metabolomics.columns[0:653]]
#scale and standardize
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
Xcolumns=X.columns
X = scaler.fit_transform(X)
X=pd.DataFrame(data=X,columns=Xcolumns,index=metabolomics.index)
print (X.shape)
#add covariates from whole_df
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

    (1475, 736)
    653
    (1475, 653)





    (1475, 661)




```python
#Square root transform Bray-Curtis Uniqueness
X['min_genus_sqrt']=np.sqrt(X['min_genus'])
X['min_sqrt']=np.sqrt(X['min'])
X.shape
```




    (1475, 663)




```python
#verify only participants who have blood draws within 21 days of the GM sample are considered
X=X[X['days_since_first_draw']>=-21]
print(X.shape)
X=X[X['days_since_first_draw']<=21]
#exclude outliers in the uniqueness measure
no_outliers=X[X['min_genus_sqrt']<(X['min_genus_sqrt'].mean()+3*X['min_genus_sqrt'].std())]
no_outliers=no_outliers[no_outliers['min_genus_sqrt']>(X['min_genus_sqrt'].mean()-3*X['min_genus_sqrt'].std())]
no_outliers.shape
```

    (1475, 663)





    (1459, 663)




```python
print('IQR',np.percentile(no_outliers['days_since_first_draw'],[25,75]))
no_outliers['days_since_first_draw'].median()
```

    IQR [-2.  4.]





    0.0




```python
#check distribution
no_outliers['min_genus_sqrt'].hist(bins=25)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f72370dbb70>




![png](output_6_1.png)



```python
#Perform linear regression on each metabolite individually predicting square root transformed Bray-Curtis Uniqueness,
#adjusting for covariates, GENUS LEVEL
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
#multiple hypothesis correction
results_met['corr_pval']=multi.multipletests(results_met['pvalue'], alpha=0.05, method='bonferroni', is_sorted=False,returnsorted=False)[1]
results_met['covariate_adjusted_Beta_coeff']=test_value
#verify correct number of mets were tested
print (len(results_met))
results_met.set_index('analyte',inplace=True)
print (len(results_met[results_met['corr_pval']<0.05]))
```

    (1459, 663)
    653
    7



```python
#Show significant mets
significant=results_met[results_met['corr_pval']<0.05]
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
      <td>2.534283e-19</td>
      <td>1.654887e-16</td>
      <td>0.013972</td>
    </tr>
    <tr>
      <th>p-cresol glucuronide*</th>
      <td>4.250690e-14</td>
      <td>2.775700e-11</td>
      <td>0.011396</td>
    </tr>
    <tr>
      <th>p-cresol sulfate</th>
      <td>1.776654e-08</td>
      <td>1.160155e-05</td>
      <td>0.009131</td>
    </tr>
    <tr>
      <th>6-hydroxyindole sulfate</th>
      <td>3.072041e-07</td>
      <td>2.006043e-04</td>
      <td>0.007612</td>
    </tr>
    <tr>
      <th>indoleacetate</th>
      <td>2.002463e-06</td>
      <td>1.307608e-03</td>
      <td>0.007207</td>
    </tr>
    <tr>
      <th>glycodeoxycholate 3-sulfate</th>
      <td>2.198746e-05</td>
      <td>1.435781e-02</td>
      <td>0.006225</td>
    </tr>
    <tr>
      <th>3-indoxyl sulfate</th>
      <td>6.450336e-05</td>
      <td>4.212069e-02</td>
      <td>0.006050</td>
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
print (len(results_met_ASV[results_met_ASV['corr_pval']<0.05]))
```

    (1459, 665)
    653
    6



```python
#Show significant ASV-level mets
significant_ASV=results_met_ASV[results_met_ASV['corr_pval']<0.05]
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
      <th>p-cresol glucuronide*</th>
      <td>4.746402e-10</td>
      <td>3.099401e-07</td>
      <td>0.007047</td>
    </tr>
    <tr>
      <th>phenylacetylglutamine</th>
      <td>1.295129e-09</td>
      <td>8.457192e-07</td>
      <td>0.007108</td>
    </tr>
    <tr>
      <th>indolepropionate</th>
      <td>2.720695e-07</td>
      <td>1.776614e-04</td>
      <td>-0.006092</td>
    </tr>
    <tr>
      <th>p-cresol sulfate</th>
      <td>5.044629e-07</td>
      <td>3.294143e-04</td>
      <td>0.006097</td>
    </tr>
    <tr>
      <th>indoleacetate</th>
      <td>1.890779e-05</td>
      <td>1.234679e-02</td>
      <td>0.004852</td>
    </tr>
    <tr>
      <th>catechol sulfate</th>
      <td>3.198993e-05</td>
      <td>2.088943e-02</td>
      <td>-0.004811</td>
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
reg_df=reg_df[reg_df['phenylacetylglutamine'].isnull()==False]
#generate figure
sns.set(font_scale=.5,context='poster',font='Arial',style='white')
plt.figure(figsize=[40,40], dpi=200)
#reg_df.dropna(inplace=True)
ax=sns.lmplot('phenylacetylglutamine','min_vend', reg_df,fit_reg=True,scatter_kws={"s": 20,'color':'darkred'},line_kws={'color': 'black'})
plt.ylim(0.3, 0.9)
ax.set_axis_labels('phenylacetylglutamine', 'Square Root Minimum Bray-Curtis')
print(scipy.stats.pearsonr(reg_df['phenylacetylglutamine'],reg_df['min_vend']))
scipy.stats.spearmanr(reg_df['phenylacetylglutamine'],reg_df['min_vend'])
```

    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.
    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.


    (0.20895105853629803, 7.407893008107571e-16)





    SpearmanrResult(correlation=0.2276665756432532, pvalue=1.3136071673742518e-18)




    <Figure size 8000x8000 with 0 Axes>



![png](output_11_4.png)



```python
#Plot phenylacetylglutamine vs genus Bray-Curtis Uniqueness
#adjust uniqueness for vendor
results = smf.ols(formula = 'min_genus~ vendor', data=reg_df).fit()
#take residuals
reg_df['min_vend_genus']=results.resid+reg_df['min_genus'].mean()
#generate figure
sns.set(font_scale=.5,context='poster',font='Arial',style='white')
ax=sns.lmplot('phenylacetylglutamine','min_vend_genus', reg_df,fit_reg=True,scatter_kws={"s": 20,'color':'darkred'},line_kws={'color': 'black'})
plt.ylim(0, 0.6)
plt.figure(figsize=[40,40], dpi=200)
ax.set_axis_labels('phenylacetylglutamine', 'Square Root Minimum Bray-Curtis')
print(scipy.stats.pearsonr(reg_df['phenylacetylglutamine'],reg_df['min_vend_genus']))
scipy.stats.spearmanr(reg_df['phenylacetylglutamine'],reg_df['min_vend_genus'])
```

    (0.2888682270776705, 1.944313844198161e-29)





    SpearmanrResult(correlation=0.2977891560699543, pvalue=2.9109416779815684e-31)




![png](output_12_2.png)



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
      <td>2.534283e-19</td>
      <td>1.654887e-16</td>
      <td>0.013972</td>
      <td>Amino Acid</td>
      <td>Phenylalanine and Tyrosine Metabolism</td>
      <td>18.596145</td>
      <td>Phenyl/Tyrosine</td>
    </tr>
    <tr>
      <th>p-cresol glucuronide*</th>
      <td>4.250690e-14</td>
      <td>2.775700e-11</td>
      <td>0.011396</td>
      <td>Amino Acid</td>
      <td>Phenylalanine and Tyrosine Metabolism</td>
      <td>13.371541</td>
      <td>Phenyl/Tyrosine</td>
    </tr>
    <tr>
      <th>p-cresol sulfate</th>
      <td>1.776654e-08</td>
      <td>1.160155e-05</td>
      <td>0.009131</td>
      <td>Amino Acid</td>
      <td>Phenylalanine and Tyrosine Metabolism</td>
      <td>7.750397</td>
      <td>Phenyl/Tyrosine</td>
    </tr>
    <tr>
      <th>6-hydroxyindole sulfate</th>
      <td>3.072041e-07</td>
      <td>2.006043e-04</td>
      <td>0.007612</td>
      <td>Xenobiotics</td>
      <td>Chemical</td>
      <td>6.512573</td>
      <td>other</td>
    </tr>
    <tr>
      <th>indoleacetate</th>
      <td>2.002463e-06</td>
      <td>1.307608e-03</td>
      <td>0.007207</td>
      <td>Amino Acid</td>
      <td>Tryptophan Metabolism</td>
      <td>5.698436</td>
      <td>Tryptophan</td>
    </tr>
    <tr>
      <th>glycodeoxycholate 3-sulfate</th>
      <td>2.198746e-05</td>
      <td>1.435781e-02</td>
      <td>0.006225</td>
      <td>Lipid</td>
      <td>Secondary Bile Acid Metabolism</td>
      <td>4.657825</td>
      <td>other</td>
    </tr>
    <tr>
      <th>3-indoxyl sulfate</th>
      <td>6.450336e-05</td>
      <td>4.212069e-02</td>
      <td>0.006050</td>
      <td>Amino Acid</td>
      <td>Tryptophan Metabolism</td>
      <td>4.190418</td>
      <td>Tryptophan</td>
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
    1     135
    2      15
    3      18
    4       8
    5     355
    6      23
    7      10
    8      14
    9      52
    10     23
    dtype: int64




```python
#Save results for figure generation
results_met.to_csv('mets_min_reg.csv')
```


```python
#Spearman correlations shown in Figure 3B
results=results_met[results_met['corr_pval']<0.05].sort_values(by='corr_pval',ascending=True)
spearman=pd.DataFrame()
analyte=[]
correlation=[]
for x in results.index.tolist():
    df=reg_df[reg_df[x].isnull()==False]
    df['response']=df[x]
    print(scipy.stats.spearmanr(df['response'],df['min_vend_genus'])[0])
    correlation.append(scipy.stats.spearmanr(df['response'],df['min_vend_genus'])[0])
    analyte.append(x)
spearman['analyte']=analyte
spearman['spearman']=correlation
```

    0.2977891560699543
    0.24114761815388916
    0.2703083266734484
    0.13806675289068035
    0.13019668476785132
    0.08660564042112333
    0.09877994932195625



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
      <td>0.297789</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>p-cresol glucuronide*</th>
      <td>p-cresol glucuronide*</td>
      <td>0.241148</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>p-cresol sulfate</th>
      <td>p-cresol sulfate</td>
      <td>0.270308</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>6-hydroxyindole sulfate</th>
      <td>6-hydroxyindole sulfate</td>
      <td>0.138067</td>
      <td>Xenobiotics</td>
    </tr>
    <tr>
      <th>indoleacetate</th>
      <td>indoleacetate</td>
      <td>0.130197</td>
      <td>Amino Acid</td>
    </tr>
    <tr>
      <th>glycodeoxycholate 3-sulfate</th>
      <td>glycodeoxycholate 3-sulfate</td>
      <td>0.086606</td>
      <td>Lipid</td>
    </tr>
    <tr>
      <th>3-indoxyl sulfate</th>
      <td>3-indoxyl sulfate</td>
      <td>0.098780</td>
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




    <matplotlib.axes._subplots.AxesSubplot at 0x7f722cc92ba8>



    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.
    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.



![png](output_19_2.png)



```python
#Spearman correlations shown in Figure 3C
results_ASV=results_met_ASV[results_met_ASV['corr_pval']<0.05].sort_values(by='corr_pval',ascending=True)
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

    0.20092116730853668
    0.22948518869719628
    -0.11551954171620031
    0.2203470097442695
    0.10364111547639061
    -0.05498227319371602



```python
spearman_ASV.index=spearman_ASV['analyte']
spearman_ASV['SUPER_PATHWAY']=results_met['SUPER_PATHWAY']
spearman_ASV['spearman_abs']=np.abs(spearman_ASV['spearman'])
```


```python
#Generate figure 3C
spearman_ASV.sort_values(by='spearman_abs',ascending=False,inplace=True)
plt.figure(figsize=[10,15], dpi=200)
sns.set(font_scale=1.75,context='poster',font='Arial',style='white')
colors = {'Amino Acid': 'darkred', 'Xenobiotics': 'darkblue','Lipid':'gold','Cofactors and Vitamins':'orange',"Carbohydrate":"grey"}
plt.ylim(-0.15,0.30)
spearman_ASV['spearman'].plot(kind='bar', color=[colors[i] for i in spearman_ASV['SUPER_PATHWAY']],edgecolor='k')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f722cca2160>




![png](output_22_1.png)



```python
#covariate adjusted beta-coefficient for ASV-level uniqueness by phenylacetylglutamine
results = smf.ols(formula = 'min_sqrt ~ vendor+age_sq+age*sex+BMI+Shannon+phenylacetylglutamine', data=reg_df).fit()
print(results.params)
results.pvalues
```

    Intercept                0.709822
    sex[T.M]                 0.010794
    vendor                   0.040454
    age_sq                   0.000013
    age                     -0.000608
    age:sex[T.M]            -0.000429
    BMI                     -0.000020
    Shannon                  0.000967
    phenylacetylglutamine    0.007108
    dtype: float64





    Intercept                1.057895e-201
    sex[T.M]                  2.642136e-01
    vendor                    2.910635e-57
    age_sq                    4.607030e-02
    age                       3.468930e-01
    age:sex[T.M]              2.651217e-02
    BMI                       9.108457e-01
    Shannon                   7.117949e-01
    phenylacetylglutamine     1.295129e-09
    dtype: float64




```python
#covariate adjusted beta-coefficient for phenylacetylglutamine with genus-level uniqueness as the dependent variable
results = smf.ols(formula = 'min_genus_sqrt ~ vendor+age_sq+age*sex+BMI+Shannon+phenylacetylglutamine', data=reg_df).fit()
print(results.params)
results.pvalues
```

    Intercept                0.453642
    sex[T.M]                 0.010287
    vendor                  -0.036347
    age_sq                   0.000012
    age                     -0.000450
    age:sex[T.M]            -0.000273
    BMI                      0.000144
    Shannon                  0.012046
    phenylacetylglutamine    0.013972
    dtype: float64





    Intercept                4.630727e-62
    sex[T.M]                 4.190009e-01
    vendor                   7.750063e-29
    age_sq                   1.748434e-01
    age                      5.970637e-01
    age:sex[T.M]             2.831038e-01
    BMI                      5.431669e-01
    Shannon                  4.880612e-04
    phenylacetylglutamine    2.534283e-19
    dtype: float64




```python
#percent variance explained in ASV-level uniqueness by phenylacetylglutamine
results_1 = smf.ols(formula = 'min_sqrt ~ vendor+phenylacetylglutamine', data=reg_df).fit()
results_2 = smf.ols(formula = 'min_sqrt ~ vendor', data=reg_df).fit()
results_1.rsquared-results_2.rsquared
```




    0.036333408348402485




```python
#percent variance explained in genus-level uniqueness by phenylacetylglutamine
results_1 = smf.ols(formula = 'min_genus_sqrt ~ vendor+phenylacetylglutamine', data=reg_df).fit()
results_2 = smf.ols(formula = 'min_genus_sqrt ~ vendor', data=reg_df).fit()
results_1.rsquared-results_2.rsquared
```




    0.07667814736077783




```python
#generate supplementary tableS4
final_results=results_met[results_met.columns[0:4]].sort_values(by='corr_pval')
final_results['pvalue_ASV']=results_met_ASV['pvalue']
final_results['corr_pvalue_ASV']=results_met_ASV['corr_pval']
final_results['cov_adj_B_coeff_ASV']=results_met_ASV['covariate_adjusted_Beta_coeff']
```


```python
#save table
final_results[final_results['pvalue']<0.01].to_csv('metabolomics_results_supplementary_tableS4.csv')
```
