```R
library(ggplot2)
library(OneR)
library(dplyr)
#devtools::install_github("bryandmartin/corncob")
library(corncob)
library(phyloseq)
library(data.table)
```


```R
vars_new<-readRDS('MROS_Phyloseq_SILVA.rds')
```


```R
#Exclude individuals excluded previously due to low number of reads
sample_data(vars_new)$reads<-sample_sums(vars_new)
vars_new<-subset_samples(vars_new,reads>=10000,trimOTUs = TRUE)
#check if sample size is correct
vars_new
```


```R
#import dataframe with health stratifications from notebook #2
df<-read.csv('demographics.csv')
dim(df)
sample_data(vars_new)$comp_healthy<-df$comp_healthy
#Exclude non-community dwelling individuals (nursing homes, assisted living, hospitalization)
community_dwellers=subset_samples(vars_new, giliveh!=1)
community_dwellers=subset_samples(community_dwellers, giliven!=1)
community_dwellers=subset_samples(community_dwellers, mhhsp!=1)
#check sample size
community_dwellers
```


```R
data.Genus <- tax_glom(community_dwellers, "Genus")
healthy<-subset_samples(data.Genus, comp_healthy==1)
unhealthy<-subset_samples(data.Genus, comp_healthy==0)
```


```R
healthy
unhealthy
```


```R
#filter OTUs present in less than 10% of samples before corncob analysis
prevalenceThreshold = 0.1*nsamples(healthy)
print (prevalenceThreshold)
prevdf = apply(X = otu_table(healthy),
               MARGIN = ifelse(taxa_are_rows(healthy), yes = 1, no = 2),
               FUN = function(x){sum(x > 0)})
# Add taxonomy and total read counts to this data.frame
prevdf = data.frame(Prevalence = prevdf,
                    TotalAbundance = taxa_sums(healthy))
keepTaxa = rownames(prevdf)[(prevdf$Prevalence >= prevalenceThreshold)]
#filter
healthy= prune_taxa(keepTaxa, healthy)
```


```R
#filter OTUs present in less than 10% of samples before corncob analysis
prevalenceThreshold = 0.1*nsamples(unhealthy)
print (prevalenceThreshold)
prevdf = apply(X = otu_table(unhealthy),
               MARGIN = ifelse(taxa_are_rows(unhealthy), yes = 1, no = 2),
               FUN = function(x){sum(x > 0)})
# Add taxonomy and total read counts to this data.frame
prevdf = data.frame(Prevalence = prevdf,
                    TotalAbundance = taxa_sums(unhealthy))
keepTaxa = rownames(prevdf)[(prevdf$Prevalence >= prevalenceThreshold)]
#filter
unhealthy= prune_taxa(keepTaxa, unhealthy)
```


```R
#Run beta-binomial models using corncob for each genus
set.seed(111)
r_healthy<-differentialTest(formula=~hwbmi+firstcohort+age, phi.formula=~hwbmi+firstcohort+age, formula_null=~hwbmi+firstcohort, phi.formula_null=~hwbmi+firstcohort+age,data=healthy, link = "logit", phi.link = "logit", test='LRT', boot=FALSE, filter_discriminant = TRUE,fdr='bonferroni', fdr_cutoff = 0.1)
#evaluate results
```


```R
#Print significant models after multiple hypothesis correction
r_healthy$significant_taxa
otu_to_taxonomy(OTU=r_healthy$significant_taxa,data=healthy)
```


```R
#Extract p-values, coefficients, and standard error
co <- sprintf("mu.%s", 'age')
dt <- data.table(feature = names(r_healthy$p),
                         coef = sapply(r_healthy$all_models, function(m) {
                             if (length(m) == 1) return(NA)
                             return(m$coefficients[co, 1])
                         }),
                         se = sapply(r_healthy$all_models, function(m) {
                             if (length(m) == 1) return(NA)
                             return(m$coefficients[co, 2])
                         }),
                         pvalue = r_healthy$p,
                         padj = r_healthy$p_fdr)
return(dt[!is.na(coef)])
#write a csv file with the results
#df.write.csv('differential_abundance_healthy.csv')
dt
```


```R
dt[order(dt$padj),]
```


```R
set.seed(123)
r_unhealthy<-differentialTest(formula=~hwbmi+firstcohort+age, phi.formula=~hwbmi+firstcohort+age, formula_null=~hwbmi+firstcohort, phi.formula_null=~hwbmi+firstcohort+age,data=unhealthy, link = "logit", phi.link = "logit", test='LRT', boot=FALSE, filter_discriminant = TRUE,fdr='bonferroni', fdr_cutoff = 0.1)
#evaluate results
```


```R
r_unhealthy$significant_taxa
otu_to_taxonomy(OTU=r_unhealthy$significant_taxa,data=unhealthy)
```


```R
#Extract p-values, coefficients, and standard error
co <- sprintf("mu.%s", 'age')
dt_unhealthy <- data.table(feature = names(r_unhealthy$p),
                         coef = sapply(r_unhealthy$all_models, function(m) {
                             if (length(m) == 1) return(NA)
                             return(m$coefficients[co, 1])
                         }),
                         se = sapply(r_unhealthy$all_models, function(m) {
                             if (length(m) == 1) return(NA)
                             return(m$coefficients[co, 2])
                         }),
                         pvalue = r_unhealthy$p,
                         padj = r_unhealthy$p_fdr)
return(dt_unhealthy[!is.na(coef)])
#write a csv file with the results
#df.write.csv('differential_abundance_healthy.csv')
```


```R
dt_unhealthy[order(dt_unhealthy$padj),]
```


```R

```
