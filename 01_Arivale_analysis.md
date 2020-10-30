```R
#Import packages
library(phyloseq)
library(vegan)
library(ggplot2)
library(pspearman)
library(OneR)
library(plyr)
library(dplyr)
```

    Loading required package: permute
    Loading required package: lattice
    This is vegan 2.5-6
    
    Attaching package: ‘dplyr’
    
    The following objects are masked from ‘package:plyr’:
    
        arrange, count, desc, failwith, id, mutate, rename, summarise,
        summarize
    
    The following objects are masked from ‘package:stats’:
    
        filter, lag
    
    The following objects are masked from ‘package:base’:
    
        intersect, setdiff, setequal, union
    



```R
#Import phyloseq objects containing ASV table, taxonomy table, and phylogenetic tree from the two vendors
genotek<-readRDS('genotek_10_12_20.rds')
genotek
second_genome<-readRDS('second_genome_df_10_05_20.rds')
second_genome
```


    phyloseq-class experiment-level object
    otu_table()   OTU Table:         [ 89022 taxa and 2539 samples ]
    sample_data() Sample Data:       [ 2539 samples by 26 sample variables ]
    tax_table()   Taxonomy Table:    [ 89022 taxa by 8 taxonomic ranks ]
    phy_tree()    Phylogenetic Tree: [ 89022 tips and 82828 internal nodes ]



    phyloseq-class experiment-level object
    otu_table()   OTU Table:         [ 89022 taxa and 1114 samples ]
    sample_data() Sample Data:       [ 1114 samples by 25 sample variables ]
    tax_table()   Taxonomy Table:    [ 89022 taxa by 8 taxonomic ranks ]
    phy_tree()    Phylogenetic Tree: [ 89022 tips and 82828 internal nodes ]



```R
#minimum no. of reads per vendor
min(sample_sums(genotek))
min(sample_sums(second_genome))
```


21123



25596



```R
#Rarefy genotek dataset to even depth
rarefied_genotek=rarefy_even_depth(genotek, sample.size = min(sample_sums(genotek)),
  rngseed = 111, replace = FALSE, trimOTUs = TRUE, verbose = TRUE)
```

    `set.seed(111)` was used to initialize repeatable random subsampling.
    Please record this for your records so others can reproduce.
    Try `set.seed(111); .Random.seed` for the full vector
    ...
    37799OTUs were removed because they are no longer 
    present in any sample after random subsampling
    
    ...



```R
#Rarefy second genome dataset to even depth
rarefied_second=rarefy_even_depth(second_genome, sample.size = min(sample_sums(second_genome)),
  rngseed = 111, replace = FALSE, trimOTUs = TRUE, verbose = TRUE)
```

    `set.seed(111)` was used to initialize repeatable random subsampling.
    Please record this for your records so others can reproduce.
    Try `set.seed(111); .Random.seed` for the full vector
    ...
    73798OTUs were removed because they are no longer 
    present in any sample after random subsampling
    
    ...



```R
#Calculate alpha diversity at the ASV level
richness_genotek <- estimate_richness(rarefied_genotek, measures=c("Shannon"))
richness_second <- estimate_richness(rarefied_second, measures=c("Shannon"))
```


```R
#save rariefied phyloseq object
saveRDS(rarefied_genotek, "rarefied_genotek.rds")
```


```R
#save rariefied phyloseq object
saveRDS(rarefied_second, "rarefied_second.rds")
```


```R
#Collapse ASVs into genera and save genus level phyloseq object for Vendor B
rarefied_second_genus<-tax_glom(rarefied_second,"Genus")
saveRDS(rarefied_second_genus, "rarefied_second_genus.rds")
```


```R
#Collapse ASVs into genera and save genus level phyloseq object for Vendor A
rarefied_genotek_genus<-tax_glom(rarefied_genotek,"Genus")
saveRDS(rarefied_genotek_genus, "rarefied_genotek_genus.rds")
```


```R
#Calculate alpha diversity at the genus level for both vendors
richness_genotek_genus <- estimate_richness(rarefied_genotek_genus, measures=c("Shannon"))
richness_second_genus <- estimate_richness(rarefied_second_genus, measures=c("Shannon"))
```


```R
#Calculate uniqueness using Bray-Curtis at the genus level
d_bray_g <- phyloseq::distance(rarefied_genotek_genus, method="bray", type="samples")
w<-as.matrix(d_bray_g)
w<-as.data.frame(w)
#check dimensions
dim(w)
#convert 0 (comparison of each sample with itself) to NA for downstream extraction of the minimum value
w[w==0]<-NA
#extract minimum dissimilarity value
min<-sapply(w,min,na.rm=TRUE)
#save to dataframe
sample_data(rarefied_genotek)$min_bray_genus<-min
```


<ol class=list-inline>
	<li>2539</li>
	<li>2539</li>
</ol>




```R
#test correlation with age
cor.test(sample_data(rarefied_genotek)$min_bray_genus,sample_data(rarefied_genotek)$age,method='spearman')
```

    Warning message in cor.test.default(sample_data(rarefied_genotek)$min_bray_genus, :
    “Cannot compute exact p-value with ties”


    
    	Spearman's rank correlation rho
    
    data:  sample_data(rarefied_genotek)$min_bray_genus and sample_data(rarefied_genotek)$age
    S = 2231222411, p-value < 2.2e-16
    alternative hypothesis: true rho is not equal to 0
    sample estimates:
          rho 
    0.1820889 




```R
#Calculate uniqueness using Bray-Curtis at the ASV level
d_bray <- phyloseq::distance(rarefied_genotek, method="bray", type="samples")
w<-as.matrix(d_bray)
#save dissimilarity matrix
write.csv(w,'bray_genotek_ASV.csv')
w<-as.data.frame(w)
#check dimensions
dim(w)
#extract minimum dissimilarity value
w[w==0]<-NA
min<-sapply(w,min,na.rm=TRUE)
#save to dataframe
sample_data(rarefied_genotek)$min_bray<-min
```


<ol class=list-inline>
	<li>2539</li>
	<li>2539</li>
</ol>




```R
#test ASV-level correlation with age
cor.test(sample_data(rarefied_genotek)$min_bray,sample_data(rarefied_genotek)$age,method='spearman')
```

    Warning message in cor.test.default(sample_data(rarefied_genotek)$min_bray, sample_data(rarefied_genotek)$age, :
    “Cannot compute exact p-value with ties”


    
    	Spearman's rank correlation rho
    
    data:  sample_data(rarefied_genotek)$min_bray and sample_data(rarefied_genotek)$age
    S = 2298908530, p-value = 1.578e-15
    alternative hypothesis: true rho is not equal to 0
    sample estimates:
          rho 
    0.1572769 




```R
#Save phyloseq object
saveRDS(rarefied_genotek, "rarefied_genotek.rds")
```


```R
#Calculate uniqueness using Bray-Curtis at the ASV level for Vendor B
set.seed(321)
d_bray_second<- phyloseq::distance(rarefied_second, method="bray", type="samples")
w<-as.matrix(d_bray_second)
w<-as.data.frame(w)
#save dissimilarity matrix
write.csv(w,'bray_second_ASV.csv')
#check dimensions
dim(w)
#extract minimum dissimilarity value
w[w==0]<-NA
min<-sapply(w,min,na.rm=TRUE)
#add to df
sample_data(rarefied_second)$min_bray<-min
```


<ol class=list-inline>
	<li>1114</li>
	<li>1114</li>
</ol>




```R
#test ASV-level correlation with age in vendor B
cor.test(sample_data(rarefied_second)$min_bray,sample_data(rarefied_second)$age,method='spearman')
```

    Warning message in cor.test.default(sample_data(rarefied_second)$min_bray, sample_data(rarefied_second)$age, :
    “Cannot compute exact p-value with ties”


    
    	Spearman's rank correlation rho
    
    data:  sample_data(rarefied_second)$min_bray and sample_data(rarefied_second)$age
    S = 188484518, p-value = 9.493e-10
    alternative hypothesis: true rho is not equal to 0
    sample estimates:
          rho 
    0.1819653 




```R
#Calculate uniqueness using Bray-Curtis at the genus level for Vendor B
set.seed(321)
d_bray_second_g<- phyloseq::distance(rarefied_second_genus, method="bray", type="samples")
w<-as.matrix(d_bray_second_g)
w<-as.data.frame(w)
#save dissimilarity matrix
write.csv(w,'bray_second_genus.csv')
#check dimensions
dim(w)
#extract minimum value
w[w==0]<-NA
min<-sapply(w,min,na.rm=TRUE)
#save to df
sample_data(rarefied_second)$min_bray_genus<-min
```


<ol class=list-inline>
	<li>1114</li>
	<li>1114</li>
</ol>




```R
#test genus-level correlation with age in vendor B
cor.test(sample_data(rarefied_second)$min_bray_genus,sample_data(rarefied_second)$age,method='spearman')
```

    Warning message in cor.test.default(sample_data(rarefied_second)$min_bray_genus, :
    “Cannot compute exact p-value with ties”


    
    	Spearman's rank correlation rho
    
    data:  sample_data(rarefied_second)$min_bray_genus and sample_data(rarefied_second)$age
    S = 186868375, p-value = 2.047e-10
    alternative hypothesis: true rho is not equal to 0
    sample estimates:
          rho 
    0.1889795 




```R
#Add genus and ASV-level alpha diversity to dataframe
sample_data(rarefied_second)$Shannon_genus<-richness_second_genus$Shannon
sample_data(rarefied_second)$Shannon<-richness_second$Shannon
sample_data(rarefied_genotek)$Shannon_genus<-richness_genotek_genus$Shannon
sample_data(rarefied_genotek)$Shannon<-richness_genotek$Shannon
```


```R
#generate age groups variable for Vendor B
df<-as.data.frame(sample_data(rarefied_second))
df$age_1[df$age<30]<-'<30'
df$age_1[df$age>=30 & df$age<40]<-'30-39'
df$age_1[df$age>=40 & df$age<50]<-'40-49'
df$age_1[df$age>=50 & df$age<60]<-'50-59'
df$age_1[df$age>=60 & df$age<70]<-'60-69'
df$age_1[df$age>=70 & df$age<80]<-'70-79'
df$age_1[df$age>=80]<-'80+'
#save dataframe for downstream analysis
write.csv(df,'second_genome_complete.csv')
```


```R
#generate age groups variable for Vendor A
df<-as.data.frame(sample_data(sample_data(rarefied_genotek)))
df$age_1[df$age<30]<-'<30'
df$age_1[df$age>=30 & df$age<40]<-'30-39'
df$age_1[df$age>=40 & df$age<50]<-'40-49'
df$age_1[df$age>=50 & df$age<60]<-'50-59'
df$age_1[df$age>=60 & df$age<70]<-'60-69'
df$age_1[df$age>=70 & df$age<80]<-'70-79'
df$age_1[df$age>=80]<-'80+'
#save dataframe for downstream analysis
write.csv(df,'genotek_complete.csv')
```


```R
#create relative abundance genus tables for uniqueness-genus correlations
OTU_df = as(otu_table(rarefied_genotek_genus), "matrix")
# transpose if necessary
if(taxa_are_rows(rarefied_genotek_genus)){OTU_df <- t(OTU_df)}
# Coerce to data.frame
OTU_df = as.data.frame(OTU_df)
colnames(OTU_df)<-tax_table(rarefied_genotek_genus)[,6]
OTU_df$public_client_id<-sample_data(rarefied_genotek_genus)$public_client_id
write.csv(OTU_df,'genus_genotek_table.csv')
OTU_df = as(otu_table(rarefied_genotek_genus), "matrix")
# transpose if necessary
if(taxa_are_rows(rarefied_genotek_genus)){OTU_df <- t(OTU_df)}
# Coerce to data.frame
OTU_df = as.data.frame(OTU_df)
colnames(OTU_df)<-tax_table(rarefied_genotek_genus)[,6]
OTU_df$public_client_id<-sample_data(rarefied_genotek_genus)$public_client_id
colnames(OTU_df)<-tax_table(rarefied_genotek_genus)[,2]
write.csv(OTU_df,'genus_phylum_genotek_table.csv')
```


```R
OTU_df = as(otu_table(rarefied_second_genus), "matrix")
# transpose if necessary
if(taxa_are_rows(rarefied_second_genus)){OTU_df <- t(OTU_df)}
# Coerce to data.frame
OTU_df = as.data.frame(OTU_df)
colnames(OTU_df)<-tax_table(rarefied_second_genus)[,6]
OTU_df$public_client_id<-sample_data(rarefied_second_genus)$public_client_id
write.csv(OTU_df,'genus_second_table.csv')
OTU_df = as(otu_table(rarefied_second_genus), "matrix")
# transpose if necessary
if(taxa_are_rows(rarefied_second_genus)){OTU_df <- t(OTU_df)}
# Coerce to data.frame
OTU_df = as.data.frame(OTU_df)
OTU_df$public_client_id<-sample_data(rarefied_second_genus)$public_client_id
colnames(OTU_df)<-tax_table(rarefied_second_genus)[,2]
write.csv(OTU_df,'genus_phylum_second_table.csv')
```
