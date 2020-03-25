```R
#Import packages
library(phyloseq)
library(vegan)
library(ggplot2)
library(pspearman)
library(OneR)
library(plyr)
```

    Loading required package: permute
    Loading required package: lattice
    This is vegan 2.5-6



```R
whole_df<-readRDS("arivale_phylo_df.rds")
```


```R
second_genome<-subset_samples(whole_df,vendor==0)
genotek<-subset_samples(whole_df,vendor==1)
genotek
second_genome
```


    phyloseq-class experiment-level object
    otu_table()   OTU Table:         [ 89022 taxa and 2539 samples ]
    sample_data() Sample Data:       [ 2539 samples by 26 sample variables ]
    tax_table()   Taxonomy Table:    [ 89022 taxa by 8 taxonomic ranks ]
    phy_tree()    Phylogenetic Tree: [ 89022 tips and 82828 internal nodes ]



    phyloseq-class experiment-level object
    otu_table()   OTU Table:         [ 89022 taxa and 1114 samples ]
    sample_data() Sample Data:       [ 1114 samples by 26 sample variables ]
    tax_table()   Taxonomy Table:    [ 89022 taxa by 8 taxonomic ranks ]
    phy_tree()    Phylogenetic Tree: [ 89022 tips and 82828 internal nodes ]



```R
#check minimum no. of read per sample for rarefaction
min(sample_sums(genotek))
min(sample_sums(second_genome))
```


13703



38813



```R
#rarefy vendor A
rarefied_genotek=rarefy_even_depth(genotek, sample.size = min(sample_sums(genotek)),
  rngseed = 111, replace = FALSE, trimOTUs = TRUE, verbose = TRUE)
```

    `set.seed(111)` was used to initialize repeatable random subsampling.
    Please record this for your records so others can reproduce.
    Try `set.seed(111); .Random.seed` for the full vector
    ...
    38005OTUs were removed because they are no longer 
    present in any sample after random subsampling
    
    ...



```R
#rarefy vendor B
rarefied_second=rarefy_even_depth(second_genome, sample.size = min(sample_sums(second_genome)),
  rngseed = 111, replace = FALSE, trimOTUs = TRUE, verbose = TRUE)
```

    `set.seed(111)` was used to initialize repeatable random subsampling.
    Please record this for your records so others can reproduce.
    Try `set.seed(111); .Random.seed` for the full vector
    ...
    54641OTUs were removed because they are no longer 
    present in any sample after random subsampling
    
    ...



```R
#Calculate alpha-diversity
richness_genotek <- estimate_richness(rarefied_genotek, measures=c("Observed", "Chao1", "ACE", "Shannon"))
richness_second <- estimate_richness(rarefied_second, measures=c("Observed", "Chao1", "ACE", "Shannon"))
```


```R
#Collapse OTUs to genera
rarefied_genotek_genus<-tax_glom(rarefied_genotek,"Genus")
```


```R
rarefied_second_genus<-tax_glom(rarefied_second,"Genus")
```


```R
set.seed(123)
#Calculate bray-curtis dissimiliarity matrix
d <- phyloseq::distance(rarefied_second_genus, method="bray", type="samples")
w<-as.matrix(d)
#Extract minimum value for each participant
w<-as.data.frame(w)
dim(w)
w[w==0]<-NA
min<-sapply(w,min,na.rm=TRUE)
sample_data(rarefied_second_genus)$min_bray<-min
```


<ol class=list-inline>
	<li>1114</li>
	<li>1114</li>
</ol>




```R
set.seed(123)
#Calculate Jaccard dissimiliarity matrix
d <- phyloseq::distance(rarefied_second_genus, method = "jaccard",binary=TRUE, type="samples")
w<-as.matrix(d)
w<-as.data.frame(w)
dim(w)
w[w==0]<-NA
min<-sapply(w,min,na.rm=TRUE)
sample_data(rarefied_second_genus)$min_jaccard<-min
```


<ol class=list-inline>
	<li>1114</li>
	<li>1114</li>
</ol>




```R
set.seed(123)
#Calculate UniFrac dissimiliarity matrix
d_uni <- phyloseq::distance(rarefied_second_genus, method = "unifrac", type="samples")
w<-as.matrix(d_uni)
#save weighted UniFrac dissimilarity matrix
#write.csv(w,'weighted_U.csv')
w<-as.data.frame(w)
dim(w)
w[w==0]<-NA
min<-sapply(w,min,na.rm=TRUE)
sample_data(rarefied_second_genus)$min_unifrac<-min
```


<ol class=list-inline>
	<li>1114</li>
	<li>1114</li>
</ol>




```R
set.seed(123)
#Calculate Weighted Unifrac dissimiliarity
d_wunifrac <- phyloseq::distance(rarefied_second_genus, method="wunifrac", type="samples")
w<-as.matrix(d_wunifrac)
w<-as.data.frame(w)
dim(w)
w[w==0]<-NA
min_wunifrac<-sapply(w,min,na.rm=TRUE)
qplot(d_wunifrac)
sample_data(rarefied_second_genus)$min_wunifrac<-min_wunifrac
```


<ol class=list-inline>
	<li>1114</li>
	<li>1114</li>
</ol>



    Don't know how to automatically pick scale for object of type dist. Defaulting to continuous.
    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.



![png](output_12_2.png)



```R
df<-as.data.frame(sample_data(rarefied_second_genus))
#Bin cohort into age groups
df$age_1[df$age<30]<-'<30'
df$age_1[df$age>=30 & df$age<40]<-'30-39'
df$age_1[df$age>=40 & df$age<50]<-'40-49'
df$age_1[df$age>=50 & df$age<60]<-'50-59'
df$age_1[df$age>=60 & df$age<70]<-'60-69'
df$age_1[df$age>=70 & df$age<80]<-'70-79'
df$age_1[df$age>=80]<-'80+'
#Add alpha diversity measures to the dataframe
df$Shannon<-richness_second$Shannon
df$Observed<-richness_second$Observed
df$Chao1<-richness_second$Chao1
#Save file for downstream analysis
#write.csv(df,'new_silva_second_genome_complete.csv')
```


```R
set.seed(123)
#Calcualte Bray-Curtis Dissimiliarity matrix for Vendor A (DNAGENOTEK)
d_bray <- phyloseq::distance(rarefied_genotek_genus, method="bray", type="samples")
#Extract minimum value for each participant
w<-as.matrix(d_bray)
w<-as.data.frame(w)
dim(w)
w[w==0]<-NA
min_bray<-sapply(w,min,na.rm=TRUE)
qplot(min_bray)
sample_data(rarefied_genotek_genus)$min_bray<-min_bray
```


<ol class=list-inline>
	<li>2539</li>
	<li>2539</li>
</ol>



    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.



![png](output_14_2.png)



```R
set.seed(123)
#Calcualte Bray-Curtis Dissimiliarity matrix for Vendor A (DNAGENOTEK)
d_bray <- phyloseq::distance(rarefied_genotek_genus, method="bray", type="samples")
#Extract minimum value for each participant
w<-as.matrix(d_bray)
w<-as.data.frame(w)
dim(w)
w[w==0]<-NA
min_bray<-sapply(w,min,na.rm=TRUE)
qplot(min_bray)
sample_data(rarefied_genotek_genus)$min_bray<-min_bray
```


<ol class=list-inline>
	<li>2539</li>
	<li>2539</li>
</ol>



    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.



![png](output_15_2.png)



```R
set.seed(123)
#Calcualte UniFrac Dissimiliarity matrix for Vendor A (DNAGENOTEK)
d_unifrac <- phyloseq::distance(rarefied_genotek_genus, method="unifrac", type="samples")
w<-as.matrix(d_unifrac)
w<-as.data.frame(w)
dim(w)
w[w==0]<-NA
min_unifrac<-sapply(w,min,na.rm=TRUE)
sample_data(rarefied_genotek_genus)$min_unifrac<-min_unifrac
```


<ol class=list-inline>
	<li>2539</li>
	<li>2539</li>
</ol>




```R
set.seed(123)
#Calcualte Weighted UniFrac Dissimiliarity matrix for Vendor A (DNAGENOTEK)
d_wunifrac <- phyloseq::distance(rarefied_genotek_genus, method="wunifrac", type="samples")
w<-as.matrix(d_wunifrac)
w<-as.data.frame(w)
dim(w)
w[w==0]<-NA
min_wunifrac<-sapply(w,min,na.rm=TRUE)
sample_data(rarefied_genotek_genus)$min_wunifrac<-min_wunifrac
```


```R
#Calcualte Jaccard Dissimiliarity matrix for Vendor A (DNAGENOTEK)
d_jaccard <- phyloseq::distance(rarefied_genotek_genus, method = "jaccard",binary=TRUE, type="samples")
w<-as.matrix(d_jaccard)
w<-as.data.frame(w)
dim(w)
w[w==0]<-NA
min<-sapply(w,min,na.rm=TRUE)
sample_data(rarefied_genotek_genus)$min_jaccard<-min
```


```R
df<-as.data.frame(sample_data(rarefied_genotek_genus))
df$age_1[df$age<30]<-'<30'
df$age_1[df$age>=30 & df$age<40]<-'30-39'
df$age_1[df$age>=40 & df$age<50]<-'40-49'
print(dim(df[df$age>=40 & df$age<50]))
df$age_1[df$age>=50 & df$age<60]<-'50-59'
df$age_1[df$age>=60 & df$age<70]<-'60-69'
df$age_1[df$age>=70 & df$age<80]<-'70-79'
df$age_1[df$age>=80]<-'80+'
df$Shannon<-richness_genotek$Shannon
df$Observed<-richness_genotek$Observed
df$Chao1<-richness_second$Chao1
#write.csv(df,'new_silva_Genotek_complete.csv')
```
