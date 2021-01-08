# This notebook performs gut microbiome data preprocessing, calculation of alpha diversity and beta-diversity measures for each MrOS participant, and also performs spearman correlation shown in Fig.4D (Bray-Curtis uniqueness vs Bacteroides)


```R
#load packages
library(phyloseq)
library(vegan)
library(ggplot2)
library(OneR)
library(plyr)
library(ape)
library(pspearman)
```

    Loading required package: permute
    Loading required package: lattice
    This is vegan 2.5-6



```R
vars_new<-readRDS('MROS_Phyloseq_SILVA.rds')
```


```R
#Subset discovery and validation cohorts
discovery_df<-subset_samples(vars_new,firstcohort==1,trimOTUs = TRUE)
dim(sample_data(discovery_df))
validation_df<-subset_samples(vars_new,firstcohort==0,trimOTUs = TRUE)
dim(sample_data(validation_df))
#Get min of counts for discovery and validation
min(sample_sums(discovery_df))
min(sample_sums(validation_df))
#how many participants fall below the specified threshold (minimum number of reads in the discovery cohort was used as the threshold))
counts<-c(sample_sums(discovery_df))
counts_validation<-sample_sums(validation_df)
length(which(counts<10000))
length(which(counts_validation<10000))

```


<ol class=list-inline>
	<li>599</li>
	<li>135</li>
</ol>




<ol class=list-inline>
	<li>320</li>
	<li>135</li>
</ol>




11038



6477



0



12



```R
#Generate a rarefied OTU table without replacement
rarefied_alpha=rarefy_even_depth(vars_new, sample.size = 10000,
  rngseed = 321, replace = FALSE, trimOTUs = TRUE, verbose = TRUE)
# Extract abundance matrix from the phyloseq object
OTU_rarefied = as(otu_table(rarefied_alpha), "matrix")
# transpose if necessary
if(taxa_are_rows(rarefied_alpha)){OTU_rarefied <- t(OTU_rarefied)}
# Coerce to data.frame
OTU_rarefied_df = as.data.frame(OTU_rarefied)
print (dim(OTU_rarefied_df))
#Remove otus with no assigned phylum
rarefied_alpha = subset_taxa(rarefied_alpha, !Phylum %in% c(NA))
#Calculate alpha diversity metrics
richness <- estimate_richness(rarefied_alpha, measures=c("Observed", "Shannon"))
#check the minimum no. of samples in original phyloseq and in rarefied otu table after removing na
min(sample_sums(vars_new))
min(sample_sums(rarefied_alpha))
#check sample size post rarefaction
rarefied_alpha
```

    `set.seed(321)` was used to initialize repeatable random subsampling.
    Please record this for your records so others can reproduce.
    Try `set.seed(321); .Random.seed` for the full vector
    ...
    12 samples removedbecause they contained fewer reads than `sample.size`.
    Up to first five removed samples are: 
    
    RN0259RN0268RN0301RN0336YA1770
    ...
    1511OTUs were removed because they are no longer 
    present in any sample after random subsampling
    
    ...


    [1]   907 11142



6477



9880



    phyloseq-class experiment-level object
    otu_table()   OTU Table:         [ 10965 taxa and 907 samples ]
    sample_data() Sample Data:       [ 907 samples by 135 sample variables ]
    tax_table()   Taxonomy Table:    [ 10965 taxa by 8 taxonomic ranks ]
    phy_tree()    Phylogenetic Tree: [ 10965 tips and 10959 internal nodes ]



```R
#Collapse OTUs into genera
rarefied_genus<- tax_glom(rarefied_alpha, "Genus",NArm=F)
```

This section takes the rarefied genus table and calculates a uniqueness score for each participant. Calculating a Weighted UniFrac dissimilarity matrix is computationally intensive, so this cell will take some time to run.


```R
#Calculate minimum dissimiliarity for discovery/validation cohorts
set.seed(321)
#calculate Weighted UniFrac ASV-level
d <- phyloseq::distance(rarefied_alpha, method="wunifrac", type="samples")
#calculate Bray-Curtis ASV-level
d_bray <- phyloseq::distance(rarefied_alpha, method="bray", type="samples")
#Obtain the uniqueness score (minimum dissimiliarity)
w<-as.matrix(d)
w<-as.data.frame(w)
dim(w)
#replace 0 (each sample compared to itself will have perfect similiarity) with NA so the next line of code can omit 0 when extracting the minium value across each row
w[w==0]<-NA
min_wunifrac<-sapply(w,min,na.rm=TRUE)
#plot minimum bray-curtis
b<-as.matrix(d_bray)
b<-as.data.frame(b)
dim(b)
b[b==0]<-NA
min_bray<-sapply(b,min,na.rm=TRUE)
#add alpha and beta-diversity metrics to the Phyloseq object metadata
sample_data(rarefied_genus)$min_bray<-min_bray
sample_data(rarefied_genus)$min_wunifrac<-min_wunifrac
sample_data(rarefied_genus)$Shannon <-richness$Shannon
sample_data(rarefied_genus)$Observed <-richness$Observed
```

    Warning message in UniFrac(physeq, weighted = TRUE, ...):
    “Randomly assigning root as -- OTU_354 -- in the phylogenetic tree in the data you provided.”


<ol class=list-inline>
	<li>907</li>
	<li>907</li>
</ol>




<ol class=list-inline>
	<li>907</li>
	<li>907</li>
</ol>




```R
#Calculate minimum dissimiliarity for discovery/validation cohorts
set.seed(321)
#calculate Weighted UniFrac genus-level
d <- phyloseq::distance(rarefied_genus, method="wunifrac", type="samples")
#calculate Bray-Curtis genus-level
d_bray <- phyloseq::distance(rarefied_genus, method="bray", type="samples")
#Obtain the uniqueness score (minimum dissimiliarity)
w<-as.matrix(d)
w<-as.data.frame(w)
dim(w)
#replace 0 (each sample compared to itself will have perfect similiarity) with NA so the next line of code can omit 0 when extracting the minium value across each row
w[w==0]<-NA
min_wunifrac_genus<-sapply(w,min,na.rm=TRUE)
#plot minimum bray-curtis
b<-as.matrix(d_bray)
b<-as.data.frame(b)
dim(b)
b[b==0]<-NA
min_bray_genus<-sapply(b,min,na.rm=TRUE)
#add alpha and beta-diversity metrics to the Phyloseq object metadata
sample_data(rarefied_genus)$min_bray_genus<-min_bray_genus
sample_data(rarefied_genus)$min_wunifrac_genus<-min_wunifrac_genus
```


<ol class=list-inline>
	<li>907</li>
	<li>907</li>
</ol>




<ol class=list-inline>
	<li>907</li>
	<li>907</li>
</ol>




```R
#calculate genus-level alpha diversity and add to the measures to the sample data
richness_genus <- estimate_richness(rarefied_genus, measures=c("Observed", "Shannon"))
sample_data(rarefied_genus)$Shannon_genus<-richness_genus$Shannon
sample_data(rarefied_genus)$Observed_genus<-richness_genus$Observed
```

Below we calculate the relative abundance of Bacteroides and Prevotella for PCoA plots shown in Figure 4B-C. This cell also saves the data as a csv file for downstream statistical analysis.


```R
#calculate relative abundance of bacteroides + Prevotella
GPr = transform_sample_counts(rarefied_genus, function(x) x/10000)
OTU_df = as(otu_table(GPr), "matrix")
# transpose if necessary
if(taxa_are_rows(GPr)){OTU_df <- t(OTU_df)}
# Coerce to data.frame
OTU_df = as.data.frame(OTU_df)
bacteroides<-OTU_df$OTU_3630
prevotella <- OTU_df$OTU_3950                            
sample_data(rarefied_genus)$bacteroides<-bacteroides
sample_data(rarefied_genus)$prevotella<-prevotella
sample_data(rarefied_genus)$P_B<-prevotella+bacteroides
sample_data(rarefied_genus)$reads<-sample_sums(rarefied_alpha)   
df<-as.data.frame(sample_data(rarefied_genus))
#save df for statistical analysis in Python
write.csv(df,'df_uniqueness_new.csv')
```


```R
OTU_df = as(otu_table(rarefied_genus), "matrix")
# transpose if necessary
if(taxa_are_rows(GPr)){OTU_df <- t(OTU_df)}
# Coerce to data.frame
OTU_df = as.data.frame(OTU_df)
colnames(OTU_df)<-tax_table(rarefied_genus)[,6]
write.csv(OTU_df,'genus_df_new.csv')
colnames(OTU_df)<-tax_table(rarefied_genus)[,2]
write.csv(OTU_df,'genus_phylum_new.csv')
```

The last cell calculates Spearman correlation between Bacteroides and Bray-Curtis Uniqueness (figure 4D)


```R
#subset the discovery cohort
disc_df<-df[which(df$firstcohort==1),]
#check sample size
dim(disc_df)
#test the association between bacteroides and Bray-Curtis Uniqueness (Fig.4D)
spearman.test(c(disc_df$bacteroides),c(disc_df$min_bray_genus))
#test the association between (bacteroides+Prevotella) and Bray-Curtis Uniqueness (Figure S1A)
spearman.test(c(df$P_B),c(df$min_bray_genus))
```


<ol class=list-inline>
	<li>599</li>
	<li>147</li>
</ol>



    Warning message in spearman.test(c(disc_df$bacteroides), c(disc_df$min_bray_genus)):
    “Cannot compute exact p-values with ties”


    
    	Spearman's rank correlation rho
    
    data:  c(disc_df$bacteroides) and c(disc_df$min_bray_genus)
    S = 62878698, p-value < 2.2e-16
    alternative hypothesis: true rho is not equal to 0
    sample estimates:
           rho 
    -0.7553977 



    Warning message in spearman.test(c(df$P_B), c(df$min_bray_genus)):
    “Cannot compute exact p-values with ties”


    
    	Spearman's rank correlation rho
    
    data:  c(df$P_B) and c(df$min_bray_genus)
    S = 224124036, p-value < 2.2e-16
    alternative hypothesis: true rho is not equal to 0
    sample estimates:
           rho 
    -0.8022638 




```R
spearman.test(c(disc_df$age),c(disc_df$min_bray_genus))
spearman.test(c(disc_df$age),c(disc_df$min_bray))
```

    Warning message in spearman.test(c(disc_df$age), c(disc_df$min_bray_genus)):
    “Cannot compute exact p-values with ties”


    
    	Spearman's rank correlation rho
    
    data:  c(disc_df$age) and c(disc_df$min_bray_genus)
    S = 32040997, p-value = 0.009793
    alternative hypothesis: true rho is not equal to 0
    sample estimates:
          rho 
    0.1055048 



    Warning message in spearman.test(c(disc_df$age), c(disc_df$min_bray)):
    “Cannot compute exact p-values with ties”


    
    	Spearman's rank correlation rho
    
    data:  c(disc_df$age) and c(disc_df$min_bray)
    S = 33181364, p-value = 0.0716
    alternative hypothesis: true rho is not equal to 0
    sample estimates:
           rho 
    0.07366893 




```R
library(ape)
discovery_genus<-subset_samples(rarefied_genus,firstcohort==1,trimOTUs = TRUE)
set.seed(321)
bray<-phyloseq::distance(discovery_genus,'bray')
w<-as.matrix(bray)
d<-as.data.frame(w)
d[d==0]<-NA
disc_df$min_bray_genus_b<-sapply(d,min,na.rm=TRUE)
disc_df$min_strat<-bin(disc_df$min_bray_genus_b,nbins=600)
disc_df$bac_strat<-bin(disc_df$bacteroides,nbins=600)
ord<-pcoa(w, correction="none", rn=NULL)
ord<-ord$vectors[,c(1,2)]
my.colors <- colorRampPalette(c('green','blue'))(600)
scl <- 3
# Open a pdf file
tiff("PCoA_min.tiff",width = 4, height = 4, units = 'in', res = 300) 
plot(ord,type = "n")
points(ord, display = "sites", cex = 0.8, pch=21, col=NULL, bg=NULL)
with(disc_df, points(ord, display = "sites", col = my.colors[min_strat],
                      scaling = 1,cex = 0.7, pch = 16, bg = my.colors[min_strat]))
dev.off() 
```

    Warning message in plot.xy(xy.coords(x, y), type = type, ...):
    “"display" is not a graphical parameter”Warning message in plot.xy(xy.coords(x, y), type = type, ...):
    “"display" is not a graphical parameter”Warning message in plot.xy(xy.coords(x, y), type = type, ...):
    “"scaling" is not a graphical parameter”


<strong>png:</strong> 2



```R
ord<-pcoa(w, correction="none", rn=NULL)
ord<-ord$vectors[,c(1,2)]
my.colors <- colorRampPalette(c('green','black'))(600)
scl <- 3
# Open a pdf file
tiff("PCoA_bac.tiff",width = 4, height = 4, units = 'in', res = 300) 
plot(ord,type = "n")
points(ord, display = "sites", cex = 0.8, pch=21, col=NULL, bg=NULL)
with(disc_df, points(ord, display = "sites", col = my.colors[bac_strat],
                      scaling = 1,cex = 0.7, pch = 16, bg = my.colors[bac_strat]))
dev.off() 
```

    Warning message in plot.xy(xy.coords(x, y), type = type, ...):
    “"display" is not a graphical parameter”Warning message in plot.xy(xy.coords(x, y), type = type, ...):
    “"display" is not a graphical parameter”Warning message in plot.xy(xy.coords(x, y), type = type, ...):
    “"scaling" is not a graphical parameter”


<strong>png:</strong> 2



```R
disc_df$prev_strat<-bin(disc_df$prevotella,nbins=120)
ord<-pcoa(w, correction="none", rn=NULL)
ord<-ord$vectors[,c(1,2)]
my.colors <- colorRampPalette(c('green','black'))(120)
scl <- 3
# Open a pdf file
tiff("PCoA_prevotella.tiff",width = 4, height = 4, units = 'in', res = 300) 
plot(ord,type = "n")
points(ord, display = "sites", cex = 0.8, pch=21, col=NULL, bg=NULL)
with(disc_df, points(ord, display = "sites", col = my.colors[prev_strat],
                      scaling = 1,cex = 0.7, pch = 16, bg = my.colors[prev_strat]))
dev.off() 
```

    Warning message in plot.xy(xy.coords(x, y), type = type, ...):
    “"display" is not a graphical parameter”Warning message in plot.xy(xy.coords(x, y), type = type, ...):
    “"display" is not a graphical parameter”Warning message in plot.xy(xy.coords(x, y), type = type, ...):
    “"scaling" is not a graphical parameter”


<strong>png:</strong> 2



```R
bray_ord = ordinate(discovery_genus, method = "PCoA", distance = "bray")
pg <-plot_ordination(discovery_genus,bray_ord,color='l') + geom_point(size = 2)+theme_bw()
pg + scale_colour_gradient(low='grey',high='red')+theme(panel.grid.major=element_blank(),axis.line=element_line(colour='black'),text=element_text(size=24))
#ggsave('PCOA.png',width=7,height=5)
```

    Warning message in plot_ordination(discovery_genus, bray_ord, color = "l"):
    “Color variable was not found in the available data you provided.No color mapped.”


![png](output_19_1.png)



```R
#looking for batch effects between cohorts
bray_ord = ordinate(rarefied_alpha, method = "PCoA", distance = "bray")
pg <-plot_ordination(rarefied_alpha,bray_ord,color='firstcohort') + geom_point(size = 2)+theme_bw()
pg + scale_colour_gradient(low='grey',high='red')+theme(panel.grid.major=element_blank(),axis.line=element_line(colour='black'),text=element_text(size=24))
#ggsave('PCOA.png',width=7,height=5)
```


![png](output_20_0.png)



```R

```
