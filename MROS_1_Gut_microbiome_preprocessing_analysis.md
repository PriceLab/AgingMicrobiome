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


This section generates a phyloseq object with metadata on MrOS participants, and excludes one sample that should not be present in the dataset ('Orwoll_JF7100_JF')


```R
#Import phyloseq object containing OTU table, taxonomy table, and phylogenetic tree
vars<-readRDS('DADA_Silva_NewIds_combined.rda')
#Add metadata to object file
sample <- read.csv("mros_v4_microbiome_subjects.csv", row.names=1)
print (dim(sample))
names <-sample_names(vars)
print (length(names))
#sample <-sample[names, ]
SAM <- sample_data(sample, errorIfNULL = T)
vars_new <- merge_phyloseq(vars, SAM)
#make sure there is the right no. of participants
  #Check whether participant Orwoll_JF7100_JF was removed by adding metadata to phyloseq object,
setdiff(sample_names(vars),sample_names(vars_new))
print (dim(otu_table(vars_new)))
#Change OTU names from sequences to numbers for ease of analysis
taxa_names(vars_new) <- paste0("OTU_", seq(ntaxa(vars_new)))
# Print first 5 rows to confirm renaming
tax_table(vars_new)[0:5]
```

    [1] 946 133
    [1] 920



'JF7100'


    [1] 12855   919



<table>
<caption>A taxonomyTable: 5 × 7 of type chr</caption>
<thead>
	<tr><th></th><th scope=col>Kingdom</th><th scope=col>Phylum</th><th scope=col>Class</th><th scope=col>Order</th><th scope=col>Family</th><th scope=col>Genus</th><th scope=col>Species</th></tr>
</thead>
<tbody>
	<tr><th scope=row>OTU_1</th><td>k__Bacteria</td><td> p__Firmicutes    </td><td> c__Clostridia         </td><td> o__Clostridiales     </td><td> f__Lachnospiraceae    </td><td> g__Lachnospiraceae_UCG-004</td><td> s__ </td></tr>
	<tr><th scope=row>OTU_2</th><td>k__Bacteria</td><td> p__Firmicutes    </td><td> c__Clostridia         </td><td> o__Clostridiales     </td><td> f__Lachnospiraceae    </td><td> g__                       </td><td> s__ </td></tr>
	<tr><th scope=row>OTU_3</th><td>k__Bacteria</td><td> p__              </td><td> c__                   </td><td> o__                  </td><td> f__                   </td><td> g__                       </td><td> s__ </td></tr>
	<tr><th scope=row>OTU_4</th><td>k__Bacteria</td><td> p__Proteobacteria</td><td> c__Deltaproteobacteria</td><td> o__Desulfovibrionales</td><td> f__Desulfovibrionaceae</td><td> g__Desulfovibrio          </td><td> s__ </td></tr>
	<tr><th scope=row>OTU_5</th><td>k__Bacteria</td><td> p__Bacteroidetes </td><td> c__Bacteroidia        </td><td> o__Bacteroidales     </td><td> f__Prevotellaceae     </td><td> g__Prevotellaceae_UCG-001 </td><td> s__ </td></tr>
</tbody>
</table>



This section assesses number of reads for rarefaction. 


```R
#Subset discovery and validation cohorts
discovery_df<-subset_samples(vars_new,firstcohort==1,trimOTUs = TRUE)
validation_df<-subset_samples(vars_new,firstcohort==0,trimOTUs = TRUE)
#Get min of counts for discovery and validation
min(sample_sums(discovery_df))
min(sample_sums(validation_df))
#how many participants fall below the specified threshold (minimum number of reads in the discovery cohort was used as the threshold))
counts<-c(sample_sums(discovery_df))
counts_validation<-sample_sums(validation_df)
length(which(counts<9424))
length(which(counts_validation<9424))
```


9424



5429



0



12



```R
#Generate a rarefied OTU table without replacement
rarefied_alpha=rarefy_even_depth(vars_new, sample.size = 9424,
  rngseed = 123, replace = FALSE, trimOTUs = TRUE, verbose = TRUE)
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

    `set.seed(123)` was used to initialize repeatable random subsampling.
    Please record this for your records so others can reproduce.
    Try `set.seed(123); .Random.seed` for the full vector
    ...
    12 samples removedbecause they contained fewer reads than `sample.size`.
    Up to first five removed samples are: 
    
    RN0259RN0268RN0301RN0336YA1770
    ...
    1352OTUs were removed because they are no longer 
    present in any sample after random subsampling
    
    ...


    [1]   907 11503



5429



9424



    phyloseq-class experiment-level object
    otu_table()   OTU Table:         [ 11503 taxa and 907 samples ]
    sample_data() Sample Data:       [ 907 samples by 133 sample variables ]
    tax_table()   Taxonomy Table:    [ 11503 taxa by 7 taxonomic ranks ]
    phy_tree()    Phylogenetic Tree: [ 11503 tips and 11502 internal nodes ]



```R
#Collapse OTUs into genera
rarefied_genus<- tax_glom(rarefied_alpha, "Genus")
```

This section takes the rarefied genus table and calculates a uniqueness score for each participant. Calculating a Weighted UniFrac dissimilarity matrix is computationally intensive, so this cell will take some time to run.


```R
#Calculate minimum dissimiliarity for discovery/validation cohorts
set.seed(321)
#calculate Weighted UniFrac
d <- phyloseq::distance(rarefied_genus, method="wunifrac", type="samples")
#calculate Bray-Curtis
d_bray <- phyloseq::distance(rarefied_genus, method="bray", type="samples")
#Obtain the uniqueness score (minimum dissimiliarity)
w<-as.matrix(d)
w<-as.data.frame(w)
dim(w)
#replace 0 (each sample compared to itself will have perfect similiarity) with NA so the next line of code can omit 0 when extracting the minium value across each row
w[w==0]<-NA
min<-sapply(w,min,na.rm=TRUE)
#plot minimum bray-curtis
b<-as.matrix(d_bray)
b<-as.data.frame(b)
dim(b)
b[b==0]<-NA
min_bray<-sapply(b,min,na.rm=TRUE)
#add alpha and beta-diversity metrics to the Phyloseq object metadata
sample_data(rarefied_genus)$min_bray<-min_bray
sample_data(rarefied_genus)$min_wunifrac<-min
sample_data(rarefied_genus)$Shannon <-richness$Shannon
sample_data(rarefied_genus)$Observed <-richness$Observed
```


<ol class=list-inline>
	<li>907</li>
	<li>907</li>
</ol>




<ol class=list-inline>
	<li>907</li>
	<li>907</li>
</ol>



Below we calculate the relative abundance of Bacteroides and Prevotella for PCoA plots shown in Figure 4B-C. This cell also saves the data as a csv file for downstream statistical analysis.


```R
#calculate relative abundance of bacteroides + Prevotella
GPr = transform_sample_counts(rarefied_genus, function(x) x/9424)
OTU_df = as(otu_table(GPr), "matrix")
# transpose if necessary
if(taxa_are_rows(GPr)){OTU_df <- t(OTU_df)}
# Coerce to data.frame
OTU_df = as.data.frame(OTU_df)
bacteroides<-OTU_df$OTU_522
prevotella <- OTU_df$OTU_9136
lachnoclostridium<-OTU_df$OTU_1336
Ruminococcaceae_UBA1819<-OTU_df$OTU_4026                              
sample_data(rarefied_genus)$bacteroides<-bacteroides
sample_data(rarefied_genus)$prevotella<-prevotella
sample_data(rarefied_genus)$P_B<-prevotella+bacteroides   
df<-as.data.frame(sample_data(rarefied_genus))
#save df for statistical analysis in Python
write.csv(df,'df_uniqueness.csv')
```

The last cell calculates Spearman correlation between Bacteroides and Bray-Curtis Uniqueness (figure 4D)


```R
#subset the discovery cohort
disc_df<-df[which(df$firstcohort==1),]
#check sample size
dim(disc_df)
#test the association between bacteroides and Bray-Curtis Uniqueness (Fig.4D)
spearman.test(c(df$bacteroides),c(df$min_bray))
#test the association between (bacteroides+Prevotella) and Bray-Curtis Uniqueness (Figure S1A)
spearman.test(c(df$P_B),c(df$min_bray))
```


<ol class=list-inline>
	<li>599</li>
	<li>140</li>
</ol>



    Warning message in spearman.test(c(df$bacteroides), c(df$min_bray)):
    “Cannot compute exact p-values with ties”


    
    	Spearman's rank correlation rho
    
    data:  c(df$bacteroides) and c(df$min_bray)
    S = 215670210, p-value < 2.2e-16
    alternative hypothesis: true rho is not equal to 0
    sample estimates:
           rho 
    -0.7342834 



    Warning message in spearman.test(c(df$P_B), c(df$min_bray)):
    “Cannot compute exact p-values with ties”


    
    	Spearman's rank correlation rho
    
    data:  c(df$P_B) and c(df$min_bray)
    S = 223708290, p-value < 2.2e-16
    alternative hypothesis: true rho is not equal to 0
    sample estimates:
           rho 
    -0.7989206 


