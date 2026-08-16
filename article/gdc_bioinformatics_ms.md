# A transferable multianalysis topological pipeline for transcriptomic data: biological groups versus unsupervised splitting in TCGA-LUAD (n = 601)

## Abstract

**Motivation.** Multivariate shape—not only means—can separate biological states. We previously developed the topo-stress pipeline (normalized topological divergence, random matrix spectra, HSIC, extreme-value analysis, Bayesian modelling, information-geometric physiotypes) for stress physiology (MAST, n = 371). Transferring the pipeline to transcriptomic data, a first analysis split 601 TCGA-LUAD RNA-seq samples by the sign of PC1 because phenotype was absent from the download manifest. Recovering the phenotype afterwards (NCI GDC sample sheet) lets us confront the unsupervised split with ground truth.

**Results.** All 601 samples belong to TCGA-LUAD: 542 tumours and 59 normal tissues from 518 patients (58 matched pairs). On the real biological groups the normalized topological divergence is d̄_topo = 0.600 (p < 0.005), 22-fold larger than on the PC1 split (0.027), showing that topology responds primarily to a change of the whole distribution; the signal persists (d̄_topo = 0.417) when genes are sampled at random rather than by variance. Both groups show spectra far beyond the Marchenko–Pastur bound (λ_max/λ₊ = 22.7 and 5.6); the top eigenvector is an immunoglobulin (plasma-cell) module in both. A significant nonlinear SFTPC×BPIFA1 dependence reported previously on the pooled sample disappears within groups (p = 0.09, 0.48): it was a mixture artifact. Maxima-of-expression tails are Fréchet in both groups; RL99 is higher in tumours (18.02 vs 13.14, permutation p = 0.005). A matched Bayesian model yields a tissue effect of −5.62 (94% HDI [−6.21, −4.95], R̂ = 1.002, zero divergences). Plate batch effects concentrate in PC3–PC5 (R² up to 0.36). The PC1 split agreed with tissue status only 40% (all normals in PC1+); the dominant transcriptomic axis is an intratumoural gradient, not tumour/normal. As a benchmark, DESeq2 identified 27,254 differentially expressed genes; the overlap between high-variance genes and DE genes is only 6% (Jaccard 0.06), and log2FC correlates with PC1 loadings at r = 0.12.

**Availability.** Pipeline and code at https://github.com/mordvinov-arch/topo-stress; raw data from NCI GDC (open access).

## 1 Introduction

Linear methods capture shifts of means; they are blind to the geometry of high-dimensional clouds. Topological data analysis (TDA) records connectivity structure across scales; random matrix theory (RMT) separates structured covariance from noise; the Hilbert–Schmidt independence criterion (HSIC) detects arbitrary nonlinear dependencies; extreme value theory (EVT) characterizes tails; information geometry summarizes distributions by their optimal-transport geometry. We assembled these tools into a single pipeline (topo-stress), validated it on acute stress physiology (MAST, n = 371), and transferred it unchanged to transcriptomic data [GDC; (1)].

The first transcriptomic application used a purely mathematical grouping: the sign of the first principal component (PC1±), because the download manifest contained no phenotype. This served as a transferability stress test. After the analysis, the NCI GDC sample sheet (2) was retrieved, which assigns file_id → case_id, tissue type and technical descriptors. Two questions become answerable. (i) How does the pipeline behave on the true biological groups, tumour versus normal tissue? (ii) How informative was the unsupervised PC1 split relative to biology? We recompute every module on tumour (n = 542) vs normal (n = 59), quantify batch effects, and benchmark the variance-based feature selection against DESeq2 differential expression (3). Our central methodological contributions are: (a) topology as a shape-level readout that separates whole-distribution changes 22-fold more strongly than an internal gradient; (b) demonstration of a mixture artifact in nonlinear dependence testing when heterogeneous subgroups are pooled; (c) the value of metadata recovery for transcriptomic re-analysis; (d) a matched Bayesian model that controls for patient pairing and plate batch.

## 2 Materials and Methods

### 2.1 Dataset and preprocessing

601 STAR gene-count files (GENCODE v36) of TCGA-LUAD were downloaded from NCI Genomic Data Commons (1, 2). Each file contains counts and TPM for one sample. We used tpm_unstranded, log1p-transformed, inner-joined across samples (59,427 genes × 601 samples), and retained the 500 most variable genes (HVG) as in the pipeline.

### 2.2 Metadata and batch effects

The GDC sample sheet provides tissue type (tumour 542, normal 59), tumour descriptor (540 primary, 2 recurrence), and case_id (518 unique patients; 58 patients contribute a matched tumour–normal pair). Technical metadata (plate and sequencing centre) were recovered from the aliquot barcode (TCGA-XX-XXXX-XXA-XXA-XXXX-XX) via the GDC API: 26 plates; 600/601 samples sequenced at centre "07". Batch effects were assessed by associating principal components of the z-scored expression with tissue, plate and centre (R² and 199-permutation tests).

### 2.3 Topological data analysis (TDA)

For two groups A, B (n₁, n₂ points in p = 500 dimensions) we compute normalized Betti-0 curves of the Vietoris–Rips filtration β̄₀(t; X) = β₀(t·D; X)/n̄, with n̄ = (n₁ + n₂)/2 and D = diam(A ∪ B), and the divergence

d̄_topo = ∫₀¹ |β̄₀(t; A) − β̄₀(t; B)| dt,

approximated on n_eps = 100 grid points. Significance by sample-level permutation (199 permutations; resolution ≤ 0.005). A combined metric d_comb = λ₁·(topology) + λ₂·(Euclidean) with λ = 0.5 is also reported.

### 2.4 Random matrix theory (RMT)

Eigenvalues of the gene correlation matrix (p = 500) versus the Marchenko–Pastur bound (4) λ₊ = (1 + √q)² with q = p/n, computed per group (tumour q = 0.92, λ₊ = 3.84; normal q = 8.47, λ₊ = 15.30). For the normal group q > 1 renders the sample covariance degenerate; that comparison is interpreted as auxiliary.

### 2.5 HSIC (nonlinear dependence)

HSIC with a Gaussian kernel of median bandwidth (5), permutation-tested (199 permutations): (a) dependence of the binary tissue label on leading gene expression; (b) dependence between SFTPC and BPIFA1 evaluated within each tissue separately, to test the mixture-artifact hypothesis.

### 2.6 Extreme value theory (EVT)

Per sample, the maximum log1p(TPM) over the 500 genes; GEV fit (6) per group; return level RL99; permutation test of the RL99 difference with full refit of the GEV in each permutation (199 permutations).

### 2.7 Bayesian model

SFTPC log-expression y ~ 1 + tissue + (1|patient) + (1|plate), where the patient random effect accounts for the 58 matched pairs and the plate effect for batch. NUTS sampler (7), two chains × (500 warmup + 500 draws), target_accept = 0.999. A first specification with plate as a fixed effect (26 levels) produced 97 divergences; the random-effect reparameterization and tighter target eliminated them.

### 2.8 Differential expression benchmark (DESeq2)

DESeq2 (3) via pydeseq2 (8) on raw counts: median-of-ratios size factors, shrunk dispersion, Wald test on the Tumour/Normal contrast, BH correction. The DE list was compared with variance-based HVG selection and with PC1 loadings of the full transcriptome (scipy svds): Jaccard overlap of top-500 sets and the Pearson correlation between log2FC and loadings.

### 2.9 Information geometry

Expression profiles of the first 200 genes are normalized to a simplex; pairwise Wasserstein distances (9); metric MDS; Ward clustering into three physiotypes; association with tissue by χ².

## 3 Results

### 3.1 Batch effects

PC1 explains 19.3% and PC2 10.6% of variance. Tissue dominates PC1–PC2 (R² = 0.25, 0.17), while plate contributes increasingly to PC3–PC5 (R² = 0.14, 0.22, 0.36); the sequencing centre effect is negligible. Plate is therefore controlled for in the Bayesian model.

### 3.2 TDA

d̄_topo = 0.600 (p < 0.005) for tumour vs normal versus 0.027 for the PC1 split; d_comb = 16.7. The topological readout is 22-fold stronger when the whole distribution changes.

### 3.3 RMT

Tumour: λ_max = 87.1 vs λ₊ = 3.84 (ratio 22.7); 3.4% of eigenvalues above the bound. Normal: λ_max = 85.5 vs λ₊ = 15.30 (ratio 5.6); 1.4% above. A single factor explains ≈17% of total variance in each group; structure is far beyond noise in both. The biological identity of this factor is read from the top eigenvector: it is dominated by immunoglobulin genes in both groups (Tumour — IGKC, IGKV3-20, IGKV1-5, IGKV3-11, IGKV3-15, IGHV1-18, IGLC2, IGHV5-51; Normal — IGKC, IGKV3-15, IGKV3-20, IGHV5-51, IGHV4-34, IGLV1-40, IGLC2, IGKV1-5). The dominant coherent covariance module is therefore B-cell/plasma-cell (humoral immunity) infiltration, present in tumour and normal tissue alike; this is why λ_max is large in both groups (Fig. 8).

### 3.4 HSIC: a mixture artifact

Tissue is nonlinearly associated with both SFTPC (p < 0.005) and BPIFA1 (p < 0.005). The previously reported nonlinear SFTPC×BPIFA1 dependence on the pooled sample does not reproduce within groups (tumour p = 0.09, normal p = 0.48). The pooled signal was an artifact of mixing two populations with different joint means. Nonlinear dependence tests on pooled heterogeneous samples should always be complemented by within-stratum checks.

### 3.5 EVT

Tumour: ξ = 0.236, Fréchet, RL99 = 18.02; normal: ξ = 0.261, Fréchet, RL99 = 13.14; RL99 difference +4.88, permutation p = 0.005. Extreme over-expression is a genuine property of the tumour transcriptome.

### 3.6 Bayesian model

Tissue effect on SFTPC = −5.62, 94% HDI [−6.21, −4.95], R̂ = 1.002, zero divergences. SFTPC (surfactant protein C, alveolar type II pneumocytes) is strongly down-regulated in tumours, consistent with the loss of mature alveolar differentiation in LUAD.

### 3.7 Physiotypes

Three Wasserstein physiotypes are strongly associated with tissue (χ² = 79.97, p = 4.3·10⁻¹⁸): physiotype 1 = 341 tumours / 3 normals (99% tumour), physiotype 2 = 155/36 (81%), physiotype 3 = 46/20 (70%). Their molecular characterization is presented in the companion biology paper.

### 3.8 Comparison with DESeq2

As a reference point, DESeq2 detects 27,254 genes differentially expressed between tumour and normal (padj < 0.05), 14,769 with |log2FC| > 1, with biologically correct directions (up: FAM83A, ETV4; down: PECAM1, EPAS1). The comparison with the pipeline is instructive at two levels.

First, the feature-selection logic differs fundamentally. The pipeline reduces dimensionality by variance (HVG); DESeq2 scores genes by evidence of a group difference. These two rankings are nearly orthogonal: the top-500 sets overlap at Jaccard 0.06, and PC1 loadings of the full transcriptome correlate with log2FC at only r = 0.12. Variance and differential expression are not synonyms; conclusions about which genes separate the groups must come from DE analysis, whereas geometry-based methods use HVG only as a representation of the shape of the cloud.

Second, the outputs answer different questions. DESeq2 answers "which genes change"; TDA answers "does the shape of the cloud change, beyond any linear readout". The two are complementary: DE exists without shape change, and shape change can reflect joint structure (gene co-regulation) that no single-gene test captures. Here the topological signal on real groups (d̄_topo = 0.600) is 22-fold larger than on the unsupervised PC1 split, while DESeq2 could not have been applied at all without biological labels—reinforcing the methodological point that metadata recovery, not expression data alone, is what unlocks biological interpretation.

### 3.9 Robustness to gene selection

The HVG-based signal is not an artifact of variance-based feature selection. Recomputing d̄_topo on 500 genes drawn uniformly at random from the 60,660 expressed genes gives d̄_topo = 0.417 (p < 0.005), i.e. ~70% of the HVG value; on the 500 lowest-variance genes d̄_topo ≈ 0 (p = 1.0). The tumour/normal shape difference thus pervades the transcriptome, and the concentration of the signal in variable genes reflects information content rather than selection bias. Robustness to feature choice is an important property for a diagnostic readout.

### 3.10 Clinical correlation

Where clinical annotations were available (Xena clinical matrix, stages for 591/601 samples), the transcriptomic axes were compared with pathological stage. The three Wasserstein physiotypes are stage-independent (I/II vs III/IV: χ² = 0.56, p = 0.76; all stages: χ² = 8.38, p = 0.40): they capture intrinsic tumour biology, not progression. The PC1 gradient, in contrast, is weakly but significantly associated with stage (tumours: PC1+ 176 early/61 advanced vs PC1− 247/49; χ² = 6.23, p = 0.013): the less-differentiated, surfactant-dominant axis enriches for stage III/IV. This is consistent with PC1 reflecting differentiation state rather than a pure tumour/normal contrast.

## 4 Discussion

Four methodological contributions stand out.

**Topology separates whole-distribution change.** The 22-fold increase in d̄_topo between the internal PC1 gradient and the tumour/normal comparison indicates that the normalized Betti-0 divergence is a sensitive readout of distributional (shape-level) differences, not merely of a mean shift. This echoes—in transcriptomics—what we observed in physiology, and suggests the metric is a generic "state-shape" indicator.

**Effect size.** d̄_topo = 0.600 has a concrete reading: after rescaling both clouds to the same diameter, the normalized component-count curves of tumour and normal tissue occupy disjoint parts of the scale axis over ~60% of the t-range (Fig. 2), i.e. the two point clouds have measurably different connectivity structures, not just different centres. As reference points, the same metric gives 0.027 for the internal PC1 gradient within tumours (22-fold smaller), and 0.417 even for a random 500-gene subset. We are not aware of published Betti-0 divergence values on comparable RNA-seq cohorts, which we provide here as a benchmark for future applications.

**Mixture artifacts in dependence testing.** The HSIC result illustrates a general pitfall: pooling subgroups with different joint means creates apparent nonlinear dependence that vanishes within strata (SFTPC×BPIFA1: p = 0.09/0.48). We recommend within-stratum HSIC as a standard companion to pooled tests whenever the sample is a mixture of known biological states.

**Metadata recoverability changes the analysis.** The phenotype was absent from the download manifest but fully recoverable from the GDC sample sheet and aliquot barcodes. This transformed an unsupervised exercise (PC1±) into a ground-truth comparison, and made possible both DESeq2 benchmarking and batch-effect control. Reproducibility pipelines should treat metadata as a first-class object.

**Sampling discipline in the Bayesian module.** Plate as a fixed effect with 26 levels produced 97 NUTS divergences; random effects and target_accept = 0.999 produced zero, at R̂ = 1.002. This replicates the lesson from the MAST analysis: hierarchical reparameterization is essential for convergence on high-dimensional, structured designs.

Limitations: permutation resolution is limited (p ≤ 0.005); the normal group is small (n = 59) and q > 1 in RMT; 58 of 59 normals are matched to tumours, so non-Bayesian within-group tests are not fully independent; the HVG representation may not capture the full transcriptome.

## 5 Conclusion

Replacing an unsupervised PC1 split with the true tumour/normal grouping of 601 TCGA-LUAD samples multiplies the topological divergence by 22 (d̄_topo = 0.600), demonstrates that a previously reported nonlinear gene–gene dependence was a mixture artifact, and enables a matched Bayesian model with zero divergences (effect −5.62, R̂ = 1.002). DESeq2 benchmarking shows that variance-based selection and differential expression are nearly orthogonal (Jaccard 0.06). The pipeline is fully transferable to molecular data; metadata recovery and within-stratum dependence testing are essential to avoid misinterpreting biological structure.

## Availability and implementation

Code and pipeline: https://github.com/mordvinov-arch/topo-stress. Raw data: NCI Genomic Data Commons (TCGA-LUAD, open access); manifest and sample sheet under data/gdc/; clinical matrix from UCSC Xena (TCGA.LUAD.sampleMap). Reproducible via scripts/gdc2_robustness.py (gene-selection robustness, λ_max module), scripts/gdc2_clinical.py (stage association) and the pipeline scripts in the repository.

## References

1. Grossman RL, Heath AP, Ferretti V, et al. Toward a shared vision for cancer genomic data. N Engl J Med 2016;375:1109–1112.
2. Heath AP, Ferretti V, Agrawal S, et al. The NCI Genomic Data Commons. Nat Genet 2021;53:257–262.
3. Love MI, Huber W, Anders S. Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. Genome Biol 2014;15:550.
4. Marchenko VA, Pastur LA. Distribution of eigenvalues for some sets of random matrices. Mat Sb 1967;72(114):507–536.
5. Gretton A, Fukumizu K, Teo CH, et al. A kernel statistical test of independence. Adv Neural Inf Process Syst 2007;20:585–592.
6. Coles S. An Introduction to Statistical Modeling of Extreme Values. Springer; 2001.
7. Capretto T, Piho C, Kumar R, et al. Bambi: a Python interface for Bayesian structural models. J Open Source Softw 2022;7(71):3903.
8. Muzellec G, Teleńczuk M, Cabeli V, et al. pydeseq2: a python package for bulk RNA-seq differential expression analysis. Bioinformatics 2023;39(9):btad547.
9. Villani C. Optimal Transport: Old and New. Springer; 2008.

## Tables and figures

- Fig. 1 (figures/gdc_batch_pca.png) — PCA of batch effects: tissue (PC1–PC2) and plate (PC3–PC5).
- Fig. 2 (figures/gdc2_tda.png) — normalized Betti-0 curves, tumour vs normal.
- Fig. 3 (figures/gdc2_rmt.png) — RMT spectra per group vs Marchenko–Pastur bounds.
- Fig. 4 (figures/gdc2_hsic.png) — SFTPC × BPIFA1 by tissue.
- Fig. 5 (figures/gdc2_evt.png) — GEV fits per group.
- Fig. 6 (figures/gdc2_bayesian.png) — posterior of the tissue effect.
- Fig. 7 (figures/gdc2_info_geometry.png) — physiotypes and tissue in MDS.
- Fig. 8 (figures/gdc2_lmax.png) — top-15 genes of the λ_max eigenvector per group (immunoglobulin module).
- Fig. 9 (figures/gdc2_cmp_deseq2.png) — DESeq2 log2FC vs PC1 loadings.
