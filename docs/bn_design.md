# Bayesian Network design and evidence register - provisional, not approved

Project 15, Team T008, Section C. Reviewed 21 October 2026.

## Current status and scientific boundary

The implemented model in `bayesian_network.py` is a genuine discrete BN: smoothed CPT estimation and exact probabilistic inference. The authorized exploratory run `artifacts/runs/20261020T193544Z_2a8a1491/` completed all five families; the app loads frozen RF/BN artifacts. The final remediation reran 38 tests and reproduced probabilities from all 25 saved models without fitting. Earlier statements that experiments had not run describe earlier stages, not the current project.

**The domain structure is only partly established.** Retain the exact existing topology and `approved: false`. The accurate description is a **provisional manually specified class-conditional BN with one additional feature dependency**. It is not a clinically validated network, an expert-approved graph, or a learned tree-augmented naive Bayes (TAN) model. A working BN implementation does not prove that its independences fit this medical domain. No graph choice was revised using the already inspected T or hospital results.

Disease Y is parent of all eleven features; oldpeak O additionally parents slope L:

`P(Y,X) = P(Y) * product_{i != slope} P(X_i | Y) * P(slope | Y,oldpeak)`.

The nine other features are mutually independent and independent of the (O,L) pair given Y. Slope is independent of the remaining nine features given Y and O. The extra arrow permits, but does not prove, residual association between O and L. Every direction here is a probabilistic factorization choice, **not medical causation**; in particular Y does not cause age or recorded sex.

## Verified sources and their limits

- **S1 - General classifier theory:** Nir Friedman, Dan Geiger and Moises Goldszmidt, *Bayesian Network Classifiers*, Machine Learning 29, 131-163 (1997), DOI [10.1023/A:1007465528199](https://link.springer.com/article/10.1023/A:1007465528199); [author abstract](https://ai.stanford.edu/~nir/Abstracts/FrGG1.html). Describes naive-Bayes conditional independence and TAN's relaxation on UCI benchmarks against naive Bayes/C4.5. Supports a classifier construction, not our particular feature dependencies, directions or clinical use. Our manual one-edge graph does not implement their TAN structure-learning procedure.
- **S2 - General BN theory:** David Heckerman, *A Tutorial on Learning With Bayesian Networks*, Microsoft Research MSR-TR-95-06, March 1995, [primary source](https://www.microsoft.com/en-us/research/publication/a-tutorial-on-learning-with-bayesian-networks/). Covers graphical probability models, prior knowledge and learning with incomplete data. It does not approve this project's training imputation, graph or medical assumptions.
- **S3 - Variable semantics:** Andras Janosi, William Steinbrunn, Matthias Pfisterer and Robert Detrano, *Heart Disease*, UCI Machine Learning Repository (1989 citation), DOI [10.24432/C52P4X](https://archive.ics.uci.edu/dataset/45/heart+disease), additional variable documentation. Defines oldpeak as exercise-induced ST depression relative to rest and slope as peak-exercise ST slope. Shared measurement context motivates considering their association; it establishes neither dependence conditional on Y nor arrow direction. The current UCI license is CC BY 4.0; exact local merge provenance remains incomplete (README).
- **S4 - Medical BN comparison:** Parya Esmaeili, Neda Roshanravan, Samad Ghaffari, Naimeh Mesri Alamdari and Mohammad Asghari-Jafarabadi, *Unraveling atherosclerotic cardiovascular disease risk factors through conditional probability analysis with Bayesian networks: insights from the AZAR cohort study*, Scientific Reports 14, 4361 (2024), DOI [10.1038/s41598-024-55141-2](https://www.nature.com/articles/s41598-024-55141-2.pdf), Methods pp. 2-4. Cross-sectional healthcare-provider study in Iran, different ASCVD outcome and cardiometabolic predictors; combines knowledge-based and algorithmic graphs and reports leave-one-out evaluation. It illustrates the need to state how knowledge enters a graph. It does not validate our angiographic target, ST edge or omitted dependencies. Published accuracy/AUC are not directly comparable to our split or population.

Titles, authors, venues, years and the listed identifiers were checked against primary publisher/author sources. S1/S2 are methods, S3 is a data definition, S4 is a different medical study. None establishes the complete current medical DAG.

## Included-edge register (all 12 actual edges)

For rows E01-E11, the shared rationale is to allow each recorded feature's distribution to vary with the binary class, following the class-conditional construction in S1. S3 supplies the feature meanings. This is a **statistical convenience**, not verified evidence for every dependence after conditioning on other measurements. No source is being claimed to establish any disease-to-feature medical direction. Each status remains an explicit modeling assumption pending domain review.

| ID | Edge / allowed distribution | Specific rationale and source limit | Conditional-independence restriction | Status / direction |
| --- | --- | --- | --- | --- |
| E01 | Y -> age; P(age\|Y) | Model class-specific recorded age distribution; S1 construction/S3 units only | age independent of all other features given Y | Explicit modeling assumption; statistical, not causal |
| E02 | Y -> sex; P(sex\|Y) | Model class-specific recorded sex distribution; S1/S3 only | sex independent of all other features given Y | Explicit modeling assumption; statistical, not causal |
| E03 | Y -> cp; P(cp\|Y) | Allow class-specific chest-pain categories; S1/S3 only | cp independent of all other features given Y | Explicit modeling assumption; statistical, not causal |
| E04 | Y -> trestbps; P(trestbps\|Y) | Allow class-specific admission-pressure bins; S1/S3 only | trestbps independent of all other features given Y | Explicit modeling assumption; statistical, not causal |
| E05 | Y -> chol; P(chol\|Y) | Allow class-specific cholesterol bins; S1/S3 only | chol independent of all other features given Y | Explicit modeling assumption; statistical, not causal |
| E06 | Y -> fbs; P(fbs\|Y) | Allow class-specific recorded high-fasting-sugar category; S1/S3 only | fbs independent of all other features given Y | Explicit modeling assumption; statistical, not causal |
| E07 | Y -> restecg; P(restecg\|Y) | Allow class-specific resting-ECG categories; S1/S3 only | restecg independent of all other features given Y | Explicit modeling assumption; statistical, not causal |
| E08 | Y -> thalch; P(thalch\|Y) | Allow class-specific maximum-exercise-heart-rate bins; S1/S3 only | thalch independent of all other features given Y | Explicit modeling assumption; statistical, not causal |
| E09 | Y -> exang; P(exang\|Y) | Allow class-specific exercise-angina categories; S1/S3 only | exang independent of all other features given Y | Explicit modeling assumption; statistical, not causal |
| E10 | Y -> oldpeak; P(O\|Y) | Allow class-specific exercise-ST-depression bins; S1/S3 only | O independent of the nine features outside (O,L) given Y | Explicit modeling assumption; statistical, not causal |
| E11 | Y -> slope; P(L\|Y,O) | Allow slope distribution to vary with class after O; no verified conditional dependency source | L independent of the other nine features given Y,O | Explicit modeling assumption; statistical, not causal |
| E12 | oldpeak -> slope; P(L\|Y,O) | S3 places both in exercise ST measurement; no verified source establishes residual dependence or orientation | Relaxes only O/L independence given Y; excludes other residual parents of L | Explicit provisional assumption; domain sufficiency unresolved; statistical direction |

E12 is retained to preserve the evaluated candidate, not because its results justify it. Reversing the O/L ordering could represent the same class-conditional pair with unrestricted CPTs, but the actual discretization, priors and estimation must not be silently relabelled as a different tested model. No such change is made.

## Omitted-dependency register

Let A={age,sex}, M={trestbps,chol,fbs,restecg}, and E={cp,thalch,exang,oldpeak,slope}. The groups below enumerate **all 54 missing feature-to-feature connections**: 55 possible pairs among eleven features minus the one included O/L connection. An omitted arrow has no direction; neither causal direction is asserted. Each group imposes conditional independence given Y for its listed pairs. S1 explains this restriction, but does not establish its truth in this cohort. These are modeling assumptions with unresolved domain adequacy, not literature-supported independences. Pairwise listings summarize the stronger joint factorization above; they do not replace it.

| ID | Omitted pairs (either direction); count | Why review is important / source limit | Assumption and status |
| --- | --- | --- | --- |
| O01 | age with sex; 1 | Their joint distribution may depend on cohort selection. No source establishes conditional independence in these records | age independent of sex given Y; unresolved domain adequacy |
| O02 | Every A member with every M or E member; 18 | Demographic/measurement relationships are discarded after Y. S4 uses a different population/features and cannot establish independence here | Each such pair independent given Y; explicit simplifying assumption, unresolved |
| O03 | Every pair within M; 6 | Resting ECG and cardiometabolic measurements need not become unrelated given disease. S3 definitions do not establish this; S4 cannot validate these exact pairs | All M jointly independent given Y; explicit simplifying assumption, unresolved |
| O04 | Every pair within E except oldpeak/slope; 9 | Several describe the same exercise assessment. S3's measurement context motivates reviewing thalch/exang/ST/cp relationships, not adding guessed edges | Listed pairs independent given Y; explicit simplifying assumption, unresolved |
| O05 | Every M member with every E member; 20 | The model ignores residual resting/exercise-measurement links. No verified source supports this whole independence set | Each such pair independent given Y; explicit simplifying assumption, unresolved |

Hospital, ID, ca and thal are not graph nodes. Hospital is reserved for grouping; omitting it cannot remove site-related measurement/missingness differences. No patient identifier or prospective event-time node is available. The README documents a concrete ID 304 hospital-label discrepancy; the graph and data are unchanged.

## Estimation and inference actually implemented

Reuse the same eleven raw predictors and binary Y as conventional models. Within each permitted training partition, numeric medians and categorical modes fill training gaps; ties in modes use fixed vocabulary order. Numeric 1/3 and 2/3 quantiles define bins; duplicate/extreme cuts are removed. Equality enters the higher bin, exterior bins are unbounded and constant columns have one state. All-missing training columns fail. Fixed categorical states retain valid unseen categories. This is statistical discretization, not clinical cutoffs. Imputation before counting suppresses training missingness uncertainty; it is not EM or a full missing-data likelihood.

For child state k, parent configuration j and K child states:

`P(X_i=k | Pa_i=j) = (N_ijk + alpha) / (N_ij + K*alpha)`.

Alpha=1 is a pseudocount **per child state and parent configuration**, including root-class counts. CPT rows sum to one; unobserved parent configurations remain normalized. With all numeric variables having three states, the graph has 49 free parameters, calculated as sum_i (r_i-1)*product_parent r_parent. Collapsed cuts can reduce that number. The exploratory run uses one alpha setting, not a validated medical prior or an equal search budget.

For each class, multiply its prior, observed independent-feature likelihoods and the O/L branch. If O is missing but L=s observed, that branch is `sum_o P(O=o|Y)*P(L=s|Y,o)`. If L is missing, sum its states to one. Restrict O to the observed state when present. Normalize class weights; log arithmetic avoids product underflow. No evidence returns the fitted class prior. This topology-specific exact inference is rejected for an incompatible graph. Valid unseen states use smoothing; invalid categories/nonfinite evidence and forbidden query keys fail. App age requirements are stricter than the BN's partial-evidence API.

Learned imputations, bins and CPTs remain local to inner/outer/final fits. C/C_H select only thresholds from frozen models; T/held-out hospitals never choose the graph or learned state. All saved models and the run's graph/CPT snapshots remain unchanged.

## Review gate and historical record

Faculty/domain review must decide whether this provisional class-conditional structure meets the assigned domain-structure standard, accept or reject each included and omitted dependency assumption, and specify an eligible exercise-test measurement stage. Training-only sensitivity analysis could later assess restrictive assumptions under a separately approved design, but cannot be chosen to repair already viewed T/hospital scores. Any material topology change requires a new version and isolated development evaluation; old BN numbers belong to this graph only. No approval, clinical validation or causal interpretation is inferred.

Historical progression: Stage 0 specified a draft and no executable graph; Stage 2 added this opt-in provisional estimator and six BN tests. Its combined suite then had 22 passes/two pre-existing pandas-3 missing-boolean errors. Stage 3 fixed that validator minimally and passed 31 tests, including historical split equivalence. Stages 4/5 completed the explicitly authorized exploratory run and app integration; current suite has 38 passes. Thus the earlier Stage 1 split-verification gap is resolved. The original JSON protocol and immutable run snapshots preserve the historical specifications. Neural-network implementation is Phase 2, not a missing Phase 1 experiment.
