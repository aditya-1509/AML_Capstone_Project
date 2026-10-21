# Phase 1 individual viva guide - Project 15

**Heart-Disease Risk - Domain-Structured Bayesian Network vs Neural Network**
**Team T008, Section C.** The implemented comparison is four conventional classifiers and a provisional BN. Neural-network implementation is assigned to Phase 2 and is outside this Phase 1 comparison.

Each member should understand the actual implementation and explain these answers in their own words. No authorship or contribution is inferred.

## Verified five-model results

Source: `artifacts/runs/20261020T193544Z_2a8a1491/pooled/metrics.json`, `comparison.json`, `freeze.json` and saved predictions. All non-CV columns below are **retrospective T evaluation**, 184 previously inspected records, at C-selected cutoffs.

| Model | Nested CV F1 | Cutoff | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.8231 | 0.51 | 0.8315 | 0.8257 | 0.8824 | 0.8531 | 0.9125 | 0.1227 |
| Random forest | 0.8358 | 0.38 | 0.8043 | 0.7538 | 0.9608 | 0.8448 | 0.9029 | 0.1308 |
| Histogram gradient boosting | 0.8104 | 0.37 | 0.8152 | 0.7698 | 0.9510 | 0.8509 | 0.8945 | 0.1318 |
| XGBoost | 0.8308 | 0.34 | 0.8043 | 0.7463 | 0.9804 | 0.8475 | 0.9039 | 0.1273 |
| Bayesian network | 0.8179 | 0.54 | 0.7880 | 0.8058 | 0.8137 | 0.8098 | 0.8765 | 0.1544 |

## Hospital results

D-only local model selection; the pooled winner is not imported. All five local families run before each held-out hospital is scored.

| Held-out D hospital | S_H | C_H | Evaluation N | Local selected family | Cutoff | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cleveland | 363 | 122 | 249 | Logistic regression | 0.50 | 0.7547 | 0.8656 |
| Hungary | 379 | 127 | 228 | Histogram gradient boosting | 0.28 | 0.6494 | 0.8416 |
| Switzerland | 478 | 160 | 96 | Random forest | 0.44 | 0.8712 | 0.7472 |
| VA Long Beach | 429 | 144 | 161 | Logistic regression | 0.30 | 0.8561 | 0.7079 |

**Weak transport results matter.** Switzerland's local RF has F1 0.8712 and accuracy 0.7813, below always-positive F1 0.9565 and accuracy 0.9167 on the same 96 rows (88 positive, 8 negative). VA Long Beach's local LR has F1 0.8561 versus reference 0.8582; both have accuracy 0.7516. It rejects only 2/40 negative records (specificity 0.0500; balanced accuracy 0.5167), despite AUC 0.7079. These are concerns about negative-class decisions and transfer, not proof of zero ranking discrimination. No model or threshold is changed in response.

The following complementary metrics are recalculated from the saved selected-model predictions, at each existing C_H-selected cutoff. Balanced accuracy is (sensitivity + specificity)/2 and is added only as a descriptive summary after evaluation. Confusion matrices use [[TN, FP], [FN, TP]]. All five families have identical evaluation IDs/labels within each experiment; the always-positive decisions are constructed on those exact same rows. Reference confusion matrices are [[0, N-negative], [0, N-positive]]. Their sensitivity is 1, specificity 0, balanced accuracy 0.5 and constant-probability AUC 0.5. Saved reference probabilities use S_H prevalence; always-positive decisions are explicitly fixed and are not obtained by thresholding that constant probability at 0.5.

| Hospital | Positive / negative | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cleveland | 113 / 136 | 117 | 19 | 33 | 80 |
| Hungary | 84 / 144 | 72 | 72 | 9 | 75 |
| Switzerland | 88 / 8 | 4 | 4 | 17 | 71 |
| VA Long Beach | 121 / 40 | 2 | 38 | 2 | 119 |

| Hospital | Precision | Sensitivity | Specificity | Balanced accuracy | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cleveland | 0.8081 | 0.7080 | 0.8603 | 0.7841 | 0.7912 |
| Hungary | 0.5102 | 0.8929 | 0.5000 | 0.6964 | 0.6447 |
| Switzerland | 0.9467 | 0.8068 | 0.5000 | 0.6534 | 0.7813 |
| VA Long Beach | 0.7580 | 0.9835 | 0.0500 | 0.5167 | 0.7516 |

| Hospital | Selected F1 | Always-positive F1 | Selected accuracy | Always-positive accuracy |
| --- | ---: | ---: | ---: | ---: |
| Cleveland | 0.7547 | 0.6243 | 0.7912 | 0.4538 |
| Hungary | 0.6494 | 0.5385 | 0.6447 | 0.3684 |
| Switzerland | 0.8712 | 0.9565 | 0.7813 | 0.9167 |
| VA Long Beach | 0.8561 | 0.8582 | 0.7516 | 0.7516 |

For positive prevalence p, always-positive precision is p and F1 is 2p/(1+p), so high positive-class F1 can reward a classifier that never recognizes a negative. AUC summarizes ranking over thresholds; it cannot ensure useful specificity at the chosen cutoff. Small negative counts make specificity unstable. These are selected historical cohorts with differing prevalence, missingness and potentially measurements/labels; patient linkage is unavailable. ID 304's hospital discrepancy adds a concrete grouping limitation. These secondary D-only experiments overlap pooled development and are not independent prospective external validation.

## Technical questions and answers

1. **What did Project 15 ask for, and what is complete?**

   The assigned title is Heart-Disease Risk - Domain-Structured Bayesian Network vs Neural Network, Team T008, Section C. Phase 1 implements four conventional classifiers and a provisional statistical BN. No neural-network comparison has been implemented; it belongs to Phase 2. Domain-graph acceptance still needs faculty review.

2. **What is the research question?**

   Compare the provisional BN with LR, RF, HGB and XGBoost using identical eligible rows, predictors and held-out partitions under a fixed leakage-controlled procedure. The target is recorded historical disease presence, not future events.

3. **Where did the dataset come from?**

   The CSV is unchanged since commit f739c2b. Mapping its fourteen measurement/target columns back to official UCI codes matches all 920 tuples to four processed files. ID 304 has a local Cleveland label but matches an official Hungarian row. That correspondence does not establish the original supplier, acquisition date, merge/renaming process, redistribution history or patient linkage. No data or old result was silently changed.

4. **What are the target and cohort?**

   Validate num as a nonmissing integer in {0,1,2,3,4}, then y=(num>0). The source 920 rows become 918 after removing two later duplicate records ignoring id. There are 508 positive and 410 negative labels. Collapsing the target loses severity.

5. **Do duplicate removal and source IDs guarantee patient independence?**

   No. Removing repeated rows reduces direct record leakage. IDs preserve row membership, not verified patient linkage; multiple different records could still describe the same person. This remains a limitation.

6. **Which predictors are excluded?**

   id, dataset and num never enter X. Hospital only partitions/reports site checks. ca and thal are excluded because 609 and 484 of 918 entries are missing, with site dependence. Eleven predictors remain, including exercise-test measurements, so the intended measurement stage needs faculty review.

7. **What cleaning is preserved?**

   Zero pressure and cholesterol become missing. Missing cleaned counts include trestbps 60, chol 201, fbs 90, restecg 2, thalch 55, exang 55, oldpeak 62 and slope 307. Age, sex and chest-pain type are complete. Missing optional predictors are allowed; malformed inputs and invalid/missing targets are rejected.

8. **How do conventional models handle missing inputs?**

   Numeric medians, missing indicators and standard scaling; categorical most-frequent imputation and one-hot encoding with unknown-category tolerance. All learned transformations fit inside each relevant training fold. The app also validates categories, so an invalid category is not silently accepted.

9. **What are D, S, C and T?**

   Seed 42 reconstructs D=734 and historical T=184, with 406 and 102 positives. Seed 43 splits D into S=550 and C=184. S performs selection and fitting. C selects cutoffs only. T supplies retrospective final scores after choices freeze; it was previously inspected and cannot be called newly untouched.

10. **Why nested CV rather than tuning then cross_val_predict?**

   For each family, each outer validation row is predicted by a configuration chosen using only its outer training rows. Inner three-fold search never sees that outer row. Tuning on all S first and then obtaining OOF scores would let validation labels influence hyperparameter choice. Separate C prevents this model-family selection from contaminating cutoff predictions.

11. **What exactly selects the model family?**

   Random Forest was selected by the highest unweighted mean outer-fold F1 at cutoff 0.50: 0.8358, versus XGBoost 0.8308. ROC-AUC breaks exact F1 ties, followed by fixed order BN, LR, RF, HGB, XGB. Full-precision scores determine ties, not rounded display values. The selected outer-CV score still has family-selection optimism.

12. **What exactly selects a hyperparameter candidate?**

   Highest mean three-fold F1 at probability >=0.50, then highest mean AUC, then the lexicographically smallest canonical JSON parameter object. Exact full-precision equality defines ties. After choosing the family, final three-fold searches fit each family on S only, allowing prespecified five-model evaluation.

13. **What are the seeds?**

   Historical D/T=42, S/C=43, outer CV=44, inner CV=45, final search=46, bootstrap=47, estimator randomness=42. Hospital split/outer/inner/final seeds are Cleveland 143/144/145/146, Hungary 243/244/245/246, Switzerland 343/344/345/346 and VA Long Beach 443/444/445/446. Exact memberships are saved.

14. **How are cutoffs selected, and is there a refit?**

   Each S-fitted model predicts all C rows. Search k/100 for k=0..100 by F1, using >=. Ties choose integer k closest to 50, then smaller k. RF cutoff=0.38; BN=0.54. No S+C refit occurs because it would change the probability model whose cutoff was chosen. C labels and T metrics cannot replace the selected family.

15. **Why call T retrospective?**

   The 184-row holdout had already been inspected and reported before the corrected run. Freezing choices before rescoring improves procedural isolation but cannot undo historical exposure. It is neither a newly untouched test nor an independent external cohort.

16. **Why keep Random Forest when LR has higher retrospective scores?**

   RF won the predeclared S nested-CV rule. LR retrospective F1=0.8531 and AUC=0.9125 are descriptive final results. Switching to LR based on T would use retrospective labels for family selection and undermine the protocol. RF is not universally superior.

17. **What are the selected RF parameters?**

   200 trees, max_depth=5, min_samples_leaf=2, max_features=sqrt, random_state=42. Its preprocessing and fitted trees are saved together in a Pipeline. Inference uses predict_proba and the stored 0.38 threshold, rather than Pipeline.predict at 0.50.

18. **What is the BN graph?**

   The provisional discrete DAG contains disease -> each of the 11 features, plus oldpeak -> slope. These directions specify a class-conditional statistical factorization, not medical causation. All feature nodes except the oldpeak/slope pair are mutually independent given disease; slope is independent of the other features given disease and oldpeak. The extra edge permits residual dependence between two recorded exercise-ST descriptors, but its orientation and the omitted dependencies are unvalidated assumptions. No edge has established medical or faculty approval.

19. **Write its factorization.**

   Let Y denote disease, O oldpeak and L slope. P(Y,X)=P(Y) times the product of P(X_i|Y) for all features except L, times P(L|Y,O). This defines a proper joint distribution with a root prior and normalized CPTs. The oldpeak/slope dependence makes it more than a pure naive Bayes factorization.

20. **Does disease -> age mean disease causes age?**

   No. Edges encode the statistical factorization P(features|class). Disease-to-demographic arrows have no approved causal interpretation. The ST-descriptor edge permits an assumed residual association and does not prove a physiological direction. No edge is medically validated or faculty approved.

21. **How are CPTs and smoothing estimated?**

   After training-only imputation/discretization, count each child state for each parent configuration. Estimate (N_ijk+alpha)/(N_ij+K_i*alpha), where alpha=1 per child state in this run. Every conditional row sums to one and positive smoothing supports states absent from permitted training rows. This is not a clinical prior elicited from doctors.

22. **How are numeric BN states defined?**

   Numeric medians fill training gaps, then training quantiles at 1/3 and 2/3 define bins. Duplicate cuts and cuts at extrema are removed. Intervals extend beyond observed extrema; equality goes into the higher bin. A constant variable has one state. These are statistical bins, not sourced clinical cutoffs. All-missing training columns are rejected.

23. **How does BN inference handle missing evidence?**

   It computes P(Y|observed evidence) by exact sum-product marginalization. Missing independent leaves sum to one. The oldpeak/slope branch is explicitly summed over any unobserved states, preserving its dependence. With no evidence it returns the fitted Y prior, not automatically 0.50. Training imputation remains an approximation.

24. **What about unseen or invalid values?**

   A valid unseen category uses its frozen state vocabulary and smoothed CPT. Numeric observations use the frozen unbounded bins. Unknown invalid categories, nonfinite evidence or forbidden query keys raise errors. Missing is an explicit omission, not a catch-all for malformed data.

25. **Is the comparison fair?**

   All five use identical eligible rows/features and the same S/C/T and CV folds. Learned preprocessing stays local, and all families have prespecified evaluation. But representation and search budgets differ: LR/HGB/XGB 8 configurations each, RF 6, BN only alpha=1. The exploratory results cannot isolate graph quality from discretization or tuning budget.

26. **Explain RF confusion-matrix arithmetic.**

   Retrospective T: TN=50, FP=32, FN=4, TP=98. Accuracy=148/184=0.8043; precision=98/130=0.7538; recall=98/102=0.9608; F1=196/232=0.8448. High recall comes with 32 false positives and specificity 50/82=0.6098. Balanced accuracy=(0.9608+0.6098)/2=0.7853. The F1 cutoff does not encode clinical error costs.

27. **How do ROC-AUC, F1 and Brier differ?**

   ROC-AUC measures probability ranking across cutoffs. F1 summarizes positive-class precision/recall at a chosen cutoff. Brier is mean (p-y)^2, where lower is better. Brier alone does not establish calibration. Five-bin reliability summaries are descriptive; no recalibration was fitted.

28. **What uncertainty is reported?**

   2000 paired row-bootstrap resamples of fixed evaluation predictions, seed 47, percentile 95% intervals. The same row draws apply to families. They condition on fitted models, omit training-selection and unknown patient-cluster uncertainty, and do not prove superiority. Undefined AUC resamples have saved valid/invalid counts.

29. **How do hospital experiments stay isolated?**

   Use D only. Hold out H's D rows, split the other D rows 75/25 into S_H/C_H, and repeat nested family selection, final fitting and cutoff selection. Holdout H and T cannot enter any local learned state. No pooled winner, fitted model, bins, CPTs or threshold transfers. Only fixed protocol, vocabulary, grids and deterministic rules are shared.

30. **Why do hospital winners and sample sizes differ?**

   Every site selects locally before scoring. Cleveland selects LR (249 rows), Hungary HGB (228), Switzerland RF (96), VA Long Beach LR (161). These are D-only experiments, not old full-site counts. Switzerland has only 8 negative evaluation rows. Within-source CV is row-stratified, not an additional hospital-disjoint nesting.

31. **Did XGBoost and the BN actually run?**

   Yes. The saved run completed all five families in pooled plus four hospital experiments, with 25 serialized final models and 2940 completed fits. Elapsed experiment time was 340.927 seconds, excluding environment setup/tests. XGBoost first needed a process-local path to scikit-learn's bundled OpenMP library; no system installation occurred.

32. **Which tests protect the work?**

   Final remediation reran all 38 tests: 33 core plus five app tests. They cover schema/targets/forbidden features, historical split equivalence/disjointness, selection/C/hospital isolation, graph validity, normalized CPTs, exact inference/missing evidence, serialization and app loading/threshold/display consistency. RF/BN saved probabilities match all 184 saved retrospective predictions to absolute tolerance 1e-12. Stage 6 metric reconciliation uses saved predictions only, without fitting.

33. **How do you demonstrate without retraining?**

   Open the existing report/slides and verified run 20261020T193544Z_2a8a1491. In an environment with pinned requirements run python -m streamlit run app.py. Show Data story, Model studio and a fictional profile. RF uses 0.38; the BN expander uses 0.54 and says provisional. Do not run eda.py or training to launch the app.

34. **What code and records should you point to?**

   modeling.py validates and preprocesses; train.py implements nested selection and isolated evaluation; experiment.py saves unique runs; bayesian_network.py estimates CPTs and infers posteriors; app.py loads frozen RF/BN artifacts and saved results. Run partitions, freeze files, code snapshots, environment, hashes and verification preserve reproducibility.

35. **What remains unresolved?**

   This is retrospective disease-presence classification in selected historical hospital records. It is not a clinically validated diagnosis or future-risk predictor. The supplied merged CSV is described as UCI-derived, but its exact download/transformation history, license chain and patient linkage remain unverified. The official-file comparison matches all 920 measurement/target tuples but identifies ID 304's inconsistent hospital label; it does not recover acquisition history. Row deduplication cannot establish patient independence. The BN graph remains provisional, unapproved and statistically restrictive. Search budgets differ (LR 8, RF 6, HGB 8, XGB 8, BN 1 configurations). Neural-network implementation belongs to Phase 2; its absence is not a Phase 1 ML gap. No prospective or independent external clinical validation, clinical utility analysis, or recalibration has been completed. The five confirmed names/enrolments appear below; genuine contribution evidence remains blank. Faculty must confirm eligible measurement stage, graph/independence assumptions, BN requirement, retrospective/D-only evaluation acceptance and final submission instructions.

36. **Why can an always-positive baseline beat our hospital model?**

   F1 ignores true negatives and depends on prevalence: all-positive F1=2p/(1+p). Switzerland has 88/96 positives, so the reference scores 0.9565 versus RF 0.8712. VA has 121/161 positives; reference F1 0.8582 exceeds LR 0.8561. VA LR rejects only 2/40 negatives; balanced accuracy 0.5167 exposes this weakness. AUC 0.7079 still indicates some ranking discrimination. Do not retune the threshold or replace the selected model using these inspected hospital outcomes.

37. **Why is a real BN not automatically a justified domain graph?**

   CPT estimation and marginalization prove the implemented probabilistic calculation, not medical edges or conditional independence. Friedman, Geiger and Goldszmidt support the classifier factorization; UCI explains ST measurements. Neither validates our oldpeak-to-slope dependency/direction or the 54 omitted feature connections. AZAR is a different cross-sectional ASCVD study using a different population, feature set and leave-one-out evaluation. Our graph remains a manually specified provisional candidate; it was not learned by TAN and has no faculty approval. Review every edge and omitted-pair group in docs/bn_design.md.

38. **What did the provenance investigation actually establish?**

   The local CSV has been unchanged since repository commit f739c2b. Reversing category labels and comparing 14-field row multisets against the official processed files matches all 920 measurement/target tuples, but ID 304 is labelled Cleveland locally and belongs to the Hungarian file in that reference. It is a D row. We disclosed this without changing data or old results. This is evidence of correspondence and a grouping discrepancy, not proof of the original merge process, its supplier or patient identities. A correction would require a separately documented dataset version and evaluation; T cannot become untouched again.

## Demonstration sequence

1. State the assigned title, actual Phase 1 scope, T008 and Section C.
2. Explain 920 to 918 rows, validated binary target, 11 features and unresolved merge provenance.
3. Draw D/T then S/C, nested folds and local hospital experiments. Explain why T is retrospective.
4. Show Model studio: RF selected by nested-CV F1; compare all five retrospective results without selecting by T.
5. Show a fictional profile, missing optional values, RF cutoff 0.38 and provisional BN cutoff 0.54.
6. Explain the BN factorization, smoothed counts and missing-evidence sums from the actual code.
7. Show verification files, current 38-test verification and unresolved limitations. No retraining is needed.

## Team record to fill with actual work

Newton School of Technology, Rishihood University. These five names/enrolments are confirmed; responsibilities, contribution evidence and viva practice must be filled by the team. Blank cells are intentional and cannot be inferred from a prepared guide or commit account.

| Student / enrolment | Actual work | Evidence | Viva topic practiced |
| --- | --- | --- | --- |
| Nishtha Gupta / 2401010303 | | | |
| Aditya Srivastava / 2401010039 | | | |
| Aanya Mehrotra / 2401010006 | | | |
| Shrijan Sanidhya / 2401010449 | | | |
| Abhay Mallik / 2401020002 | | | |

## Verified methodological references

- UCI Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease (variable definitions and original databases, not certification of this exact merge).
- Detrano et al. (1989), *International application of a new probability algorithm for the diagnosis of coronary artery disease*. DOI: [10.1016/0002-9149(89)90524-9](https://pubmed.ncbi.nlm.nih.gov/2756873/). Historical transport motivation; cohorts differ from this CSV.
- Breiman (2001), *Random Forests*. DOI: [10.1023/A:1010933404324](https://link.springer.com/article/10.1023/A:1010933404324).
- Chen and Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*. DOI: [10.1145/2939672.2939785](https://arxiv.org/abs/1603.02754).
- Collins et al. (2024), *TRIPOD+AI statement*. DOI: [10.1136/bmj-2023-078378](https://pubmed.ncbi.nlm.nih.gov/38626948/). Reporting guidance, not a claim of compliance or clinical validity.
- Heckerman (1995), *A Tutorial on Learning With Bayesian Networks*, MSR-TR-95-06: https://www.microsoft.com/en-us/research/publication/a-tutorial-on-learning-with-bayesian-networks/ (general BN methods, no support for our medical edges).
- scikit-learn, [Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) (methodological separation of tuning and evaluation).

- Friedman, Geiger and Goldszmidt (1997), *Bayesian Network Classifiers*, Machine Learning 29, 131-163. DOI: [10.1023/A:1007465528199](https://link.springer.com/article/10.1023/A:1007465528199); [author abstract](https://ai.stanford.edu/~nir/Abstracts/FrGG1.html). Supports classifier factorization and relaxing naive-Bayes independence; does not validate our medical edges or make our manual one-edge graph a learned TAN.
- Esmaeili, Roshanravan, Ghaffari, Mesri Alamdari and Asghari-Jafarabadi (2024), *Unraveling atherosclerotic cardiovascular disease risk factors through conditional probability analysis with Bayesian networks: insights from the AZAR cohort study*, Scientific Reports 14, 4361. DOI: [10.1038/s41598-024-55141-2](https://www.nature.com/articles/s41598-024-55141-2.pdf). Cross-sectional Iranian healthcare-provider study; different ASCVD endpoint, predictors and leave-one-out validation. Neither its graph nor its performance transfers to our angiographic label.
- Van Calster, McLernon, van Smeden, Wynants and Steyerberg (2019), *Calibration: the Achilles heel of predictive analytics*, BMC Medicine 17, 230. DOI: [10.1186/s12916-019-1466-7](https://link.springer.com/article/10.1186/s12916-019-1466-7). Distinguishes discrimination from reliable probabilities; our five descriptive bins do not establish clinical calibration.
- scikit-learn developers, [Ensembles: histogram-based gradient boosting](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting), documentation checked 21 October 2026. Explains the implemented HGB family; our shared imputation pipeline is retained even though HGB supports missing values natively.
