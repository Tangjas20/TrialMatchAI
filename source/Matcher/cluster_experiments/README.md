### Evaluation of clinical trial matching across three experiments: biomarker diversity, feature discrimination, and geographic appropriateness.

## Files
## Data

- cluster_100_patients_updated.json - 300 patients (100 EGFR, 100 KRAS, 100 ROS1)
- 10_kras_patients.json - 10 targeted KRAS patients
- geographic_patients.json - 10 location-matched patients (5 US, 5 China)
- targeted_kras_experiment.json - Ground truth for targeted experiment
- trial_locations_cache.json - Trial site locations from ClinicalTrials.gov
- kras_g12c_nct_ids.txt - 88 KRAS G12C trials
- 10K_lung_nct_ids.txt - Full lung cancer trial corpus
- clusters_metadata.csv - Trial cluster metadata

Scripts

- generate_cluster_patients.py - Generate three-biomarker patients
- generate_targeted_patients.py - Generate targeted KRAS patients
- geographic_experiment.py - Generate/evaluate geographic experiment
- cluster_evaluator.py - Evaluate three-biomarker results
- targeted_evaluator.py - Evaluate targeted KRAS results
- clinicaltrials_api.py - Fetch trial locations
- cluster_study.ipynb - Notebook for three-biomarker experiments

Results Directories (Default names)

- results/lung_results/ - Three-biomarker matching results
- results/targeted_kras/ - Targeted KRAS matching results
- results/geo_results_baseline/ - Geographic baseline (no geo reasoning)
- results/geo_results_with_prompt/ - Geographic enhanced (with geo reasoning)

### How to run the experiments

## 1. Three Biomarkers

```bash  
   # Generate patients
python generate_cluster_patients.py \
    --output-file data/cluster_100_patients_updated.json \
    --num-patients 100

# Run matching
python ablation_study.py \
    --patients-file data/cluster_100_patients_updated.json \
    --output-dir results/lung_results

# Evaluate
python cluster_evaluator.py \
    --results-folder results/lung_results \
    --output-file results/cluster_evaluation.json
```

## 2. Targeted KRAS (10 patients)

```bash  

# Generate patients
python generate_targeted_patients.py \
    --ground-truth data/targeted_kras_experiment.json \
    --output-file data/10_kras_patients.json

# Run matching
python ablation_study.py \
    --patients-file data/10_kras_patients.json \
    --output-dir results/targeted_kras

# Evaluate
python targeted_evaluator.py \
    --results-folder results/targeted_kras \
    --ground-truth data/targeted_kras_experiment.json \
    --output-file results/targeted_evaluation.json
```

## 3. Geographic Assessment

```bash  

# Fetch trial locations
python clinicaltrials_api.py \
    --nct-ids-file data/kras_g12c_nct_ids.txt \
    --output-file data/trial_locations_cache.json

# Generate patients
python geographic_experiment.py \
    --mode generate \
    --nct-ids-file data/kras_g12c_nct_ids.txt \
    --output-file data/geographic_patients.json \
    --num-patients 5

# Run baseline (no geo)
python ablation_study.py \
    --patients-file data/geographic_patients.json \
    --output-dir results/geo_results_baseline

# Run enhanced (with geo)
python ablation_study.py \
    --patients-file data/geographic_patients.json \
    --output-dir results/geo_results_with_prompt \
    --use-geographic-reasoning

# Evaluate
python geographic_experiment.py \
    --mode evaluate \
    --nct-ids-file data/kras_g12c_nct_ids.txt \
    --results-folder results/geo_results_baseline \
    --output-file data/geographic_patients.json

```
