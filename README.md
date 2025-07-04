# Genetic Trait Prediction from SNP Data

This repository presents a modular machine learning framework developed as part of my Master's thesis in Data Science, aimed at predicting human traits from SNP genotype data. The framework is designed to be **reusable, interpretable, and adaptable** — it can be extended based on application needs or as research evolves and new genetic associations are discovered.

---

## 🧬 Project Overview

This project uses SNP (Single Nucleotide Polymorphism) data to predict phenotypic traits such as **blood type**, **eye color**, **ancestry**, and **biological sex**. These traits serve as examples of how the framework can be applied to diverse phenotype classification tasks.

It combines:
- Literature-curated SNPs
- Allele-based genotype encoding
- Multiple supervised ML classifiers
- Cross-validation-based evaluation

---

## 🧱 Structure
genetic-trait-prediction/
├── src/  # Core processing and modeling classes
│├── data_processor.py  # Cleans and standardizes genotype input data
│├── genotype_processor.py  # Trait standardization & grouping, SNP encoding, visualizations
│└── trait_classifier.py  # Trains/evaluates models, visualizes via confusion matrices, and outputs classificaton reports
├── notebooks/  # Trait-specific workflows
│├── blood_type.ipynb
│├── eye_color.ipynb
│├── ancestry.ipynb
│└── gender.ipynb
├── models/ # Trained models (.pkl) for reuse
│└── *.pkl
├── results/ # Evaluation metrics and classification outputs
│├── *_model_comparison.csv
│└── figures/ # Confusion matrix plots (.png)
├── requirements.txt  # Required Python libraries
└── README.md  # This file

---

## ⏩ Workflow

Each notebook walks through the following steps:

1. **Load and preprocess SNP data**
   - Read SNP genotype CSV and a corresponding SNP metadata file
   - Removes duplicate samples and filters low-quality SNPs
2. **Standardize genotypes** (using strand-corrected mappings and allele normalization)
   - Convert reverse strand alleles
   - Enforce consistent allele ordering
   - Handle edge cases (e.g., `rs8176719`)
3. **Standardize and group phenotype labels** 
   - Normalize redundant trait labels (e.g., "light brown" / "dark brown" → "brown")
   - Groups similar phenotypes into broader classes
4. **Impute missing genotypes**
   - Use mode imputation per trait group for 'NN' or missing values (optional)
5. **Encode genotypes**
   - Converts genotype strings (e.g., 'AA', 'AG') into additive dosage scores using effect/other allele mappings
6. **Train classification models**
   - Evaluate five classifiers: Logistic Regression, Random Forest, Gradient Boosting, MLP, Decision Tree
   - Use stratified 10-fold cross-validation
   - Handle class imbalance with computed weights
7. **Evaluate and visualize**
   - Save classification reports
   - Saves trained model (.pkl) to the models/ folder for reuse.
8. **Visualize results**
    - Using confusion matrices.

### 📓 Available Notebooks
- 'blood_type.ipynb'
- 'eye_color.ipynb'
- 'ancestry.ipynb'
- 'gender.ipynb'

> ❗ Due to privacy and ethical concerns, raw SNP data has been excluded from the repository.  
> However, all **trained models**, **classification reports**, and **visualizations** are available for reuse and inspection.

---

## 📋 Results

Each trait-specific model was evaluated using:
- Weighted F1 Score  
- Accuracy  
- Class distribution analysis  
- Confusion matrices

Results are saved in:
- 'results/*_model_comparison.csv' – Model performance summary
- 'results/figures/*.png' – Confusion matrices
- 'models/*.pkl' – Pickled model files for reuse

---

## 👩‍💻 Author
**Sapir S. Sharoni**  
M.Sc. in Data Science  
[LinkedIn](https://www.linkedin.com/in/sapir-sharoni-5896b2343/)
