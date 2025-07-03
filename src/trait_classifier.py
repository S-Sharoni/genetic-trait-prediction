import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
f1_score, accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import pickle
import re

class TraitClassifier:
    """
    Reusable class for classification modeling of genetic/trait data.
    Handles model fitting, cross-validation, plotting, and reporting.
    parameters:
        X: DataFrame with SNP columns
        y: DataFrame with trait column
        snp_cols: list of SNP column names
        label_name: name of the trait column
        out_prefix: prefix for the output files
    functions:
        gender_model: predicts biological gender using direct allele inspection.
        export_results: exports the results to a csv file.
    """
    def __init__(self, X, y, snp_cols, label_name="Trait", out_prefix="model"):
        self.X = X
        self.y = y
        self.snp_cols = snp_cols
        self.label_name = label_name
        self.out_prefix = self._sanitize_filename(out_prefix)
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        self.class_names = list(self.label_encoder.classes_)
        self.class_weights = compute_class_weight('balanced', classes=np.unique(self.y_encoded), y=self.y_encoded)
        self.class_weight_dict = dict(enumerate(self.class_weights))
        self.models = self._get_models()
        self.results_summary = []
        self.classification_reports = []

    def _sanitize_filename(self, s):
        return re.sub(r'[^A-Za-z0-9_\-]', '_', s)

    def _get_models(self):
        """Returns the classification models to evaluate."""
        return {
            "Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight=self.class_weight_dict),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=300),
            "Decision Tree": DecisionTreeClassifier(class_weight=self.class_weight_dict)}

    def _wrap_labels(self, labels, max_words=2):
        """Nicely wraps labels for display in confusion matrices."""
        return ['\n'.join(label.split(' ', max_words - 1)) for label in labels]

    def print_class_distribution_and_weights(self):
        """
        Prints original class counts, class frequencies, and computed class weights.
        """
        # Original counts
        class_counts = pd.Series(self.y_encoded).value_counts().sort_index()
        class_frequencies = class_counts / class_counts.sum()
        # Get mapping from integer-encoded to label
        class_labels = self.label_encoder.inverse_transform(class_counts.index)
        print("\n=== Class Distribution (Original) ===")
        for label, count, freq in zip(class_labels, class_counts, class_frequencies):
            print(f"{label:<20} Count: {count:>4}   Frequency: {freq:.3f}")

        print("\n=== Class Weights (Balanced) ===")
        for i, weight in self.class_weight_dict.items():
            label = self.label_encoder.inverse_transform([i])[0]
            print(f"{label:<20} Weight: {weight:.3f}")

    def evaluate(self, cv=10, save_models=True):
        """
        Evaluates the models and plots the confusion matrices.
        parameters:
            cv: number of cross-validation folds (default: 10)
            save_models: whether to save the models (default: True)
        """
        # Print class distribution and weights before evaluation
        self.print_class_distribution_and_weights()
        skf = StratifiedKFold(n_splits=cv, shuffle=True)
        cms = []
        names = []

        # Evaluate each model
        for name, model in self.models.items():
            y_pred = cross_val_predict(model, self.X, self.y_encoded, cv=skf)
            
            # Metrics/reporting
            print(f"\n=== Evaluating {name} ===")
            report_dict = classification_report(
                self.y_encoded, y_pred, target_names=self.class_names, zero_division=0, output_dict=True)
            self.classification_reports.append(pd.DataFrame(report_dict).transpose().assign(Model=name))
            acc = accuracy_score(self.y_encoded, y_pred)
            f1 = f1_score(self.y_encoded, y_pred, average='weighted', zero_division=0)
            self.results_summary.append({'Model': name, 'CV Accuracy': acc, 'CV Weighted F1': f1})
            print(classification_report(self.y_encoded, y_pred, target_names=self.class_names, zero_division=0))
            
            # Collect for grid plotting
            cm = confusion_matrix(self.y_encoded, y_pred, labels=np.unique(self.y_encoded))
            cms.append(cm)
            names.append(name)

            # Save model if save_models is True
            if save_models:
                with open(f"{self.out_prefix}_{name.replace(' ', '_').lower()}_model.pkl", 'wb') as f:
                    pickle.dump(model, f)

        # Grid plot for confusion matrices (3 rows x 2 columns)
        n_models = len(cms)
        nrows, ncols = 2, 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
        axes = axes.flatten()
        for idx, (cm, name) in enumerate(zip(cms, names)):
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self._wrap_labels(self.class_names))
            disp.plot(ax=axes[idx], cmap=plt.cm.RdPu, values_format='d', xticks_rotation=90, colorbar=False)
            axes[idx].set_title(name, fontsize=14)
        for idx in range(len(cms), nrows * ncols):
            fig.delaxes(axes[idx])
        plt.suptitle("Confusion Matrices for All Models", fontsize=18, y=1)
        plt.tight_layout()
        plt.show()

        # Save model
        if save_models:
            with open(f"{self.out_prefix}_{name.replace(' ', '_').lower()}_model.pkl", 'wb') as f:
                pickle.dump(model, f)

        # Summary DataFrames
        self.summary_df = pd.DataFrame(self.results_summary)
        
        # Round numeric columns to 2 decimal places for display
        display_df = self.summary_df.copy()
        for col in display_df.select_dtypes(include=[np.number]).columns:
            display_df[col] = display_df[col].round(2)
        self.classification_reports_df = pd.concat(self.classification_reports)
        
        # Print results summary
        print("\n=== Model Comparison Summary ===")
        print(display_df.to_string(index=False))

    def gender_model(self, user_data, snp_data, trait_col='Gender Standardized', new_col='predicted_gender',
                 export_csv=False, folder="results", prefix=None):
        """
        Predicts biological gender using direct allele inspection.
        - user_data: DataFrame with genotype columns and trait_col
        - snp_data: DataFrame with columns 'rsid', 'effect_allele', 'other_allele'
        - trait_col: column with true gender labels
        - new_col: name for predicted gender column
        """
        allele_columns = snp_data['rsid'].tolist()
        effect_alleles = snp_data['effect_allele'].tolist()
        other_alleles = snp_data['other_allele'].tolist()

        def classify_gender(row):
            for col, effect, other in zip(allele_columns, effect_alleles, other_alleles):
                val = row.get(col, 'NN')
                if pd.isnull(val):
                    continue
                val_str = str(val)
                if val_str != 'NN' and ((isinstance(effect, str) and effect in val_str) or (isinstance(other, str) and other in val_str)):
                    return "Biological Male"
            return "Biological Female"

        user_data[new_col] = user_data.apply(classify_gender, axis=1)
        true_labels = user_data[trait_col]
        predicted_labels = user_data[new_col]

        print("\n=== Gender Prediction Results ===")
        report_str = classification_report(true_labels, predicted_labels, zero_division=0)
        print(report_str)
        acc = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        print(f"Accuracy: {acc:.3f}")
        print(f"Weighted F1 Score: {f1:.3f}")

        # Save results if requested
        if export_csv:
            os.makedirs(folder, exist_ok=True)
            if prefix is None:
                prefix = self.out_prefix if hasattr(self, 'out_prefix') else "gender"
            # Save predictions
            predictions_path = os.path.join(folder, f"{prefix}_gender_predictions.csv")
            user_data[[trait_col, new_col]].to_csv(predictions_path, index=False)
            # Save classification report as txt
            report_path = os.path.join(folder, f"{prefix}_gender_classification_report.txt")
            with open(report_path, "w") as f:
                f.write(report_str)
                f.write(f"\nAccuracy: {acc:.3f}\n")
                f.write(f"Weighted F1 Score: {f1:.3f}\n")
            print(f"Predictions saved to: {predictions_path}")
            print(f"Report saved to: {report_path}")

        # Confusion Matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        display_labels = ['Biologically Female', 'Biologically Male']
        fig, ax = plt.subplots(figsize=(6, 4))
        disp = ConfusionMatrixDisplay(conf_matrix, display_labels=display_labels)
        disp.plot(cmap="BuPu", ax=ax)
        plt.title("Confusion Matrix for Gender Prediction Model")
        plt.tight_layout()
        plt.show()

    def export_results(self, folder="results", prefix=None):
        """
        Exports summary and classification reports as CSVs to the results folder.
        parameters:
            folder: folder to export the results to (default: "results")
            prefix: prefix for the exported files (default: None)
        """
        import os
        if not hasattr(self, "summary_df") or not hasattr(self, "classification_reports_df"):
            raise AttributeError("Please run `.evaluate()` before exporting results.")
        if prefix is None:
            prefix = self.out_prefix
        os.makedirs(folder, exist_ok=True)
        summary_path = os.path.join(folder, f"{prefix}_model_comparison.csv")
        report_path = os.path.join(folder, f"{prefix}_classification_reports.csv")
        self.summary_df.to_csv(summary_path, index=False)
        self.classification_reports_df.to_csv(report_path, index=True)
        print(f"Model summary exported to: {summary_path}")
        print(f"Classification reports exported to: {report_path}")

