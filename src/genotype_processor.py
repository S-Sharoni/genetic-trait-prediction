import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

class GenotypeProcessor:
    """
    Class for processing genotype data. 
    Attributes:
        df: DataFrame with trait and SNP columns
        trait_col: column name for trait
        snp_cols: list of SNP column names

    Functions:
        filter_by_min_samples: Keeps only traits with at least min_samples.
        impute_missing_genotypes: Imputes missing genotypes per (trait, SNP) using mode.
        standardize_traits: Maps raw trait names to standardized labels for consistent analysis.
        group_traits: Groups traits based on predefined groupings in group_mapping.
        print_unique_traits: Prints unique values in the given column.
        encode_snps: Encodes SNP genotypes as additive dosage.
        plot_pca_mds: Plots PCA and MDS for trait means of encoded features.
        plot_dendrograms: Plots hierarchical clustering dendrograms (Euclidean & Manhattan) for group means of encoded features.
    """
    def __init__(self, df, trait_col, snp_cols):
        """
        parameters:
            df: DataFrame with trait and SNP columns
            trait_col: column name for trait
            snp_cols: list of SNP column names
        """
        self.df = df.copy()
        self.trait_col = trait_col
        self.snp_cols = snp_cols

    def filter_by_min_samples(self, min_samples=5):
        """
        Keeps only traits with at least min_samples.
        parameters:
            min_samples: minimum number of samples for a trait to be kept
        returns:
            self
        """
        trait_counts_before = self.df[self.trait_col].value_counts()
        n_before = len(trait_counts_before)
        
        # Filtering
        valid_traits = trait_counts_before[trait_counts_before >= min_samples].index
        self.df = self.df[self.df[self.trait_col].isin(valid_traits)].copy()
        
        trait_counts_after = self.df[self.trait_col].value_counts()
        n_after = len(trait_counts_after)
        
        # Print summary
        print(f"\n--- Trait Counts (Bottom 10, After Filtering) ---")
        print(trait_counts_after.tail(10))
        print('-----------------------')
        print(f'Number of traits before filtering: {n_before}')
        print(f'Number of traits after filtering:  {n_after}\n')
        return self

    def impute_missing_genotypes(self, missing_val='NN'):
        """
        Impute missing genotypes per (trait, SNP) using mode.
        parameters:
            missing_val: value to impute for missing genotypes
        returns:
            self
        """
        modes = self.df.loc[self.df[self.snp_cols].ne(missing_val).any(axis=1)] \
            .groupby(self.trait_col)[self.snp_cols] \
            .agg(lambda x: x[x != missing_val].mode().iloc[0] if not x[x != missing_val].mode().empty else missing_val)

        def fill_row(row):
            if row[self.trait_col] in modes.index:
                for snp in self.snp_cols:
                    if row[snp] == missing_val:
                        row[snp] = modes.loc[row[self.trait_col], snp]
            return row

        self.df = self.df.apply(fill_row, axis=1)
        return self

    def standardize_traits(self, trait_mapping, col, new_col, df=None):
        """
        Standardizes trait values based on a mapping dictionary.
        parameters:
            trait_mapping: dictionary mapping trait values to new values
            col: column name for trait
            new_col: column name for new trait
            df: DataFrame to map (default: self.df)
        returns:
            DataFrame with mapped trait
        """
        mapping = {v: k for k, vs in trait_mapping.items() for v in vs}
        df = df if df is not None else self.df
        df[new_col] = df[col].map(mapping)
        if df is self.df:
            self.df = df
        return df

    def group_traits(self, group_mapping, col, new_col, df=None):
        """
        Groups traits based on predefined groupings in group_mapping.
        parameters:
            group_mapping: dictionary mapping trait values to group names
            col: column name for trait
            new_col: column name for new trait
            df: DataFrame to map (default: self.df)
        returns:
            DataFrame with mapped trait
        """
        # Flatten the mapping: each possible trait value points to its group
        mapping = {v: k for k, vs in group_mapping.items() for v in vs}
        df = df if df is not None else self.df
        # Map values, keeping originals for unmapped
        df[new_col] = df[col].map(mapping).fillna(df[col])
        if df is self.df:
            self.df = df
        return df

    def print_unique_traits(self, col, df=None):
        """
        Prints unique values in the given column.
        parameters:
            col: column name for trait
            df: DataFrame to print unique values from (default: self.df)
        returns:
            None
        """
        df = df if df is not None else self.df
        unique_traits = df[col].unique()
        print(f'Unique {col}: {unique_traits},')

    def encode_snps(self, snp_info, effect_col='effect_allele', other_col='other_allele',
                    effect_dosage=1, other_dosage=0, missing_value=0):
        """
        Encodes SNP genotypes as additive dosage:
        genotype = n_effect * effect_dosage + n_other * other_dosage
        e.g. AA (effect=1) -> 2, AT -> 1, TT (other=0) -> 0 if effect_dosage=1 and other_dosage=0
        Parameters:
            snp_info: DataFrame with SNP info, indexed by SNP (rsid).
            effect_col: column name for effect allele in snp_info.
            other_col: column name for other allele in snp_info.
            effect_dosage: numeric value to assign per effect allele (e.g. 1)
            other_dosage: numeric value to assign per other allele (e.g. 0)
            missing_value: value to assign for missing/ambiguous genotypes
        Returns:
            DataFrame with trait column and encoded SNPs.
        """
        encoded = pd.DataFrame(index=self.df.index)
        encoded[self.trait_col] = self.df[self.trait_col]

        for snp in self.snp_cols:
            if snp in snp_info.index:
                effect = str(snp_info.loc[snp, effect_col])
                other = str(snp_info.loc[snp, other_col])

                def encode(genotype):
                    if genotype == 'NN' or len(genotype) != 2:
                        return missing_value
                    n_effect = sum(1 for a in genotype if a == effect)
                    n_other = sum(1 for a in genotype if a == other)
                    return n_effect * effect_dosage + n_other * other_dosage

                encoded[snp] = self.df[snp].apply(encode)
        return encoded
    
    def plot_pca_mds(self, encoded_df, trait_col=None, figsize=(14, 5), annotate=True):
        """
        Plot PCA and MDS for trait means of encoded features.
        Parameters:
            encoded_df: DataFrame with trait_col and encoded feature columns.
            trait_col: Column to group by
                       If None, uses self.trait_col.
            figsize: Tuple for figure size.
            annotate: Whether to label points with group names.
        returns:
            None
        """
        trait_col = trait_col or self.trait_col
        trait_means = encoded_df.groupby(trait_col).mean()
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(trait_means)
        mds = MDS(n_components=2, dissimilarity='euclidean')
        mds_result = mds.fit_transform(trait_means)
        pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=trait_means.index)
        mds_df = pd.DataFrame(mds_result, columns=['MDS1', 'MDS2'], index=trait_means.index)

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].scatter(pca_df['PC1'], pca_df['PC2'])
        if annotate:
            for label, (x, y) in pca_df.iterrows():
                axes[0].text(x + 0.03, y, str(label), fontsize=12)
        axes[0].set_title(f'PCA: Encoded Means by {trait_col}')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].grid(True)

        axes[1].scatter(mds_df['MDS1'], mds_df['MDS2'])
        if annotate:
            for label, (x, y) in mds_df.iterrows():
                axes[1].text(x + 0.03, y, str(label), fontsize=10)
        axes[1].set_title(f'MDS: Encoded Means by {trait_col}')
        axes[1].set_xlabel('MDS1')
        axes[1].set_ylabel('MDS2')
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_dendrograms(self, encoded_df, trait_col=None, figsize=(10, 12)):
        """
        Plot vertical hierarchical clustering dendrograms (Euclidean & Manhattan) for group means of encoded features.
        Parameters:
            encoded_df: DataFrame with trait_col and encoded feature columns.
            trait_col: Column to group by
                       If None, uses self.trait_col.
            figsize: Tuple for figure size.
        returns:
            None
        """
        trait_col = trait_col or self.trait_col
        trait_means = encoded_df.groupby(trait_col).mean()
        labels = trait_means.index.astype(str)
        euclidean_dist = pairwise_distances(trait_means, metric='euclidean')
        manhattan_dist = pairwise_distances(trait_means, metric='manhattan')
        linkage_euc = linkage(euclidean_dist, method='ward')
        linkage_man = linkage(manhattan_dist, method='average')
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        # Plot vertical dendrograms (orientation='left')
        dendrogram(
            linkage_euc, 
            labels=labels, 
            orientation='left', 
            leaf_font_size=10, 
            ax=axes[0]
        )
        axes[0].set_title(f'Hierarchical Clustering (Euclidean) by {trait_col}')
        axes[0].set_xlabel('Distance')
        axes[0].set_ylabel(trait_col)
        dendrogram(
            linkage_man, 
            labels=labels, 
            orientation='left', 
            leaf_font_size=10, 
            ax=axes[1]
        )
        axes[1].set_title(f'Hierarchical Clustering (Manhattan) by {trait_col}')
        axes[1].set_xlabel('Distance')
        axes[1].set_ylabel(trait_col)
        plt.tight_layout()
        plt.show()