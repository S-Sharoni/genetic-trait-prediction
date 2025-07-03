import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

class DataProcessor:
    """
    This class provides functionality for loading, cleaning, and standardizing genetic data files.
    It includes methods for:
    - Loading and validating genetic data files
    - Removing duplicate samples based on ID numbers
    - Identifying and processing SNP columns
    - Standardizing phenotype data
    - Handling strand conversions and allele ordering
    - Special processing for specific SNPs (e.g., rs8176719)
    - Cleaning SNP columns and rows
    - Saving cleaned data to a new file
    
    Functions:
    - load_data(): Loads the genetic data file into a pandas DataFrame
    - remove_duplicates(): Removes duplicate samples based on ID numbers
    - define_snp_columns(): Identifies and processes SNP columns
    - standardize_phenotype(): Standardizes the phenotype/trait column
    - standardize_genotypes(): Standardizes genotypes for all SNPs
    - clean_columns(): Cleans SNP columns
    - clean_rows(): Cleans SNP rows
    """

    def __init__(self, data_file: str, snps_file: str):
        self.data_file = data_file
        self.snps_file = snps_file
        self.dataset_name = os.path.basename(data_file).split('_')[0]
        self.df = None
        self.snp_columns = []
        self.snps_info = pd.read_csv(snps_file)
        
        # Reverse strand conversion dictionaries
        self.reverse_to_forward = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        
        # Order of alleles dictionary
        self.replacements = {
            "--": "NN",
            "0": "NN",
            "00": "NN",
            "GA": "AG",
            "GC": "CG",
            "TG": "GT",
            "TC": "CT",
            "TA": "AT",
            "CA": "AC",
            "nan": "NN",
            "NaN": "NN",
            "NAN": "NN",
        }
        # Special conversion for rs8176719
        self.rs8176719_reverse_to_forward = {'A': 'T', 'G': 'C'}
        self.rs8176719_mapping = {
            'TT': 'DI', 
            'TC': 'II', 
            'CC': 'II', 
            '-C': 'II', 
            '-T': 'DD', 
            'T': 'DD', 
            'C': 'II', 
            '--': 'NN'
        }

    def load_data(self):
        """
        Loads the genetic data file into a pandas DataFrame.
        Prints the sample size and columns.
        Parameters: None
        input: self.data_file - path to the genetic data file
        output: self.df - pandas DataFrame of the genetic data
        """
        self.df = pd.read_csv(self.data_file)
        print(f"Sample size: {len(self.df)}")
        print(f"Columns: {self.df.columns.tolist()}")

    def remove_duplicates(self):
        """
        Removes duplicate samples based on ID numbers.
        parameters: None
        input: self.df - pandas DataFrame of the genetic data
        output: self.df - pandas DataFrame of the genetic data with duplicates removed
        """
        self.df = self.df.drop_duplicates(subset='ID-number')
        print(f"Sample size after removing duplicates: {len(self.df)}")

    def define_snp_columns(self):
        """
        Identifies and processes SNP columns.
        Parameters: None
        input: self.df - pandas DataFrame of the genetic data
        output: self.snp_columns - list of SNP column names
        """
        self.snp_columns = [col for col in self.df.columns if col.startswith('rs') or col.startswith('i')]
        print(f"SNP columns: {self.snp_columns}")

    def standardize_phenotype(self, trait_col='trait', new_col=None):
        """
        Standardizes the phenotype/trait column.
        Parameters: new_col - name of the new trait column (str)
        input: trait_col - name of the trait column
        output: self.df - pandas DataFrame of the genetic data with the standardized trait column
        """
        col_to_write = new_col if new_col else trait_col

        def _standardize(val):
            if pd.isna(val):
                return val
            s = str(val).strip()
            if not s:
                return s
            return s[0].upper() + s[1:].lower()
        
        self.df[col_to_write] = self.df[trait_col].apply(_standardize)
        print(f"Standardized {trait_col} values written to '{col_to_write}'.")

    
    def standardize_genotypes(self):
        """
        Standardizes genotypes for all SNPs, with robust special handling for rs8176719.
        Parameters: None
        input: self.df - pandas DataFrame of the genetic data
        output: self.df - pandas DataFrame of the genetic data with the standardized genotypes
        """
        def convert_rs8176719(genotype):
            """
            Converts the genotype of rs8176719 to the standard D/I codes.
            Parameters: None
            input: genotype - string of the genotype
            output: string of the standardized genotype
            """
            if genotype == 'NN':  
                return genotype
            # Convert nucleotides A→T, G→C
            converted = ''.join([self.rs8176719_reverse_to_forward.get(allele, allele) for allele in genotype])
            # Pad if single-letter (e.g. "T"→"TT")
            if len(converted) == 1:
                converted = converted * 2
            # Map to D/I codes if possible, otherwise return converted
            return self.rs8176719_mapping.get(converted, converted)

        def convert_other_rs(genotype, rsid):
            """
            Converts the genotype of other SNPs to the standard D/I codes.
            Parameters: None
            input: genotype - string of the genotype
            output: string of the standardized genotype
            """
            if pd.isna(genotype) or genotype == 'NN':
                return 'NN'
            if rsid not in self.snps_info['rsid'].values:
                return genotype
            effect = self.snps_info.loc[self.snps_info['rsid'] == rsid, 'effect_allele'].values[0]
            other = self.snps_info.loc[self.snps_info['rsid'] == rsid, 'other_allele'].values[0]
            converted = ""
            for allele in genotype:
                if allele in [effect, other]:
                    converted += allele
                else:
                    converted += self.reverse_to_forward.get(allele, allele)
            if len(converted) == 1:
                converted = converted * 2
            return converted[:2]

        def standardize_order(genotype):
            """
            Standardizes the order of alleles in the genotype.
            Parameters: None
            input: genotype - string of the genotype
            output: string of the standardized genotype
            """
            return self.replacements.get(genotype, genotype)

        # Apply SNP-specific conversions
        for rsid in self.snp_columns:
            if rsid == 'rs8176719':
                self.df[rsid] = self.df[rsid].apply(convert_rs8176719)
            elif rsid in self.snps_info['rsid'].values:
                self.df[rsid] = self.df[rsid].apply(lambda g: convert_other_rs(g, rsid))
        
        # Apply final allele order standardization
        for rsid in self.snp_columns:
            self.df[rsid] = self.df[rsid].apply(standardize_order)

        print("Genotypes standardized.")
        print("Unique values per SNP (showing up to 10):")
        for rsid in self.snp_columns:
            print(f"{rsid}: {self.df[rsid].unique()[:10]}")

        standardized_path = f"{self.dataset_name}_standardized.csv"
        self.df.to_csv(standardized_path, index=False)
        print(f"Standardized dataset saved to: {standardized_path}")


    def clean_columns(self, cutoff=3, inplace=True):
        """
        Remove SNP columns with more than (1/cutoff) missing (NN) values.
        Parameters: cutoff - the cutoff for the number of missing values (int)
                    inplace - whether to modify the DataFrame in place (bool)
        output: self.df - pandas DataFrame of the genetic data with the cleaned SNP columns
        """
        df = self.df if inplace else self.df.copy()
        col_threshold = len(df) // cutoff
        cols_to_remove = [col for col in self.snp_columns if (df[col] == 'NN').sum() > col_threshold]
        df.drop(columns=cols_to_remove, inplace=True)
        print(f"Removed {len(cols_to_remove)} columns due to >1/{cutoff} missing values.")
        # Update SNP columns list
        self.snp_columns = [col for col in df.columns if col.startswith('rs') or col.startswith('i')]
        if inplace:
            self.df = df
        print(f"Columns after cleaning: {self.snp_columns}")
        return df

    def clean_rows(self, cutoff=3, inplace=True):
        """     
        Remove rows (samples) with more than (1/cutoff) missing (NN) SNPs.
        Parameters: cutoff - the cutoff for the number of missing values (int)
                    inplace - whether to modify the DataFrame in place (bool)
        output: self.df - pandas DataFrame of the genetic data with the cleaned SNP rows
        """
        df = self.df if inplace else self.df.copy()
        row_threshold = len(self.snp_columns) // cutoff
        df['nn_count'] = df[self.snp_columns].apply(lambda row: (row == 'NN').sum(), axis=1)
        before = len(df)
        df = df[df['nn_count'] <= row_threshold].drop(columns=['nn_count'])
        after = len(df)
        print(f"Removed {before - after} rows due to >1/{cutoff} missing SNPs.")
        if inplace:
            self.df = df
        return df

    def save_cleaned(self, suffix="_cleaned"):
        """
        Saves the cleaned dataset to a new file.
        Parameters: suffix - the suffix for the cleaned file (str)
        output: self.df - pandas DataFrame of the genetic data with the cleaned SNP rows
        """
        cleaned_path = f"{self.dataset_name}{suffix}.csv"
        self.df.to_csv(cleaned_path, index=False)
        print(f"Cleaned dataset saved to: {cleaned_path}")
