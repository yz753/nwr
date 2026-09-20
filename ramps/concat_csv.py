import pandas as pd
from pathlib import Path
import numpy as np


class GenotypeMapping:
    genotype_dict = {
        'NWR1': {
            'M1': 'NA',
            'M2': 'NA',
        },
    }
    
    def get_genotype(self, df_row):
        experiment_map = self.genotype_dict.get(df_row['experiment'], {})
        return experiment_map.get(df_row['mouse'], None)


if __name__ == "__main__":
    results_root = Path(f'/exports/eddie/scratch/s2155699/ephys/ramps/ramps_results')
    combined_csv = results_root / f'ramps_classification.csv'

    sub_csv = sorted(
        f for f in results_root.rglob('*_M*_D*_unit*_results.csv')
        )
    if not sub_csv:
        raise ValueError(f"No CSV files found in {results_root}")

    df = pd.concat(
        (pd.read_csv(f) for f in sub_csv 
         if f.stat().st_size > 5),
        ignore_index=True
        ) # ignore empty csv files, otherwise error
    
    # delete firing rate col if exists because it is not calculated correctly
    if 'firing_rate' in df.columns:
        df = df.drop(columns=['firing_rate'])
    
    # add genotype to the csv
    genotype_mapper = GenotypeMapping()
    df['genotype'] = df.apply(genotype_mapper.get_genotype, axis=1)
    
    # for ramps classification, add ramp type and task params into the csv
    df['ramp_type'] = df['outbound_sign'].str.strip() + df['homebound_sign'].str.strip()
        
    df.to_csv(combined_csv, index=False)
    # all sub csvs can be deleted after this
    
    choice = input("Delete all sub CSVs (yes / no)?").strip().lower()
    if choice not in ['yes', 'no']:
        raise ValueError(f"Invalid choice: {choice}")
    if choice == 'yes':
        to_delete = [
            f for f in results_root.glob("*.csv") 
            if f.name != f'ramps_classification.csv'
            ]
        for f in to_delete:
            print("Will delete:", f) # preview

        for f in to_delete:
            f.unlink()
