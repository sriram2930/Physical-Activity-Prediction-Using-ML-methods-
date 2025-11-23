"""
PAMAP2 Memory-Efficient Data Loading and Cleaning Pipeline
===========================================================
This version handles large datasets without running out of memory.

Author: [Your Name]
Date: Fall 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import gc  # Garbage collector for memory management
warnings.filterwarnings('ignore')

class PAMAP2DataLoader:
    """
    Memory-efficient data loader and preprocessor for PAMAP2 dataset
    """
    
    # Activity labels mapping
    ACTIVITY_LABELS = {
        0: 'transient',
        1: 'lying',
        2: 'sitting',
        3: 'standing',
        4: 'walking',
        5: 'running',
        6: 'cycling',
        7: 'Nordic_walking',
        9: 'watching_TV',
        10: 'computer_work',
        11: 'car_driving',
        12: 'ascending_stairs',
        13: 'descending_stairs',
        16: 'vacuum_cleaning',
        17: 'ironing',
        18: 'folding_laundry',
        19: 'house_cleaning',
        20: 'playing_soccer',
        24: 'rope_jumping'
    }
    
    # Column names for the dataset (54 columns total)
    COLUMN_NAMES = [
        'timestamp', 'activityID', 'heart_rate',
        # IMU hand (17 features)
        'hand_temp', 'hand_3D_acc_x', 'hand_3D_acc_y', 'hand_3D_acc_z',
        'hand_3D_acc_x2', 'hand_3D_acc_y2', 'hand_3D_acc_z2',
        'hand_3D_gyro_x', 'hand_3D_gyro_y', 'hand_3D_gyro_z',
        'hand_3D_mag_x', 'hand_3D_mag_y', 'hand_3D_mag_z',
        'hand_orientation_1', 'hand_orientation_2', 'hand_orientation_3', 'hand_orientation_4',
        # IMU chest (17 features)
        'chest_temp', 'chest_3D_acc_x', 'chest_3D_acc_y', 'chest_3D_acc_z',
        'chest_3D_acc_x2', 'chest_3D_acc_y2', 'chest_3D_acc_z2',
        'chest_3D_gyro_x', 'chest_3D_gyro_y', 'chest_3D_gyro_z',
        'chest_3D_mag_x', 'chest_3D_mag_y', 'chest_3D_mag_z',
        'chest_orientation_1', 'chest_orientation_2', 'chest_orientation_3', 'chest_orientation_4',
        # IMU ankle (17 features)
        'ankle_temp', 'ankle_3D_acc_x', 'ankle_3D_acc_y', 'ankle_3D_acc_z',
        'ankle_3D_acc_x2', 'ankle_3D_acc_y2', 'ankle_3D_acc_z2',
        'ankle_3D_gyro_x', 'ankle_3D_gyro_y', 'ankle_3D_gyro_z',
        'ankle_3D_mag_x', 'ankle_3D_mag_y', 'ankle_3D_mag_z',
        'ankle_orientation_1', 'ankle_orientation_2', 'ankle_orientation_3', 'ankle_orientation_4'
    ]
    
    def __init__(self, data_dir='pamap2_data', sample_size=None):
        """
        Initialize the data loader
        
        Parameters:
        -----------
        data_dir : str
            Directory containing PAMAP2 .dat files
        sample_size : int, optional
            If provided, sample this many rows per subject to save memory
            Example: sample_size=50000 means 50k rows per subject (450k total)
        """
        self.data_dir = Path(data_dir)
        self.sample_size = sample_size
        self.raw_data = None
        self.cleaned_data = None
        self.stats = {}
        
    def load_single_subject(self, subject_id):
        """
        Load data for a single subject with memory optimization
        
        Parameters:
        -----------
        subject_id : int
            Subject ID (101-109)
            
        Returns:
        --------
        pd.DataFrame
            Raw data for the subject
        """
        file_path = self.data_dir / f'subject{subject_id}.dat'
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"📂 Loading subject {subject_id}...")
        
        # Load data with space delimiter
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=self.COLUMN_NAMES)
        
        # Add subject ID column
        df['subject_id'] = subject_id
        
        initial_rows = len(df)
        
        # Sample if requested (memory optimization)
        if self.sample_size and len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42)
            print(f"   ✅ Loaded {len(df):,} rows (sampled from {initial_rows:,})")
        else:
            print(f"   ✅ Loaded {len(df):,} rows")
        
        # Optimize memory usage by converting to appropriate dtypes
        df = self._optimize_dtypes(df)
        
        return df
    
    def _optimize_dtypes(self, df):
        """Optimize data types to reduce memory usage"""
        
        # Convert integers to smaller types
        for col in ['activityID', 'subject_id']:
            if col in df.columns:
                df[col] = df[col].astype('int8')
        
        # Convert floats to float32 (uses half the memory of float64)
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = df[col].astype('float32')
        
        return df
    
    def load_all_subjects(self, subject_ids=None):
        """
        Load data for all or specified subjects (memory-efficient version)
        
        Parameters:
        -----------
        subject_ids : list, optional
            List of subject IDs to load. If None, loads all (101-109)
            
        Returns:
        --------
        pd.DataFrame
            Combined raw data from all subjects
        """
        if subject_ids is None:
            subject_ids = range(101, 110)  # 101 to 109
        
        print("=" * 60)
        print("🚀 LOADING PAMAP2 DATASET (MEMORY-EFFICIENT MODE)")
        print("=" * 60)
        
        if self.sample_size:
            print(f"⚡ Memory optimization: Using {self.sample_size:,} samples per subject")
        
        dfs = []
        total_rows = 0
        
        for subject_id in subject_ids:
            try:
                df = self.load_single_subject(subject_id)
                dfs.append(df)
                total_rows += len(df)
                
                # Force garbage collection after each subject
                gc.collect()
                
            except FileNotFoundError as e:
                print(f"   ⚠️  Skipping subject {subject_id}: {e}")
            except MemoryError:
                print(f"   ⚠️  Memory error loading subject {subject_id}")
                print(f"   💡 Try reducing sample_size or loading fewer subjects")
                break
        
        if not dfs:
            raise ValueError("No data files found!")
        
        # Concatenate with memory optimization
        print("\n🔄 Combining all subjects...")
        try:
            self.raw_data = pd.concat(dfs, ignore_index=True)
            del dfs  # Free memory
            gc.collect()
        except MemoryError:
            print("❌ Memory error during concatenation!")
            print("💡 Solution: Reduce sample_size or use fewer subjects")
            raise
        
        print("=" * 60)
        print(f"✅ Total rows loaded: {len(self.raw_data):,}")
        print(f"✅ Total subjects: {self.raw_data['subject_id'].nunique()}")
        print(f"✅ Memory usage: {self.raw_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print("=" * 60)
        
        return self.raw_data
    
    def analyze_missing_values(self):
        """Analyze missing values in the dataset"""
        
        if self.raw_data is None:
            raise ValueError("Data not loaded yet. Call load_all_subjects() first.")
        
        print("\n🔍 MISSING VALUE ANALYSIS")
        print("=" * 60)
        
        # Calculate missing percentages
        missing_percent = (self.raw_data.isna().sum() / len(self.raw_data) * 100).sort_values(ascending=False)
        
        # Show top columns with missing values
        missing_cols = missing_percent[missing_percent > 0]
        
        if len(missing_cols) > 0:
            print(f"\n📊 Columns with missing values ({len(missing_cols)} columns):")
            print("-" * 60)
            for col, pct in missing_cols.head(20).items():
                print(f"   {col:30s}: {pct:6.2f}% missing")
            
            if len(missing_cols) > 20:
                print(f"   ... and {len(missing_cols) - 20} more columns")
        else:
            print("✅ No missing values found!")
        
        self.stats['missing_values'] = missing_cols.to_dict()
        
        return missing_cols
    
    def clean_data(self, remove_transient=True, handle_missing='drop'):
        """
        Clean and preprocess the data (memory-efficient version)
        
        Parameters:
        -----------
        remove_transient : bool
            Whether to remove transient activities (label 0)
        handle_missing : str
            'drop': Drop rows with missing values (recommended for memory)
            'interpolate': Interpolate missing values (uses more memory)
            'forward_fill': Forward fill missing values
            
        Returns:
        --------
        pd.DataFrame
            Cleaned data
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded yet. Call load_all_subjects() first.")
        
        print("\n🧹 CLEANING DATA")
        print("=" * 60)
        
        df = self.raw_data
        initial_rows = len(df)
        
        # Step 1: Remove transient activities (label 0)
        if remove_transient:
            transient_count = (df['activityID'] == 0).sum()
            df = df[df['activityID'] != 0].copy()
            print(f"✅ Removed {transient_count:,} transient activity rows")
            gc.collect()  # Free memory
        
        # Step 2: Add activity name
        df['activity_name'] = df['activityID'].map(self.ACTIVITY_LABELS)
        print(f"✅ Added activity names")
        
        # Step 3: Handle missing values
        print(f"\n📊 Handling missing values (method: {handle_missing})...")
        
        if handle_missing == 'drop':
            # Drop rows with ANY missing values (most memory-efficient)
            nan_rows = df.isna().any(axis=1).sum()
            df = df.dropna()
            print(f"   ✅ Dropped {nan_rows:,} rows with missing values")
        
        elif handle_missing == 'interpolate':
            # Interpolate within each subject and activity
            print(f"   ⚠️  This may take a while and use more memory...")
            for subject in df['subject_id'].unique():
                for activity in df[df['subject_id'] == subject]['activityID'].unique():
                    mask = (df['subject_id'] == subject) & (df['activityID'] == activity)
                    df.loc[mask] = df.loc[mask].interpolate(method='linear', limit_direction='both')
            
            # Drop remaining NaN
            nan_rows = df.isna().any(axis=1).sum()
            if nan_rows > 0:
                df = df.dropna()
                print(f"   ✅ Dropped {nan_rows:,} rows with remaining NaN")
        
        elif handle_missing == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
            nan_rows = df.isna().any(axis=1).sum()
            if nan_rows > 0:
                df = df.dropna()
                print(f"   ✅ Dropped {nan_rows:,} rows with remaining NaN")
        
        # Step 4: Reset index
        df = df.reset_index(drop=True)
        
        # Free memory
        del self.raw_data
        self.raw_data = None
        gc.collect()
        
        final_rows = len(df)
        rows_removed = initial_rows - final_rows
        
        print("=" * 60)
        print(f"✅ Cleaning complete!")
        print(f"   Initial rows: {initial_rows:,}")
        print(f"   Final rows: {final_rows:,}")
        print(f"   Rows removed: {rows_removed:,} ({rows_removed/initial_rows*100:.2f}%)")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print("=" * 60)
        
        self.cleaned_data = df
        self.stats['initial_rows'] = initial_rows
        self.stats['final_rows'] = final_rows
        self.stats['rows_removed'] = rows_removed
        
        return df
    
    def get_activity_distribution(self):
        """Get distribution of activities"""
        
        if self.cleaned_data is None:
            raise ValueError("Data not cleaned yet. Call clean_data() first.")
        
        print("\n📊 ACTIVITY DISTRIBUTION")
        print("=" * 60)
        
        activity_counts = self.cleaned_data['activity_name'].value_counts()
        
        for activity, count in activity_counts.items():
            pct = count / len(self.cleaned_data) * 100
            print(f"   {activity:20s}: {count:8,} ({pct:5.2f}%)")
        
        print("=" * 60)
        
        return activity_counts
    
    def save_cleaned_data(self, filename='pamap2_cleaned.csv'):
        """Save cleaned data to CSV"""
        
        if self.cleaned_data is None:
            raise ValueError("Data not cleaned yet. Call clean_data() first.")
        
        output_path = self.data_dir / filename
        
        print(f"\n💾 Saving cleaned data...")
        self.cleaned_data.to_csv(output_path, index=False)
        
        print(f"   ✅ Saved to: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024**2:.2f} MB")
        print(f"   Rows: {len(self.cleaned_data):,}")
        print(f"   Columns: {len(self.cleaned_data.columns)}")
    
    def get_summary_stats(self):
        """Get summary statistics"""
        
        if self.cleaned_data is None:
            raise ValueError("Data not cleaned yet. Call clean_data() first.")
        
        print("\n📈 SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total samples: {len(self.cleaned_data):,}")
        print(f"Total subjects: {self.cleaned_data['subject_id'].nunique()}")
        print(f"Total activities: {self.cleaned_data['activityID'].nunique()}")
        print(f"Total features: {len(self.cleaned_data.columns)}")
        print(f"Time range: {self.cleaned_data['timestamp'].min():.2f} - {self.cleaned_data['timestamp'].max():.2f}")
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("⚡ PAMAP2 MEMORY-EFFICIENT DATA LOADING")
    print("=" * 60)
    print("\nThis script handles large datasets without memory errors.")
    print("\nOptions:")
    print("1. Load full dataset (may use 1-2 GB RAM)")
    print("2. Load sampled dataset (uses less memory)")
    print("\n" + "=" * 60)
    
    # Option 1: Try full dataset first
    try:
        print("\n🔄 Attempting to load full dataset...")
        loader = PAMAP2DataLoader(data_dir='pamap2_data', sample_size=None)
        raw_data = loader.load_all_subjects()
        
    except MemoryError:
        print("\n⚠️  Memory error with full dataset!")
        print("🔄 Switching to sampled mode (50,000 rows per subject)...")
        
        # Option 2: Load sampled data
        loader = PAMAP2DataLoader(data_dir='pamap2_data', sample_size=50000)
        raw_data = loader.load_all_subjects()
    
    # Analyze missing values
    print("\n" + "=" * 60)
    loader.analyze_missing_values()
    
    # Clean data (use 'drop' method for memory efficiency)
    print("\n" + "=" * 60)
    cleaned_data = loader.clean_data(remove_transient=True, handle_missing='drop')
    
    # Get activity distribution
    loader.get_activity_distribution()
    
    # Get summary stats
    loader.get_summary_stats()
    
    # Save cleaned data
    loader.save_cleaned_data()
    
    print("\n✅ Data loading and cleaning complete!")
    print("\n💡 Next steps:")
    print("   1. Review the cleaned data")
    print("   2. Run feature engineering")
    print("   3. Create visualizations")
    print("   4. Train ML models")