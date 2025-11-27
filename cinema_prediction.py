"""
Cinema Audience Prediction System
Original implementation for time-series forecasting competition
Author: Sumit Ghughtyal
Date: June 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error as calc_mae
from sklearn.metrics import mean_squared_error as calc_mse
import lightgbm
from sklearn.preprocessing import LabelEncoder as CatEncoder

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class CinemaDataLoader:
    """Custom data loading and initial processing"""
    
    def __init__(self, base_paths=None):
        self.datasets = {}
        self.paths = base_paths or self._get_default_paths()
    
    def _get_default_paths(self):
        """Define file paths"""
        return {
            'bn_bookings': 'booknow_booking.csv',
            'bn_theaters': 'booknow_theaters.csv',
            'bn_visits': 'booknow_visits.csv',
            'cp_bookings': 'cinePOS_booking.csv',
            'cp_theaters': 'cinePOS_theaters.csv',
            'calendar': 'date_info.csv',
            'theater_links': 'movie_theater_id_relation.csv',
            'submission_template': 'sample_submission.csv'
        }
    
    def load_all_data(self):
        """Load all CSV files into memory"""
        print("=" * 70)
        print("INITIALIZING DATA LOADING PROCESS")
        print("=" * 70)
        
        for key, path in self.paths.items():
            try:
                self.datasets[key] = pd.read_csv(path)
                rows, cols = self.datasets[key].shape
                print(f"✓ {key:25s} | Rows: {rows:8,d} | Cols: {cols:3d}")
            except Exception as e:
                print(f"✗ {key:25s} | Error: {str(e)}")
        
        return self.datasets
    
    def show_statistics(self):
        """Display dataset statistics"""
        print("\n" + "=" * 70)
        print("DATASET STATISTICS")
        print("=" * 70)
        
        if 'bn_visits' in self.datasets:
            target_data = self.datasets['bn_visits']
            if pd.api.types.is_numeric_dtype(target_data['audience_count']):
                print(f"\nTarget Variable Analysis:")
                print(f"  Mean audience: {target_data['audience_count'].mean():.2f}")
                print(f"  Median audience: {target_data['audience_count'].median():.2f}")
                print(f"  Std deviation: {target_data['audience_count'].std():.2f}")
                print(f"  Zero counts: {(target_data['audience_count'] == 0).sum():,d}")
        
        print(f"\nTheater Coverage:")
        print(f"  BookNow theaters: {self.datasets['bn_theaters'].shape[0]:,d}")
        print(f"  CinePOS theaters: {self.datasets['cp_theaters'].shape[0]:,d}")
        print(f"  Theater mappings: {self.datasets['theater_links'].shape[0]:,d}")


class FeatureEngineer:
    """Custom feature engineering pipeline"""
    
    def __init__(self):
        self.encoders = {}
        
    def extract_time_attributes(self, dataframe, date_column):
        """Extract temporal components from dates"""
        df_copy = dataframe.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy[date_column])
        
        # Basic temporal features
        df_copy['year_num'] = df_copy['timestamp'].dt.year
        df_copy['month_num'] = df_copy['timestamp'].dt.month
        df_copy['day_num'] = df_copy['timestamp'].dt.day
        df_copy['weekday_num'] = df_copy['timestamp'].dt.dayofweek
        df_copy['yearday_num'] = df_copy['timestamp'].dt.dayofyear
        df_copy['week_num'] = df_copy['timestamp'].dt.isocalendar().week.astype(int)
        
        # Binary indicators
        df_copy['is_weekend_flag'] = (df_copy['weekday_num'] >= 5).astype(np.int8)
        df_copy['is_monthstart_flag'] = df_copy['timestamp'].dt.is_month_start.astype(np.int8)
        df_copy['is_monthend_flag'] = df_copy['timestamp'].dt.is_month_end.astype(np.int8)
        df_copy['is_quarterstart_flag'] = df_copy['timestamp'].dt.is_quarter_start.astype(np.int8)
        
        return df_copy
    
    def merge_theater_metadata(self, main_df, bn_theater_df, cp_theater_df, mapping_df):
        """Combine theater information from both systems"""
        result = main_df.copy()
        
        if 'cine_theater_id' in cp_theater_df.columns:
            cp_theater_df = cp_theater_df.rename(columns={'cine_theater_id': 'movie_theater_id'})
            
        if 'id' in bn_theater_df.columns and 'book_theater_id' not in bn_theater_df.columns:
            bn_theater_df = bn_theater_df.rename(columns={'id': 'book_theater_id'})

        # --- UNIVERSAL ID CLEANING ---
        def clean_id_column(df, col_name):
            if col_name in df.columns:
                # Extract regex pattern (\d+) which means "digits"
                df[col_name] = df[col_name].astype(str).str.extract(r'(\d+)')[0].fillna(0).astype(int)
            return df

        result = clean_id_column(result, 'book_theater_id')
        bn_theater_df = clean_id_column(bn_theater_df, 'book_theater_id')
        mapping_df = clean_id_column(mapping_df, 'book_theater_id')
        result = clean_id_column(result, 'movie_theater_id')
        cp_theater_df = clean_id_column(cp_theater_df, 'movie_theater_id')
        # -----------------------------
        
        if 'book_theater_id' in result.columns:
            result = result.merge(bn_theater_df, on='book_theater_id', 
                                how='left', suffixes=('', '_booknow'))
        
        result = result.merge(mapping_df, on='book_theater_id', how='left')
        
        if 'movie_theater_id' in result.columns:
            result = result.merge(cp_theater_df, on='movie_theater_id', 
                                how='left', suffixes=('', '_cinepos'))
        
        return result
    
    def aggregate_booking_metrics(self, main_df, bn_bookings, cp_bookings):
        """Calculate booking statistics from both platforms"""
        df_result = main_df.copy()
        
        if 'book_theater_id' in bn_bookings.columns:
             bn_bookings['book_theater_id'] = bn_bookings['book_theater_id'].astype(str).str.extract(r'(\d+)')[0].fillna(0).astype(int)
             
        if not bn_bookings.empty:
            date_field = self._find_date_column(bn_bookings)
            bn_temp = bn_bookings.copy()
            bn_temp['booking_timestamp'] = pd.to_datetime(bn_temp[date_field])
            
            bn_aggregated = bn_temp.groupby(['book_theater_id', 'booking_timestamp']).size().reset_index(name='booknow_count')
            bn_aggregated.rename(columns={'booking_timestamp': 'timestamp'}, inplace=True)
            
            df_result = df_result.merge(bn_aggregated, 
                                        on=['book_theater_id', 'timestamp'], 
                                        how='left')
            df_result['booknow_count'].fillna(0, inplace=True)
        
        if not cp_bookings.empty and 'movie_theater_id' in df_result.columns:
            date_field = self._find_date_column(cp_bookings)
            cp_temp = cp_bookings.copy()
            cp_temp['booking_timestamp'] = pd.to_datetime(cp_temp[date_field])
            
            if 'movie_theater_id' in cp_temp.columns:
                 if cp_temp['movie_theater_id'].dtype == 'object':
                     cp_temp['movie_theater_id'] = cp_temp['movie_theater_id'].astype(str).str.extract(r'(\d+)')[0].fillna(0).astype(int)

            cp_aggregated = cp_temp.groupby(['movie_theater_id', 'booking_timestamp']).size().reset_index(name='cinepos_count')
            cp_aggregated.rename(columns={'booking_timestamp': 'timestamp'}, inplace=True)
            
            df_result = df_result.merge(cp_aggregated,
                                        on=['movie_theater_id', 'timestamp'],
                                        how='left')
            df_result['cinepos_count'].fillna(0, inplace=True)
        
        if 'booknow_count' in df_result.columns and 'cinepos_count' in df_result.columns:
            df_result['combined_bookings'] = df_result['booknow_count'] + df_result['cinepos_count']
            df_result['online_proportion'] = df_result['booknow_count'] / (df_result['combined_bookings'] + 0.01)
        
        return df_result
    
    def create_historical_features(self, dataframe, metric='audience_count', 
                                   delays=[1, 7, 14, 28]):
        """Generate lagged and rolling window features"""
        df_sorted = dataframe.sort_values(['book_theater_id', 'timestamp']).copy()
        
        for delay in delays:
            col_name = f'{metric}_prev_{delay}d'
            df_sorted[col_name] = df_sorted.groupby('book_theater_id')[metric].shift(delay)
        
        for window_size in [7, 14, 28]:
            mean_col = f'{metric}_avg_{window_size}d'
            std_col = f'{metric}_std_{window_size}d'
            
            df_sorted[mean_col] = (
                df_sorted.groupby('book_theater_id')[metric]
                .transform(lambda x: x.rolling(window_size, min_periods=1).mean())
            )
            df_sorted[std_col] = (
                df_sorted.groupby('book_theater_id')[metric]
                .transform(lambda x: x.rolling(window_size, min_periods=1).std())
            )
        
        return df_sorted
    
    def encode_categorical_features(self, dataframe, cat_columns):
        """Transform categorical variables to numeric"""
        df_encoded = dataframe.copy()
        
        for col in cat_columns:
            if col in df_encoded.columns and df_encoded[col].dtype == 'object':
                if df_encoded[col].nunique() < 500:
                    if col not in self.encoders:
                        self.encoders[col] = CatEncoder()
                        df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
                    else:
                        df_encoded[col] = df_encoded[col].map(
                            lambda x: self.encoders[col].transform([str(x)])[0]
                            if str(x) in self.encoders[col].classes_ else -1
                        )
        
        return df_encoded
    
    def _find_date_column(self, dataframe):
        """Identify date column in dataframe"""
        for col in dataframe.columns:
            if 'date' in col.lower():
                return col
        return None
    
    def build_complete_features(self, base_df, bn_bookings, cp_bookings, 
                               bn_theaters, cp_theaters, mapping, calendar):
        """Orchestrate full feature engineering pipeline"""
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 70)
        
        date_col = self._find_date_column(base_df)
        
        print("Step 1: Extracting temporal attributes...")
        enhanced_df = self.extract_time_attributes(base_df, date_col)
        
        print("Step 2: Merging theater information...")
        enhanced_df = self.merge_theater_metadata(enhanced_df, bn_theaters, 
                                                 cp_theaters, mapping)
        
        print("Step 3: Aggregating booking metrics...")
        enhanced_df = self.aggregate_booking_metrics(enhanced_df, bn_bookings, 
                                                    cp_bookings)
        
        if 'audience_count' in enhanced_df.columns:
            print("Step 4: Creating historical features...")
            enhanced_df = self.create_historical_features(enhanced_df)
        
        print(f"\n✓ Feature engineering complete")
        print(f"  Initial columns: {len(base_df.columns)}")
        print(f"  Enhanced columns: {len(enhanced_df.columns)}")
        
        return enhanced_df


class PredictionModel:
    """Custom model training and prediction"""
    
    def __init__(self, hyperparams=None):
        self.model = None
        self.params = hyperparams or self._default_hyperparams()
        self.feature_list = None
    
    def _default_hyperparams(self):
        """Default model configuration"""
        return {
            'objective': 'regression',
            'metric': 'mae', # LightGBM will optimize MAE of Log values
            'boosting': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'random_state': RANDOM_SEED,
            'verbosity': -1
        }
    
    def prepare_training_data(self, dataframe, target_name='audience_count'):
        """Split features and target with LOG TRANSFORMATION"""
        exclude_cols = ['book_theater_id', 'movie_theater_id', 'timestamp', 
                        target_name, 'ID']
        exclude_cols.extend([c for c in dataframe.columns if 'date' in c.lower() and c != 'timestamp'])
        
        feature_candidates = [c for c in dataframe.columns if c not in exclude_cols]
        
        numeric_features = []
        for col in feature_candidates:
            if dataframe[col].dtype in ['int64', 'float64', 'int8', 'float32']:
                numeric_features.append(col)
        
        X = dataframe[numeric_features].fillna(-999)
        
        # --- UPDATE: Log Transform Target ---
        if target_name in dataframe.columns:
            y = dataframe[target_name]
            y = np.log1p(y) # log(1+x) to handle zeros safely
        else:
            y = None
        # ------------------------------------
        
        return X, y, numeric_features
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train gradient boosting model"""
        print("\n" + "=" * 70)
        print("MODEL TRAINING")
        print("=" * 70)
        
        print(f"Training samples: {len(X_train):,d}")
        print(f"Validation samples: {len(X_val):,d}")
        print(f"Number of features: {X_train.shape[1]}")
        
        train_set = lightgbm.Dataset(X_train, label=y_train)
        val_set = lightgbm.Dataset(X_val, label=y_val, reference=train_set)
        
        print("\nTraining in progress...")
        self.model = lightgbm.train(
            self.params,
            train_set,
            num_boost_round=1000,
            valid_sets=[train_set, val_set],
            valid_names=['training', 'validation'],
            callbacks=[
                lightgbm.early_stopping(50),
                lightgbm.log_evaluation(100)
            ]
        )
        
        # Evaluate (on Log scale first)
        val_predictions_log = self.model.predict(X_val)
        val_predictions_log = np.clip(val_predictions_log, 0, None)
        
        # Convert back to Real scale for readable metrics
        val_predictions_real = np.expm1(val_predictions_log)
        y_val_real = np.expm1(y_val)
        
        mae_score = calc_mae(y_val_real, val_predictions_real)
        rmse_score = np.sqrt(calc_mse(y_val_real, val_predictions_real))
        
        print(f"\n{'='*70}")
        print(f"VALIDATION RESULTS (Real Scale)")
        print(f"{'='*70}")
        print(f"MAE:  {mae_score:.4f}")
        print(f"RMSE: {rmse_score:.4f}")
        
        importance_df = pd.DataFrame({
            'feature_name': X_train.columns,
            'importance_score': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance_score', ascending=False)
        
        print(f"\n{'='*70}")
        print(f"TOP 15 IMPORTANT FEATURES")
        print(f"{'='*70}")
        for idx, row in importance_df.head(15).iterrows():
            print(f"{row['feature_name']:40s} {row['importance_score']:10.0f}")
        
        return mae_score, rmse_score
    
    def predict(self, X_test):
        """Generate predictions with INVERSE TRANSFORM"""
        predictions = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        predictions = np.clip(predictions, 0, None)
        
        # --- UPDATE: Inverse Log Transform ---
        predictions = np.expm1(predictions) # exp(x) - 1
        # -------------------------------------
        
        return predictions


# Main execution pipeline
def main():
    """Execute complete pipeline with RECURSIVE FORECASTING"""
    
    loader = CinemaDataLoader()
    data = loader.load_all_data()
    loader.show_statistics()
    
    engineer = FeatureEngineer()
    
    print("\n[TRAINING PHASE] preparing features...")
    train_enhanced = engineer.build_complete_features(
        data['bn_visits'],
        data['bn_bookings'],
        data['cp_bookings'],
        data['bn_theaters'],
        data['cp_theaters'],
        data['theater_links'],
        data['calendar']
    )
    
    cat_cols = train_enhanced.select_dtypes(include=['object']).columns.tolist()
    train_enhanced = engineer.encode_categorical_features(train_enhanced, cat_cols)
    
    model_trainer = PredictionModel()
    
    train_enhanced_sorted = train_enhanced.sort_values('timestamp')
    split_idx = int(len(train_enhanced_sorted) * 0.9)
    
    train_data = train_enhanced_sorted.iloc[:split_idx]
    val_data = train_enhanced_sorted.iloc[split_idx:]
    
    X_train, y_train, _ = model_trainer.prepare_training_data(train_data)
    X_val, y_val, _ = model_trainer.prepare_training_data(val_data)
    
    model_trainer.train_model(X_train, y_train, X_val, y_val)
    
    # =========================================================================
    # RECURSIVE FORECASTING LOOP
    # =========================================================================
    print("\n" + "=" * 70)
    print("STARTING RECURSIVE PREDICTION LOOP")
    print("=" * 70)
    
    test_template = data['submission_template'].copy()
    split_ids = test_template['ID'].str.split('_', expand=True)
    test_template['book_theater_id'] = split_ids[1].astype(int)
    test_template['show_date'] = pd.to_datetime(split_ids[2])
    
    if 'audience_count' in test_template.columns:
        test_template = test_template.drop(columns=['audience_count'])
        
    test_enhanced = engineer.build_complete_features(
        test_template,
        data['bn_bookings'],
        data['cp_bookings'],
        data['bn_theaters'],
        data['cp_theaters'],
        data['theater_links'],
        data['calendar']
    )
    test_enhanced = engineer.encode_categorical_features(test_enhanced, cat_cols)
    
    test_enhanced['audience_count'] = np.nan
    common_cols = list(set(train_enhanced.columns) & set(test_enhanced.columns))
    full_data = pd.concat([train_enhanced[common_cols], test_enhanced[common_cols]], ignore_index=True)
    full_data = full_data.sort_values(['book_theater_id', 'timestamp']).reset_index(drop=True)
    
    unique_dates = sorted(test_enhanced['show_date'].unique())
    print(f"Forecasting for {len(unique_dates)} unique dates...")
    
    for i, current_date in enumerate(unique_dates):
        print(f"[{i+1}/{len(unique_dates)}] Forecasting for: {pd.to_datetime(current_date).date()} ...", end=" ")
        
        full_data = engineer.create_historical_features(full_data)
        
        current_mask = (full_data['timestamp'] == current_date) & (full_data['audience_count'].isna())
        
        if current_mask.sum() == 0:
            print("Done (No missing rows).")
            continue
            
        current_batch = full_data[current_mask].copy()
        
        X_test, _, _ = model_trainer.prepare_training_data(current_batch)
        daily_preds = model_trainer.predict(X_test)
        
        full_data.loc[current_mask, 'audience_count'] = daily_preds
        
        print(f"Predicted {len(daily_preds)} rows. Mean: {daily_preds.mean():.2f}")

    final_test_data = full_data[full_data['timestamp'].isin(unique_dates)].copy()
    
    submission_keys = test_template[['ID', 'book_theater_id', 'show_date']].copy()
    submission_keys.rename(columns={'show_date': 'timestamp'}, inplace=True)
    
    final_output = submission_keys.merge(
        final_test_data[['book_theater_id', 'timestamp', 'audience_count']], 
        on=['book_theater_id', 'timestamp'], 
        how='left'
    )
    
    final_output['audience_count'] = final_output['audience_count'].fillna(0)
    final_output[['ID', 'audience_count']].to_csv('submission_recursive.csv', index=False)
    
    print("\n" + "=" * 70)
    print("RECURSIVE PIPELINE COMPLETE")
    print("=" * 70)
    print(f"✓ Submission file: submission_recursive.csv")
    print(f"  Total predictions: {len(final_output):,d}")
    print(f"  Average prediction: {final_output['audience_count'].mean():.2f}")
    print(f"  Min: {final_output['audience_count'].min():.2f} | Max: {final_output['audience_count'].max():.2f}")

if __name__ == "__main__":
    main()