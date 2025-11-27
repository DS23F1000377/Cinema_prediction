# Cinema Audience Forecasting

Time-series forecasting model to predict daily cinema audience counts using data from two booking platforms: BookNow (online) and CinePOS (point-of-sale).

## 📊 Problem Statement

Predict daily audience attendance at movie theaters by combining:
- **BookNow**: Online booking platform data
- **CinePOS**: Theater point-of-sale system data

## 🗂️ Dataset

The project uses 8 CSV files:

| File | Description |
|------|-------------|
| `booknow_booking.csv` | Online booking transactions |
| `booknow_theaters.csv` | Theater metadata (BookNow) |
| `booknow_visits.csv` | **Target data** - Daily audience counts |
| `cinePOS_booking.csv` | POS booking transactions |
| `cinePOS_theaters.csv` | Theater metadata (CinePOS) |
| `date_info.csv` | Calendar information |
| `movie_theater_id_relation.csv` | Theater ID mapping between systems |
| `sample_submission.csv` | Submission format |

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd CINEMA_PREDICTION
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add Data Files
Place all CSV files in the root directory (not tracked by Git due to size).

### 5. Run the Model
```bash
python cinema_prediction.py
```

## 📈 Features Engineered

The pipeline automatically creates 50+ features including:

### Temporal Features
- Day of week, month, year
- Weekend indicators
- Month/quarter start/end flags

### Booking Metrics
- BookNow booking counts
- CinePOS booking counts
- Combined booking volumes
- Online vs offline ratios

### Historical Features
- Lag features (1, 7, 14, 28 days)
- Rolling averages (7, 14, 28-day windows)
- Rolling standard deviations

### Theater Features
- Metadata from both platforms
- Cross-platform theater mappings

## 🏗️ Architecture

```
CinemaDataLoader
├── Load all CSV files
└── Generate dataset statistics

FeatureEngineer
├── Extract temporal features
├── Merge theater metadata
├── Aggregate booking metrics
└── Create historical lag/rolling features

PredictionModel
├── Prepare training data
├── Train LightGBM model
└── Generate predictions
```

## 📊 Model Details

- **Algorithm**: LightGBM (Gradient Boosting)
- **Validation**: Time-based split (80/20)
- **Metrics**: MAE (Mean Absolute Error), RMSE
- **Features**: 50+ engineered features
- **Output**: `submission.csv`

## 📁 Project Structure

```
CINEMA_PREDICTION/
├── cinema_prediction.py      # Main pipeline script
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
├── README.md                 # This file
├── *.csv                     # Data files (not in Git)
└── submission.csv            # Output (not in Git)
```

## 🎯 Results

The model outputs:
- Validation MAE and RMSE scores
- Top 15 most important features
- `submission.csv` ready for competition upload

## 🔧 Customization

To modify the model:

1. **Adjust hyperparameters**: Edit `_default_hyperparams()` in `PredictionModel` class
2. **Add features**: Extend methods in `FeatureEngineer` class
3. **Change validation split**: Modify split ratio in `main()` function

## 📦 Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- lightgbm >= 4.0.0
- matplotlib >= 3.7.0 (optional)
- seaborn >= 0.12.0 (optional)

## 📝 License

This project is for competition/educational purposes.

## 👤 Author

Sumit Ghughtyal

## 🤝 Contributing

Feel free to open issues or submit pull requests with improvements.