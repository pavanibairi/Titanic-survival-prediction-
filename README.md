# 🚢 Titanic Survival Prediction

Predict passenger survival on the Titanic using Machine Learning with Feature Engineering and Model Comparison.

## 📊 Project Overview
This project analyzes the famous Titanic dataset to predict which passengers survived the tragedy. Uses 3 different ML models and compares their performance with proper evaluation metrics.

**Key Goal**: Achieve high accuracy using Logistic Regression, Decision Tree, and Random Forest while handling missing data and engineering useful features.

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Models**: Logistic Regression, Decision Tree, Random Forest
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## 📁 Project Structure
```bash
Titanic-Survival-Prediction/
│
├── train.csv              # Training data from Kaggle
├── test.csv               # Test data for submission
├── titanic_model.py       # Main ML pipeline script
├── submission.csv         # Generated predictions file
├── plots/                 # Saved visualization graphs
│   ├── survival_by_sex.png
│   ├── survival_by_pclass.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
└── README.md
```

## ⚙️ Features & Preprocessing
1. **Feature Engineering**:
   - `Title` extracted from Name: Mr, Miss, Mrs, Rare
   - `FamilySize` = SibSp + Parch + 1
   - `IsAlone` flag for solo travelers

2. **Missing Value Handling**:
   - `Age`: Filled with median age by Title
   - `Embarked`: Filled with mode
   - `Fare`: Filled with median fare by Pclass
   - `Cabin`: Dropped due to 77% missing

3. **Encoding**: Label Encoding for Sex, Embarked, Title

## 📈 EDA Insights
| Plot                    | Key Finding                                                       |
|-------------------------|-------------------------------------------------------------------|
| **Survival by Sex**     | Females 74% survival vs Males 19%. Sex is the strongest predictor |
| **Survival by Pclass**  | 1st class 63% vs 3rd class 24%. Wealth = Higher survival chance   |
| **Correlation Heatmap** | Sex, Pclass, Fare have highest correlation with Survived          |

All graphs are auto-saved to `/plots` folder when script runs.

## 🤖 Model Performance
| Model               | Validation Accuracy |
|---------------------|---------------------|
| Logistic Regression | ~80.4%              |
| Decision Tree       | ~81.0%              |
| **Random Forest**   | **~82.1%** 🏆       |

**Best Model**: Random Forest with 82.1% accuracy

**Final Evaluation Metrics**:
- **Accuracy**: 82.1%
- **Precision**: 80.5%
- **Recall**: 75.3%
- **F1-Score**: 77.8%
- **ROC-AUC**: 0.86

## process:
```bash
*Install dependencies*:
   pip install pandas numpy scikit-learn matplotlib seaborn

*Download dataset* from Kaggle Titanic Competition and place `train.csv`, `test.csv` in root folder

*Run the script*:
   python titanic_model.py

*Output*: 
   - `submission.csv` generated for Kaggle submission
   - All plots saved in `/plots` folder
```

## 📝 Key Learnings
1. Feature engineering like `Title` and `FamilySize` improves model accuracy by 3-4%
2. Random Forest handles non-linear relationships better than Logistic Regression
3. `Sex` and `Pclass` are the most important features for survival prediction
4. Threshold tuning using ROC curve helps balance Precision vs Recall

## 🔮 Future Improvements
- Hyperparameter tuning with GridSearchCV
- Try XGBoost / LightGBM for better accuracy
- Build Flask/Gradio web app for real-time predictions
- Add SHAP values for model explainability