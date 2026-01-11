# Fraud Detection & Anomaly Detection

Anomaly detection and fraud classification for financial transactions. Addresses class imbalance with SMOTE, Isolation Forest, and ensemble methods.

## The Challenge

Fraud detection is difficult because fraudulent transactions are rare (typically 0.1-1% of all transactions). A model predicting "not fraud" for everything achieves 99% accuracy but catches zero fraud.

## Approaches

### Supervised Learning (with Labels)

When you have labeled fraud/not-fraud data:

- **Logistic Regression**: Interpretable baseline
- **Random Forest**: Handles non-linear relationships
- **XGBoost/LightGBM**: State-of-the-art for tabular data
- **Neural Networks**: For complex patterns with sufficient data

### Unsupervised Anomaly Detection (without Labels)

When labels are unavailable:

- **Isolation Forest**: Isolates anomalies by random partitioning
- **One-Class SVM**: Learns boundary around normal data
- **Autoencoders**: High reconstruction error indicates anomaly

## Handling Class Imbalance

```python
# SMOTE Oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Class weights in XGBoost
model = XGBClassifier(scale_pos_weight=100)

# Undersampling majority class
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_train, y_train)
```

## Metrics for Imbalanced Data

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| Precision | TP / (TP + FP) | Of predicted fraud, how many were actually fraud? |
| Recall | TP / (TP + FN) | Of actual fraud, how many did we catch? |
| F1 Score | 2 × (P × R) / (P + R) | Balance of precision and recall |
| PR-AUC | Area under PR curve | Performance across thresholds |

## Feature Engineering

### Transaction-Level
- Transaction amount (unusual values)
- Time of transaction (unusual hours)
- Location mismatch with history
- Merchant category risk

### Behavioral
- Transaction velocity (count in last hour/day)
- Amount deviation from typical spending
- New merchant or location
- Failed attempt history

### Network Features
- Device fingerprinting
- IP reputation
- Connection to known fraud rings

## Usage

```python
from sklearn.ensemble import IsolationForest

# Isolation Forest for anomaly detection
iso_forest = IsolationForest(
    contamination=0.01,  # expect 1% anomalies
    random_state=42,
    n_estimators=100
)

# Fit and predict (-1 for anomaly, 1 for normal)
predictions = iso_forest.fit_predict(X)
anomalies = X[predictions == -1]
```

## Tech Stack

- Python
- Scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- Pandas / NumPy

## Ethical Considerations

- **False positives**: Blocking legitimate transactions frustrates customers
- **Bias**: Models may discriminate based on biased historical data
- **Explainability**: Regulations may require explaining flagged transactions
- **Privacy**: Balance security with user privacy in behavioral tracking

## Data

Uses the [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle - anonymized European cardholder transactions with 492 frauds out of 284,807 transactions.

## References

- [imbalanced-learn Documentation](https://imbalanced-learn.org/)
- SMOTE: Chawla et al. (2002)
- Isolation Forest: Liu et al. (2008)

## License

MIT
