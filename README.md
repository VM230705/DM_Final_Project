### Model Performance
#### Before handling text features: 
1. Logistic Regression:
   - Cross-Validation Scores: [0.863, 0.871, 0.857, 0.855, 0.865]
   - Cross-Validated Accuracy: 0.86 +/- 0.01
   - Validation Accuracy: 0.85
   - Micro Average AUC (Validation): 0.97
   - Micro Average AUC (Test): 0.98
   - Test Accuracy: 0.87

2. Support Vector Machine (SVM):
   - Cross-Validation Scores: [0.871, 0.868, 0.875, 0.853, 0.873]
   - Cross-Validated Accuracy: 0.87 +/- 0.01
   - Validation Accuracy: 0.86
   - Micro Average AUC (Validation): 0.96
   - Micro Average AUC (Test): 0.96
   - Test Accuracy: 0.88

3. Decision Tree:
   - Cross-Validation Scores: [0.873, 0.865, 0.861, 0.863, 0.860]
   - Cross-Validated Accuracy: 0.86 +/- 0.00
   - Validation Accuracy: 0.86
   - Micro Average AUC (Validation): 0.92
   - Micro Average AUC (Test): 0.92
   - Test Accuracy: 0.86

4. Random Forest:
   - Cross-Validation Scores: [0.871, 0.867, 0.870, 0.860, 0.861]
   - Cross-Validated Accuracy: 0.87 +/- 0.00
   - Validation Accuracy: 0.86
   - Micro Average AUC (Validation): 0.98
   - Micro Average AUC (Test): 0.98
   - Test Accuracy: 0.87

#### After handling text features(snippet and title):
1. Logistic Regression:
   - Cross-Validation Scores: [0.898, 0.894, 0.901, 0.900, 0.907]
   - Cross-Validated Accuracy: 0.90 +/- 0.00
   - Validation Accuracy: 0.90
   - Micro Average AUC (Validation): 0.99
   - Micro Average AUC (Test): 0.99
   - Test Accuracy: 0.91

2. Support Vector Machine (SVM):
   - Cross-Validation Scores: [0.906, 0.913, 0.912, 0.908, 0.914]
   - Cross-Validated Accuracy: 0.91 +/- 0.00
   - Validation Accuracy: 0.91
   - Micro Average AUC (Validation): 0.99
   - Micro Average AUC (Test): 0.99
   - Test Accuracy: 0.92

3. Decision Tree:
   - Cross-Validation Scores: [0.882, 0.885, 0.881, 0.886, 0.886]
   - Cross-Validated Accuracy: 0.88 +/- 0.00
   - Validation Accuracy: 0.88
   - Micro Average AUC (Validation): 0.91
   - Micro Average AUC (Test): 0.91
   - Test Accuracy: 0.88

4. Random Forest:
   - Cross-Validation Scores: [0.894, 0.900, 0.905, 0.908, 0.906]
   - Cross-Validated Accuracy: 0.90 +/- 0.01
   - Validation Accuracy: 0.90
   - Micro Average AUC (Validation): 0.99
   - Micro Average AUC (Test): 0.99
   - Test Accuracy: 0.90

#### Only use snippet as feature:
1. SVM 
   - Text Classification Summary:

   - **Overall Accuracy:** 0.88
   - Classification Report:
  
    |           | Precision | Recall | F1-Score | Support |
    |-----------|-----------|--------|----------|---------|
    | High      | 0.86      | 0.87   | 0.86     | 663     |
    | Low       | 0.86      | 0.82   | 0.84     | 599     |
    | Medium    | 0.91      | 0.92   | 0.92     | 1103    |

   - Accuracy: 0.88
   - Macro Average: Precision 0.88, Recall 0.87, F1-Score 0.87
   - Weighted Average: Precision 0.88, Recall 0.88, F1-Score 0.88

   - Confusion Matrix:

    |           | Predicted High | Predicted Low | Predicted Medium |
    |-----------|----------------|---------------|------------------|
    | **Actual High**     | 577            | 45            | 41               |
    | **Actual Low**      | 48             | 493           | 58               |
    | **Actual Medium**   | 49             | 35            | 1019             |

### Observations:
1.	Company Feature Impact:
The 'company' feature has a substantial positive impact on Logistic Regression, Decision Tree, and Random Forest models, resulting in significant accuracy improvements.
The inclusion of 'company' provides valuable insights into salary predictions, particularly enhancing the performance of models that struggled initially.
2.	SVM Sensitivity to Feature Scale:
SVM's initial poor performance was attributed to sensitivity to the scale of numerical features.
Standardizing numerical features using StandardScaler considerably improved SVM's accuracy, demonstrating the critical role of feature scaling in SVM models.
