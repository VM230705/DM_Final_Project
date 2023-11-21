1.	Logistic Regression:
    - Initially, Logistic Regression achieved moderate accuracy (around 50%) before the inclusion of the 'company' feature.
    - After adding the 'company' feature, accuracy significantly improved to 72%, highlighting the impact of this feature.
    - Further improvement (86%) was observed after addressing the sensitivity of SVM to feature scale using StandardScaler.
    - Micro and macro-average AUC scores remain consistently high (around 0.98), showcasing strong discriminatory power.
2.	SVM (Support Vector Machine):
    - SVM initially demonstrated poor performance (around 40%) before the inclusion of the 'company' feature.
    - Even after adding the 'company' feature, accuracy remained relatively low.
    - Sensitivity to feature scale was identified as a key factor affecting SVM's performance.
    - After standardizing numerical features with StandardScaler, SVM's accuracy improved to a competitive level (around 88%).
    - Micro and macro-average AUC scores are high (around 0.96), indicating robust discriminatory ability.
3.	Decision Tree:
    - Decision Tree accuracy improved from 69% to 86% after adding the 'company' feature.
    - Consistent and balanced precision, recall, and F1-scores across salary classes.
    - Micro and macro-average AUC scores are high (around 0.92), demonstrating reliable discriminatory power.
4.	Random Forest:
    - Similar to Decision Tree, Random Forest experienced a notable improvement in accuracy (from 69% to 86%) with the inclusion of the 'company' feature.
    - Precision, recall, and F1-scores exhibit balanced performance.
    - Micro and macro-average AUC scores are notably high (around 0.98), signifying excellent discriminatory ability.

### Observations:
1.	Company Feature Impact:
The 'company' feature has a substantial positive impact on Logistic Regression, Decision Tree, and Random Forest models, resulting in significant accuracy improvements.
The inclusion of 'company' provides valuable insights into salary predictions, particularly enhancing the performance of models that struggled initially.
2.	SVM Sensitivity to Feature Scale:
SVM's initial poor performance was attributed to sensitivity to the scale of numerical features.
Standardizing numerical features using StandardScaler considerably improved SVM's accuracy, demonstrating the critical role of feature scaling in SVM models.
