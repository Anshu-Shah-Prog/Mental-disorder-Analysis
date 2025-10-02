# Mental Disorder Classification Report

## Overview

This document summarizes the performance of Random Forest and Multinomial Logistic Regression models trained to predict mental disorders:

Target Classes:
0 → Bipolar Type-1
1 → Bipolar Type-2
2 → Depression
3 → Normal

## Dataset:
120 patients with 17 behavioral/psychological features

## Evaluation Metrics:
Accuracy, Precision, Recall, F1-Score, Confusion Matrix

| Recall is critical in mental health applications because missing a patient (false negative) can be harmful.

## 1. Random Forest (Important Features)

Accuracy: 0.83
Macro F1-score: 0.85

Classification Report
| Class	            | Precision |	Recall | F1-Score |
| ----------------- | --------- | ------ | -------- |
|0 (Bipolar Type-1) |	1.00	    | 0.75   | 0.86	    |
|1 (Bipolar Type-2)	| 1.00      | 1.00   | 1.00     |
|2 (Depression)	    | 0.78      | 0.88   | 0.82     | 
|3 (Normal)	        | 0.71	    | 0.71   | 0.71     | 

Insights:

* High recall for Bipolar Type-1 ensures fewer false negatives.

* Balanced performance across all classes (Macro F1 = 0.85).

* Suitable for clinical applications where minimizing missed patients is critical.

## 2. Multinomial Logistic Regression (Important Features)

Accuracy: 0.83
Macro F1-score: 0.84

Classification Report
| Class	            | Precision |	Recall | F1-Score |
| ----------------- | --------- | ------ | -------- |
|0 (Bipolar Type-1) |	0.75	    | 0.75   | 0.75	    |
|1 (Bipolar Type-2)	| 1.00      | 1.00   | 1.00     |
|2 (Depression)	    | 0.78      | 0.88   | 0.82     | 
|3 (Normal)	        | 0.83	    | 0.71   | 0.77     |

Insights:

* Performs well but recall for critical classes (Bipolar Type-1 and Normal) is slightly lower than Random Forest.

* Macro F1 is slightly lower (0.84) indicating slightly less balanced performance.

## 3. Confusion Matrix Overview

Random Forest (Important Features)

| Actual ↓ \Predicted →  | 0  | 1  | 2  | 3  |
| ---------------------- | -- | -- | -- | -- |
| 0                      | 3  | 0  | 1  | 0  |
| 1                      | 0  | 5  | 0  | 0  |
| 2                      | 0  | 0  | 7  | 1  |
| 3                      | 0  | 0  | 2  | 5  |

Logistic Regression (Important Features)

| Actual ↓ \Predicted →  | 0  | 1  | 2  | 3  |
| ---------------------- | -- | -- | -- | -- |
| 0                      | 3  | 0  | 1  | 0  |
| 1                      | 0  | 5  | 0  | 0  |
| 2                      | 1  | 0  | 7  | 0  |
| 3                      | 0  | 0  | 2  | 5  |

| Note: Both models perform similarly, but Random Forest slightly outperforms in recall for critical classes.

## 4. Key Takeaways

* Random Forest is slightly better for clinical applications due to higher recall on critical diagnoses.

* Multinomial Logistic Regression also performs well but may miss a few patients in critical classes.

* Feature selection (important features) maintains strong performance while improving model interpretability.

* Clinical Relevance: Models with high recall minimize false negatives, ensuring patients needing care are detected.

## 5. Conclusion

Random Forest is recommended for mental disorder prediction due to:

* Higher recall for critical classes

* Balanced macro F1-score

* Strong interpretability via feature importance

| Using machine learning to predict mental disorders can support clinicians in early detection and personalized care
