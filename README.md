# ADBoost-Classifier-From-Scratch
"ML models implemented from scratch using NumPy and Pandas only"
📘 What is AdaBoost?

AdaBoost (Adaptive Boosting) is an ensemble learning algorithm that combines multiple weak learners (e.g., decision stumps) to form a strong classifier.

Each weak learner focuses on previous mistakes by adjusting sample weights.

Learners with low error get higher importance (alpha) in the final model.

The final model is a weighted majority vote of all learners.



---

🔑 Key Formulas

1. Weighted Error Rate

$$\varepsilon_t = \frac{\sum_{i=1}^{N} w_i \cdot I(y_i \neq h_t(x_i))}{\sum_{i=1}^{N} w_i}$$

2. Learner’s Weight (Alpha)

$$\alpha_t = \frac{1}{2} \ln \left( \frac{1 - \varepsilon_t}{\varepsilon_t} \right)$$

3. Update Rule for Sample Weights

$$w_i \leftarrow w_i \cdot e^{\alpha_t \cdot I(y_i \neq h_t(x_i))}$$

4. Normalization of Weights

$$w_i \leftarrow \frac{w_i}{\sum_{j=1}^{N} w_j}$$

5. Final Strong Classifier

$$H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t \cdot h_t(x)\right)$$


---

⚙️ Implementation Details

Dataset: Titanic: Machine Learning from Disaster (Kaggle)

Preprocessing:

Selected features: Pclass, Sex, Age, Fare, Embarked

Missing values handled (median for numerics, mode for categorical)

One-hot encoding for categorical variables


Weak Learners: Decision stumps (single-split trees)

Evaluation: Accuracy, Precision, Recall, F1-score, AUC



---

📊 Visualizations

Feature Importances (Top 10)

Training Error vs Boosting Rounds

Validation Error vs Boosting Rounds (Learning Curve)

Margin Distribution

Confusion Matrix Heatmap

ROC Curve with AUC

Precision-Recall Curve



---

🏆 Results

Accuracy: ~79% (validation set)

AUC: ~0.80

Boosting improves performance compared to a single decision stump.




---

📚 References

1. Yoav Freund, Robert E. Schapire. A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting (1997). 🔗 PDF


2. Schapire, R.E. Explaining AdaBoost. 🔗 PDF


3. Hastie, Tibshirani, Friedman. The Elements of Statistical Learning. 🔗 Book


4. Scikit-learn Documentation: AdaBoost Classifier


5. StatQuest: AdaBoost Clearly Explained (YouTube)


6. Stanford CS229 Notes: Boosting




---

🚀 Future Work

Implement AdaBoost with depth-2 CART trees instead of stumps.

Compare AdaBoost with Gradient Boosting and XGBoost on the same dataset.

Try other datasets (spam detection, image recognition).



---

✨ This project helps you understand boosting by building AdaBoost step by step from scratch instead of relying on libraries.
