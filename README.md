# Machine Learning Notebooks

Hands-on implementations of the classical machine learning toolkit: linear models, SVMs, decision trees, ensembles, dimensionality reduction and unsupervised learning. Built with Python, NumPy, pandas and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626)

## About

I'm **Md. Dillshadul Islam**, a CSE graduate (BUBT) based in Bangladesh, building toward a remote junior ML engineering role.

This repository is my working record of studying *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron (Chapters 1–9). I worked through each chapter in code, ran every experiment, and extended several of them with my own variations (listed below). The book's datasets and structure are the foundation. The notebooks show that I can implement, run and interpret these methods, not that I invented them.

## What this repo demonstrates

- **End-to-end ML workflow**: data exploration, stratified splitting, preprocessing pipelines, model comparison, hyperparameter search and final test evaluation (California Housing).
- **Understanding beyond `.fit()`**: Normal Equation, SVD/least squares, Batch / Stochastic / Mini-batch Gradient Descent and Ridge regularization written in plain NumPy and checked against scikit-learn.
- **Proper evaluation**: cross-validation, baselines (DummyClassifier), precision/recall trade-offs, ROC/AUC, confusion-matrix error analysis, learning curves, bias/variance diagnosis.
- **Wide model coverage**: linear/logistic/softmax regression, SVM (linear, polynomial, RBF, SVR), decision trees, bagging, random forests, AdaBoost, gradient boosting, stacking.
- **Unsupervised and semi-supervised learning**: PCA variants, manifold learning, K-Means, DBSCAN, Gaussian Mixtures, anomaly detection, label propagation.

## Notebook index

| # | Notebook | Book chapter | Topic |
|---|----------|--------------|-------|
| 1 | [`lifeSatisfaction.ipynb`](lifeSatisfaction.ipynb) | Ch. 1 | Model-based (linear regression) vs instance-based (k-NN) learning |
| 2 | [`california_housing.ipynb`](california_housing.ipynb) | Ch. 2 | Full end-to-end regression project |
| 3 | [`classfication.ipynb`](classfication.ipynb) | Ch. 3 | MNIST classification, metrics, error analysis |
| 4 | [`linear_model_training_principles.ipynb`](linear_model_training_principles.ipynb) | Ch. 4 | How linear models are trained, regularization, logistic/softmax |
| 5 | [`SVM.ipynb`](SVM.ipynb) | Ch. 5 | Support Vector Machines for classification and regression |
| 6 | [`decision_trees.ipynb`](decision_trees.ipynb) | Ch. 6 | Decision trees, regularization, sensitivity to rotation |
| 7 | [`ensemble_learning_and_random_forest.ipynb`](ensemble_learning_and_random_forest.ipynb) | Ch. 7 | Voting, bagging, random forests, boosting, stacking |
| 8 | [`Dimensionality_Reduction.ipynb`](Dimensionality_Reduction.ipynb) | Ch. 8 | PCA, random projection, manifold learning |
| 9 | [`unsupervised_learning.ipynb`](unsupervised_learning.ipynb) | Ch. 9 | Clustering, semi-supervised learning, Gaussian mixtures, anomaly detection |

## Results at a glance

| Task | Model / setup | Result |
|------|---------------|--------|
| California Housing (regression) | Linear regression, training RMSE | ≈ $68,973 (underfits) |
| | Decision tree: training vs 10-fold CV RMSE | $0 vs ≈ $66,551 (clear overfit) |
| | Random Forest + `RandomizedSearchCV` (3-fold CV RMSE) | ≈ $42,250 (best: 45 geo-clusters, `max_features=9`) |
| | Final Random Forest on held-out test set | RMSE ≈ $41,446 |
| MNIST "5 detector" | `DummyClassifier` baseline accuracy | 90.97% (shows why accuracy alone misleads) |
| | SGD classifier: CV accuracy / ROC AUC | ≈ 95–96% / 0.960 |
| | SGD at the 90%-precision threshold | recall drops to 48% |
| | Random Forest at threshold 0.44 | precision 0.981, recall 0.906, F1 0.942 |
| MNIST multilabel (≥7 and odd) | k-NN, macro F1 | 0.976 |
| Moons classification | Voting (hard / soft), Bagging, Stacking test accuracy | 0.904 / 0.920, 0.920, 0.928 |
| | Decision tree: unregularized vs `min_samples_leaf=5` | 90.1% vs 91.8% |
| MNIST dimensionality | PCA components needed for 95% variance | 154 of 784 |
| Semi-supervised digits | 50 labeled samples → 50 cluster representatives → label propagation | 75.8% → 81.9% → 87.4% |
| Clustering | K-Means on 5 blobs: silhouette score | 0.627 |

## Notebook details

### 1. `lifeSatisfaction.ipynb`: Linear regression vs k-NN
Fits `LinearRegression` and `KNeighborsRegressor(k=3)` on GDP-per-capita vs life-satisfaction data and compares predictions for the same new country (6.30 vs 6.33).

### 2. `california_housing.ipynb`: End-to-end regression project
- Exploratory analysis: histograms, geographic scatter plots, correlation matrix (median income is the strongest predictor, r ≈ 0.69).
- Income-category **stratified split**, compared against a purely random split.
- Feature engineering: rooms per house, bedroom ratio, people per house.
- Preprocessing with `Pipeline` and `ColumnTransformer`: median imputation, scaling, log transforms, ratio features, one-hot encoding, and a KMeans-based geographic-similarity transformer.
- Compared Linear Regression, Decision Tree and Random Forest using cross-validation, then tuned with `RandomizedSearchCV`.
- Feature-importance analysis, plus **my own experiment**: dropping low-importance features and retraining. It gave no improvement (test RMSE ≈ 42.4k vs ≈ 41.4k with all features), so I kept the full feature set.

### 3. `classfication.ipynb`: MNIST classification
- Binary "5 detector" with SGD and Random Forest, evaluated with CV, confusion matrix, precision/recall curves, threshold selection and ROC/AUC.
- Multiclass strategies: One-vs-Rest, One-vs-One, SVC, SGD with feature scaling.
- **My own work**: a hand-built One-vs-One classifier (45 pairwise Random Forests with probability voting).
- Error analysis with normalized and error-weighted confusion matrices; 3-vs-5 confusion study.
- Multilabel (`KNeighborsClassifier`), `ClassifierChain`, and multioutput **image denoising** with k-NN.

### 4. `linear_model_training_principles.ipynb`: Training principles
- **Normal Equation**, **SVD / `lstsq`**, and **Batch, Stochastic and Mini-batch Gradient Descent** implemented from scratch in NumPy. All recover θ ≈ (4.21, 2.77) on synthetic data generated from (4, 3), matching scikit-learn.
- Learning-rate and learning-schedule experiments; `SGDRegressor`, including `partial_fit`.
- Polynomial regression, `PolynomialFeatures` feature expansion, learning curves (underfitting vs overfitting with a degree-10 model).
- Regularization: Ridge (closed form, gradient descent and SGD), Lasso, Elastic Net with CV variants, early stopping.
- Logistic regression and Softmax regression on Iris, including decision boundaries and class probabilities.

### 5. `SVM.ipynb`: Support Vector Machines
Linear SVM with scaling, polynomial features + `LinearSVC`, polynomial kernel and Gaussian RBF kernel on the moons dataset (effect of `degree`, `coef0`, `gamma`, `C`), and SVM **regression** (`LinearSVR`, kernelized `SVR`, effect of ε and C).

### 6. `decision_trees.ipynb`: Decision trees
Training and visualizing trees on Iris (Graphviz export, class probabilities), regularization with `max_depth` and `min_samples_leaf`, `DecisionTreeRegressor` on noisy quadratic data, and a PCA-rotation experiment on tree sensitivity to feature orientation.

### 7. `ensemble_learning_and_random_forest.ipynb`: Ensembles
Hard vs soft voting (with the soft-vote averaging worked through by hand for one sample), bagging with out-of-bag evaluation, Random Forest and feature importances (Iris, and a pixel-importance heatmap for MNIST), AdaBoost, Gradient Boosting **built manually by fitting trees to residuals**, `GradientBoostingRegressor` with early stopping (stopped at 102 trees), `HistGradientBoostingRegressor` on the housing data (CV RMSE ≈ $47.6k), and Stacking.

### 8. `Dimensionality_Reduction.ipynb`: Dimensionality reduction
PCA computed manually via SVD and with scikit-learn, choosing the number of components by explained variance (154 for 95% on MNIST), PCA + Random Forest tuned jointly with `RandomizedSearchCV`, `IncrementalPCA` with memory-mapped arrays, Gaussian and sparse random projection (Johnson–Lindenstrauss bound, sparse projection matrix built by hand), and manifold learning (LLE, MDS, Isomap, t-SNE) on a Swiss roll.

### 9. `unsupervised_learning.ipynb`: Unsupervised learning
K-Means and Mini-batch K-Means (inertia, elbow method, silhouette analysis), **image segmentation** by color quantization, **semi-supervised learning** (representative samples, label propagation, outlier filtering; 96.1% propagation accuracy on the retained samples), DBSCAN (plus a k-NN classifier on top and distance-based anomaly flagging), Gaussian Mixture Models (covariance types, sampling, density-based **anomaly detection**, BIC/AIC model selection) and Bayesian Gaussian Mixtures.

## Skills and tools

**Libraries:** NumPy, pandas, SciPy, scikit-learn, Matplotlib, Graphviz, Pillow
**Practices:** reproducible splits (`random_state`), scikit-learn pipelines, cross-validation, hyperparameter search, metric selection for imbalanced problems, error analysis, regularization, reading learning curves

**Not covered in this repo (yet):** neural networks / deep learning, NLP, model deployment.

## Running the notebooks

The notebooks were executed on Google Colab (Python 3.12). Each one can be opened there:

`https://colab.research.google.com/github/19mddill/Machine_Learning_Notebooks/blob/main/<notebook_name>.ipynb`

To run locally:

```bash
git clone https://github.com/19mddill/Machine_Learning_Notebooks.git
cd Machine_Learning_Notebooks
pip install numpy pandas scipy scikit-learn matplotlib pillow graphviz jupyter
jupyter notebook
```

Notes:
- MNIST (`fetch_openml`) and the California Housing data are downloaded on first run, so an internet connection is needed.
- Rendering the trees in `decision_trees.ipynb` also needs the Graphviz system binary (`apt install graphviz` / `brew install graphviz`).
- `unsupervised_learning.ipynb` loads `images/unsupervised_learning/ladybug.png`; place the image at that path before running the segmentation section.
- A few cells (K-Means timing loop, MNIST random forests) take several minutes.

## Credits

Structure, datasets and many code patterns follow Aurélien Géron's *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* and its [companion code](https://github.com/ageron/handson-ml3). Experiments marked as "my own" above are my additions.

