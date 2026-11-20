import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class ApolloPipeline(BaseEstimator, ClassifierMixin):
    """
    The RF+SVM classifier of Lita et al. (the 'Apollo' method; used here for the binary task too):
    1. Random Forest for feature selection (top 20 bands).
    2. SVM (RBF kernel) on those 20 bands.

    Hyperparameters follow the authors' published notebook.
    """
    def __init__(self, n_features=20, random_state=42):
        self.n_features = n_features
        self.random_state = random_state
        self.selector_model = None
        self.selected_indices_ = None
        
        # --- 1. Feature Selector: Random Forest ---
        # Parameters matched exactly to Apollo Notebook #4
        self.rf_selector = RandomForestClassifier(
            n_estimators=300,
            criterion="entropy",
            max_depth=10,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=50, 
            min_samples_split=50,
            n_jobs=-1,
            random_state=self.random_state
        )
        
        # --- 2. Classifier: SVM ---
        # Parameters matched exactly to Apollo Notebook #4
        self.svm_classifier = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            decision_function_shape='ovo', # One-vs-One strategy
            probability=True, # Need probabilities for AUC/Metrics
            random_state=self.random_state
        )
        
        # Pipeline logic handles scaling internally before SVM
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """
        Fits RF, selects top features, then fits SVM on reduced data.
        """
        # Step A: Feature Selection
        print(f"  [Apollo] Fitting RF Selector on {X.shape} features...")
        self.rf_selector.fit(X, y)
        
        # Select Top K features (Apollo uses fixed count, usually ~20 for visuals, 
        # but in the notebook they use SelectFromModel with threshold=-inf to pick exactly max_features)
        
        # We use SelectFromModel to grab top K
        self.selector_model = SelectFromModel(
            self.rf_selector, 
            prefit=True, 
            max_features=self.n_features,
            threshold=-np.inf
        )
        
        X_reduced = self.selector_model.transform(X)
        self.selected_indices_ = self.selector_model.get_support(indices=True)
        print(f"  [Apollo] Selected {len(self.selected_indices_)} features.")
        
        # Step B: Classification
        # Scale features for SVM (Critical for RBF kernel)
        X_scaled = self.scaler.fit_transform(X_reduced)
        
        print(f"  [Apollo] Fitting SVM Classifier...")
        self.svm_classifier.fit(X_scaled, y)
        
        return self

    def predict(self, X):
        # 1. Select
        X_reduced = self.selector_model.transform(X)
        # 2. Scale
        X_scaled = self.scaler.transform(X_reduced)
        # 3. Predict
        return self.svm_classifier.predict(X_scaled)

    def predict_proba(self, X):
        # 1. Select
        X_reduced = self.selector_model.transform(X)
        # 2. Scale
        X_scaled = self.scaler.transform(X_reduced)
        # 3. Predict
        return self.svm_classifier.predict_proba(X_scaled)