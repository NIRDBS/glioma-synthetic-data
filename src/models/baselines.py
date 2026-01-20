import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, ClassifierMixin

class ApolloPipeline(BaseEstimator, ClassifierMixin):
    """
    Exact reproduction of the 'Apollo' methodology:
    1. Random Forest for Feature Selection (Top 20 bands).
    2. SVM (RBF Kernel) for Classification on those 20 bands.
    """
    def __init__(self, n_features=20, random_state=42):
        self.n_features = n_features
        self.random_state = random_state
        self.selected_indices_ = None
        
        # Configuration extracted strictly from '4 Six class...ipynb'
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
        
        self.svm_classifier = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            decision_function_shape='ovo',
            probability=False,
            random_state=self.random_state
        )
        
        self.selector_model = None

    def fit(self, X, y):
        """
        Fits RF, selects top features, then fits SVM on reduced data.
        """
        # 1. Fit Random Forest
        print(f"  [Apollo] Fitting RF Selector on {X.shape}...")
        self.rf_selector.fit(X, y)
        
        # 2. Select Top K Features
        # Apollo forced exactly 20 features using threshold=-inf
        self.selector_model = SelectFromModel(
            self.rf_selector, 
            prefit=True, 
            max_features=self.n_features,
            threshold=-np.inf
        )
        
        self.selected_indices_ = self.selector_model.get_support(indices=True)
        print(f"  [Apollo] Selected {len(self.selected_indices_)} features.")
        
        # 3. Transform Data
        X_reduced = self.selector_model.transform(X)
        
        # 4. Fit SVM
        print(f"  [Apollo] Fitting SVM Classifier...")
        self.svm_classifier.fit(X_reduced, y)
        
        return self

    def predict(self, X):
        if self.selector_model is None:
            raise RuntimeError("You must train the model before predicting!")
            
        # 1. Transform using selected features
        X_reduced = self.selector_model.transform(X)
        # 2. Predict
        return self.svm_classifier.predict(X_reduced)