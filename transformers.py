from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce
from transformers import *

# ── 1. Temporal Features ──────────────────────────────────────────
class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['TransactionHour'] = (X['TransactionDT'] // 3600) % 24
        X['TransactionDay']  = (X['TransactionDT'] // 86400)
        X['TransactionWeek'] = (X['TransactionDT'] // (86400 * 7))
        X.drop(columns=['TransactionDT'], inplace=True)
        return X

# ── 2. Log Transformations ────────────────────────────────────────
class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = np.log1p(X[col])
        return X

# ── 3. Missing Indicators ─────────────────────────────────────────
class MissingIndicator(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[f'{col}_is_missing'] = X[col].isnull().astype(int)
        return X

# ── 4. addr2 Dominant Region Flag ────────────────────────────────
class DominantRegionFlag(BaseEstimator, TransformerMixin):
    def __init__(self, col='addr2', dominant_region=87):
        self.col = col
        self.dominant_region = dominant_region

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['is_dominant_region'] = (X[self.col] == self.dominant_region).astype(int)
        X.drop(columns=[self.col], inplace=True)
        return X

# ── 5. FillNa for Categorical Columns ────────────────────────────
class CategoricalFiller(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].fillna('Missing').astype(str)
        return X

# ── 6. Target Encoder ─────────────────────────────────────────────
class CustomTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, smoothing=10):
        self.cols = cols
        self.smoothing = smoothing
        self.encoder = None

    def fit(self, X, y):
        self.encoder = ce.TargetEncoder(cols=self.cols, smoothing=self.smoothing)
        self.encoder.fit(X[self.cols], y)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols] = self.encoder.transform(X[self.cols])
        return X

# Smoothing in target encoding:
# Without smoothing: a card1 value with 2 transactions, both fraud → 100% fraud rate
# This would massively overfit on rare categories
# 
# With smoothing=10: 
# encoded_value = (n * category_rate + smoothing * global_rate) / (n + smoothing)
# where n = number of times this category appears in train
#
# For rare categories (small n): pulls toward global fraud rate (3.5%)
# For common categories (large n): trusts the observed category rate
# smoothing=10 is a standard starting point

# ── 7. One-Hot Encoder ────────────────────────────────────────────
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.train_columns = None

    def fit(self, X, y=None):
        X = X.copy()
        # Convert categorical columns to strings (Necessary for columns like id_32)
        X[self.cols] = X[self.cols].astype(str)
        
        # Remember columns from training so val can be aligned
        temp = pd.get_dummies(X, columns=self.cols, drop_first=False, dtype=int)
        self.train_columns = temp.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        # Convert categorical columns to strings
        X[self.cols] = X[self.cols].astype(str)
        
        X = pd.get_dummies(X, columns=self.cols, drop_first=False, dtype=int)
        # Align to training columns
        X = X.reindex(columns=self.train_columns, fill_value=0)
        return X

# - drop_first=False — normally you'd use drop_first=True to avoid multicollinearity. 
# But for tree-based models (LightGBM/XGBoost) this doesn't matter — trees don't suffer from multicollinearity. 

# Reindex — this is critical. 
# If val set never sees a particular category (e.g. a rare DeviceInfo group), get_dummies won't create that column in val. 
# Reindex fills it with 0, ensuring train and val have identical column sets.

# ── 8. Categorical Grouping ───────────────────────────────────────
class CategoricalGrouper(BaseEstimator, TransformerMixin):
    # Groups high-cardinality string columns into families
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if 'DeviceInfo' in X.columns:
            X['DeviceInfo'] = X['DeviceInfo'].apply(self._categorise_device)
        if 'P_emaildomain' in X.columns:
            X['P_emaildomain'] = X['P_emaildomain'].apply(self._categorise_email)
        if 'R_emaildomain' in X.columns:
            X['R_emaildomain'] = X['R_emaildomain'].apply(self._categorise_email)
        if 'id_30' in X.columns:
            X['id_30'] = X['id_30'].apply(self._categorise_os)
        if 'id_31' in X.columns:
            X['id_31'] = X['id_31'].apply(self._categorise_browser)
        if 'id_33' in X.columns:
            X['id_33'] = X['id_33'].apply(self._categorise_resolution)
        
        return X

    @staticmethod
    def _categorise_device(device):
        if pd.isnull(device): return 'Missing'
        device = str(device).lower()
        if 'windows' in device: return 'Windows'
        if 'ios' in device or 'iphone' in device: return 'iOS'
        if 'macos' in device or 'mac os' in device: return 'MacOS'
        if 'trident' in device or 'rv:11' in device: return 'IE_Trident'
        if device.startswith('rv:'): return 'Firefox'
        if 'sm-' in device or 'samsung' in device: return 'Samsung_Android'
        if 'huawei' in device or 'ale-' in device: return 'Huawei_Android'
        if 'moto' in device: return 'Motorola_Android'
        if 'lg-' in device: return 'LG_Android'
        if 'android' in device: return 'Other_Android'
        if 'linux' in device: return 'Linux'
        return 'Other'
    
    @staticmethod
    def _categorise_email(domain):
        if pd.isnull(domain): return 'Missing'
        domain = str(domain).lower()
        if any(x in domain for x in ['gmail', 'googlemail']): return 'Google'
        if any(x in domain for x in ['yahoo', 'ymail']): return 'Yahoo'
        if any(x in domain for x in ['hotmail', 'outlook', 'live', 'msn']): return 'Microsoft'
        if any(x in domain for x in ['icloud', 'mac.com', 'me.com']): return 'Apple'
        if any(x in domain for x in ['aol', 'aim']): return 'AOL'
        if any(x in domain for x in ['.es', '.fr', '.mx', '.de', '.jp']): return 'Foreign'
        if domain == 'anonymous.com': return 'Anonymous'
        return 'Other'

    @staticmethod
    def _categorise_os(os):
        if pd.isnull(os) or str(os).strip() == 'Missing': return 'Missing'
        os = str(os).lower()
        if 'windows' in os: return 'Windows'
        if 'ios' in os: return 'iOS'
        if 'android' in os: return 'Android'
        if 'mac' in os: return 'MacOS'
        if 'linux' in os: return 'Linux'
        return 'Other'
    
    @staticmethod
    def _categorise_browser(browser):
        if pd.isnull(browser) or str(browser).strip() == 'Missing': return 'Missing'
        browser = str(browser).lower()
        if 'samsung' in browser: return 'Samsung_Browser'
        if 'android' in browser and 'chrome' in browser: return 'Chrome_Android'
        if 'chrome' in browser: return 'Chrome_Desktop'
        if 'mobile safari' in browser: return 'Mobile_Safari'
        if 'safari' in browser: return 'Safari_Desktop'
        if 'firefox' in browser: return 'Firefox'
        if 'edge' in browser: return 'Edge'
        if 'ie' in browser: return 'IE'
        return 'Other'
    
    @staticmethod
    def _categorise_resolution(res):
        if pd.isnull(res) or str(res).strip() == 'Missing': return 'Missing'
        try:
            w, h = int(res.split('x')[0]), int(res.split('x')[1])
            if w > 2000: return 'HighRes'
            if w >= 1280 and h >= 720: return 'Desktop'
            if w < 1280: return 'Mobile'
        except:
            pass
        return 'Other'

# ── 9. V Feature Selector ────────────────────────────────────────
class VFeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.selected_features_ = None
    
    def fit(self, X, y):
        v_cols = [c for c in X.columns if c.startswith('V')]
        
        # Correlation with target
        target_corr = X[v_cols].corrwith(y).abs()
        
        # Sort by target correlation descending
        remaining = target_corr.sort_values(ascending=False).index.tolist()
        
        # Greedy selection
        selected = []
        v_corr_matrix = X[v_cols].corr().abs()
        
        for feature in remaining:
            if not selected:
                selected.append(feature)
            else:
                max_corr = v_corr_matrix.loc[feature, selected].max()
                if max_corr < self.threshold:
                    selected.append(feature)
        
        self.selected_features_ = selected
        print(f"V features selected: {len(selected)} from {len(v_cols)}")
        return self
    
    def transform(self, X):
        X = X.copy()
        v_cols = [c for c in X.columns if c.startswith('V')]
        drop_cols = [c for c in v_cols if c not in self.selected_features_]
        X.drop(columns=drop_cols, inplace=True)
        return X