# scripts/model_definitions.py

import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.ensemble import (
    HistGradientBoostingRegressor, HistGradientBoostingClassifier, 
    RandomForestRegressor, RandomForestClassifier
)
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV, StratifiedShuffleSplit
from scipy.stats import randint, uniform

# =============================================================================
# 1. LIGHT TUNING SEARCH SPACES (Sklearn)
# =============================================================================
HGB_REG_DIST = {
    "learning_rate": uniform(0.05, 0.15),
    "max_leaf_nodes": randint(20, 60),
    "max_depth": [5, 10, None],
    "l2_regularization": uniform(0, 1),
}
RF_REG_DIST = {
    "n_estimators": [50, 100],
    "max_depth": [10, 15, 20],
    "min_samples_leaf": [20, 50, 100],
}
GLM_REG_DIST = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
}
HGB_CLF_DIST = {
    "learning_rate": uniform(0.01, 0.2),
    "max_leaf_nodes": randint(15, 63),
    "max_depth": [5, 10, 15],
    "l2_regularization": uniform(0, 1),
}
RF_CLF_DIST = {
    "n_estimators": [50, 100],
    "max_depth": [10, 15, 20],
    "min_samples_leaf": [20, 50, 100],
}
LOGREG_CLF_DIST = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
}

# =============================================================================
# 2. TORCH DATASETS & HELPERS (MEMORY SAFE)
# =============================================================================

class RamSafeDataset(Dataset):
    """
    Optimized Dataset that avoids copying memory.
    Enforces float32 to prevent scaler bloat.
    """
    def __init__(self, X, y=None):
        if torch.is_tensor(X):
            self.X = X.float()
        elif isinstance(X, np.ndarray):
            self.X = torch.as_tensor(X, dtype=torch.float32)
        else:
            self.X = torch.as_tensor(X.values, dtype=torch.float32)
            
        if y is not None:
            if torch.is_tensor(y):
                self.y = y.float()
            else:
                self.y = torch.as_tensor(y, dtype=torch.float32)
            if self.y.ndim == 1:
                self.y = self.y.unsqueeze(1)
        else:
            self.y = None

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class SequenceDataset(Dataset):
    def __init__(self, X, y, indices, seq_len=4):
        # OPTIMIZATION: Use as_tensor to share memory
        if not torch.is_tensor(X):
            self.X = torch.as_tensor(X, dtype=torch.float32)
        else:
            self.X = X.float()
        
        # Safety Check for X dimensions
        if self.X.ndim < 2:
             raise ValueError(f"SequenceDataset expects 2D input X, got {self.X.shape}")

        if not torch.is_tensor(y):
            self.y = torch.as_tensor(y, dtype=torch.float32)
        else:
            self.y = y.float()

        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(1)

        self.indices = indices
        self.seq_len = seq_len

    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        anchor_idx = self.indices[idx]
        start_idx = anchor_idx - self.seq_len + 1
        
        # Handle padding for start of dataset
        if start_idx < 0:
            seq_x = self.X[0:anchor_idx+1]
            pad_len = self.seq_len - seq_x.size(0)
            if pad_len > 0:
                padding = seq_x[0].unsqueeze(0).repeat(pad_len, 1)
                seq_x = torch.cat([padding, seq_x], dim=0)
        else:
            seq_x = self.X[start_idx:anchor_idx+1]
            
        target_y = self.y[anchor_idx]
        return seq_x, target_y

def predict_with_torch(model, X, batch_size=4096, device=None, is_gru=False):
    """
    Runs inference in batches to avoid OOM on 16GB RAM.
    Batch size kept high (4096) for inference speed, as gradients aren't stored.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)
    preds = []
    
    if hasattr(X, "values"): X = X.values
    
    n_samples = len(X)
    n_batches = int(np.ceil(n_samples / batch_size))
    
    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_samples)
            X_batch = X[start:end]
            X_t = torch.as_tensor(X_batch, dtype=torch.float32).to(device)
            
            if is_gru and X_t.ndim == 2:
                # Direct GRU: (N, F) -> (N, 1, F)
                X_t = X_t.unsqueeze(1)
            
            out = model(X_t)
            preds.append(out.cpu().numpy().flatten())
            
    return np.concatenate(preds)

# =============================================================================
# 3. MODEL CLASSES
# =============================================================================

class RobustMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], dropout=0.2):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            layers.append(nn.BatchNorm1d(h_d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_d = h_d
        layers.append(nn.Linear(in_d, 1))
        layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class RobustGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.act = nn.Softplus()
    def forward(self, x):
        # x shape: (Batch, Seq, Feat)
        out, _ = self.gru(x)
        last_step_out = out[:, -1, :] 
        pred = self.fc(last_step_out)
        return self.act(pred)

class HurdleModel:
    def __init__(self, classifier, regressor, regressor_type="sklearn"):
        self.classifier = classifier
        self.regressor = regressor
        self.regressor_type = regressor_type

    def predict(self, X):
        p_event = self.classifier.predict_proba(X)[:, 1]
        if self.regressor_type == "torch":
            is_gru = isinstance(self.regressor, RobustGRU)
            y_cond = predict_with_torch(self.regressor, X, is_gru=is_gru)
        else:
            y_cond = self.regressor.predict(X)
        return p_event * y_cond

def train_torch_model(model, train_loader, val_loader, epochs=15, patience=5, lr=0.001, device="cpu"):
    """
    Standard PyTorch training loop with Early Stopping.
    UPDATED: Now accepts 'patience' argument for stability control.
    """
    criterion = nn.PoissonNLLLoss(log_input=False)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float("inf")
    patience_counter = 0
    best_state = None
    
    model.to(device)
    
    for ep in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                vpred = model(vx)
                loss = criterion(vpred, vy)
                val_loss += loss.item() * vx.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    if best_state:
        model.load_state_dict(best_state)
    return model

# =============================================================================
# 4. MAIN INTERFACE
# =============================================================================

def get_optimized_model(
    algo_name,
    approach,
    input_dim,
    X_train,
    y_train,
    light_mode=True,
    return_cv_results=False,
):
    approach_norm = str(approach).strip().lower()
    is_direct = approach_norm in ["direct", "direct_regressor", "directregressor"]
    
    # --- A. GATEKEEPERS ---
    if approach_norm in ["classifier", "classification", "gatekeeper", "clf"]:
        if len(X_train) > 300_000:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=300_000, random_state=42)
            y_arr = np.array(y_train)
            for idx, _ in sss.split(X_train, y_arr):
                X_sub, y_sub = X_train[idx], y_arr[idx]
        else:
            X_sub, y_sub = X_train, y_train
        y_bin = (y_sub > 0).astype(int)

        if algo_name == "HGB":
            base = HistGradientBoostingClassifier(random_state=42)
            params = HGB_CLF_DIST
        elif algo_name == "RF":
            base = RandomForestClassifier(n_jobs=1, random_state=42)
            params = RF_CLF_DIST
        elif algo_name == "LogReg":
            base = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)
            params = LOGREG_CLF_DIST
        else:
            raise ValueError(f"Unknown Classifier: {algo_name}")

        # --- FIX: SPARSITY CHECK ---
        # If positives are < 5%, HalvingSearch can fail by picking empty chunks.
        # Use RandomizedSearchCV in that case.
        pos_rate = y_bin.mean()
        is_sparse = pos_rate < 0.05
        
        if is_sparse:
            search = RandomizedSearchCV(
                base, params, n_iter=6 if light_mode else 20, cv=3, n_jobs=1,
                random_state=42, scoring="roc_auc"
            )
        else:
            search = HalvingRandomSearchCV(
                base, params, factor=2, cv=3, n_jobs=1,
                random_state=42, scoring="roc_auc", 
                n_candidates=6 if light_mode else 20,
                min_resources=1000 
            )

        try:
            search.fit(X_sub, y_bin)
            if return_cv_results: return search.best_estimator_, search.cv_results_
            return search.best_estimator_
        except:
            model = base.fit(X_sub, y_bin)
            if return_cv_results: return model, {}
            return model

    # --- B. NEURAL NETWORKS (TORCH) ---
    if algo_name in ["MLP", "GRU"]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Define Search Space (Light vs Heavy)
        if light_mode:
            # Fast, single config
            # UPDATE: Increased epochs/patience for stability with smaller batch size
            if algo_name == "MLP": 
                configs = [{"lr": 0.001, "layers": [32, 16, 8], "dropout": 0.2, "epochs": 12, "patience": 5}]
            else: 
                # GRU: Forced hidden=16
                configs = [{"lr": 0.001, "hidden": 16, "layers": 2, "dropout": 0.2, "epochs": 12, "patience": 5}]
        else:
            # Heavy Mode: Tune Architecture
            # UPDATE: Increased epochs to 30, patience to 10. Forced GRU hidden=16.
            if algo_name == "MLP":
                # Heavy Mode (Multiple Configs)
                configs = [
                    {"lr": 0.001, "layers": [32, 16, 8],  "dropout": 0.2, "epochs": 15, "patience": 6}, # Option 1 (Balanced)
                    {"lr": 0.005, "layers": [32, 16],     "dropout": 0.1, "epochs": 15, "patience": 6}, # Shallow
                    {"lr": 0.001, "layers": [64, 32, 16], "dropout": 0.3, "epochs": 15, "patience": 6}, # The "Big" One (Reduced)
                    {"lr": 0.005, "layers": [16, 8],      "dropout": 0.1, "epochs": 15, "patience": 6}, # Speedster
                ]
            else: # GRU
                # Forced hidden=16 for all heavy candidates to prevent OOM
                configs = [
                    {"lr": 0.001, "hidden": 16, "layers": 2, "dropout": 0.2, "epochs": 30, "patience": 10},
                    {"lr": 0.0005,"hidden": 16, "layers": 2, "dropout": 0.3, "epochs": 30, "patience": 10}, 
                    {"lr": 0.005, "hidden": 16, "layers": 1, "dropout": 0.0, "epochs": 30, "patience": 10}, 
                    {"lr": 0.001, "hidden": 16, "layers": 3, "dropout": 0.3, "epochs": 30, "patience": 10}, 
                ]

        # 2. Prepare Datasets
        if algo_name == "GRU" and not is_direct:
            y_arr = np.array(y_train)
            pos_indices = np.where(y_arr > 0)[0]
            # Must have at least seq_len history
            pos_indices = pos_indices[pos_indices >= 3] 
            
            n_val = int(len(pos_indices) * 0.2)
            np.random.shuffle(pos_indices)
            val_idx = pos_indices[:n_val]; train_idx = pos_indices[n_val:]
            
            train_ds = SequenceDataset(X_train, y_arr, train_idx, seq_len=4)
            val_ds = SequenceDataset(X_train, y_arr, val_idx, seq_len=4)
        else:
            if algo_name == "GRU": # Direct GRU needs (N,1,F)
                X_t = torch.as_tensor(X_train, dtype=torch.float32).unsqueeze(1)
            else:
                X_t = torch.as_tensor(X_train, dtype=torch.float32)

            y_t = torch.as_tensor(y_train, dtype=torch.float32).unsqueeze(1)

            split = int(0.8 * len(X_t))
            train_ds = RamSafeDataset(X_t[:split], y_t[:split])
            val_ds = RamSafeDataset(X_t[split:], y_t[split:])

        # UPDATE: Batch size reduced from 4096 -> 256 for stability
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256)
        
        # 3. Model Selection Loop
        best_loss = float("inf"); best_model = None; best_cfg = None
        tuning_results = []
        
        print(f"   [Deep Learning] Tuning {algo_name} ({len(configs)} configs)...")
        for cfg in configs:
            if algo_name == "MLP": 
                model = RobustMLP(input_dim, hidden_dims=cfg["layers"], dropout=cfg["dropout"])
            else: 
                model = RobustGRU(input_dim, hidden_dim=cfg["hidden"], num_layers=cfg["layers"], dropout=cfg["dropout"])
            
            try:
                # UPDATE: Pass dynamic epochs and patience
                trained = train_torch_model(
                    model, train_loader, val_loader, 
                    epochs=cfg.get("epochs", 15), 
                    patience=cfg.get("patience", 3),
                    lr=cfg["lr"], 
                    device=device
                )
                tuning_results.append({**cfg, "status": "success"})
                best_model = trained; best_cfg = cfg
            except Exception as e:
                print(f"    Config failed: {e}")

        if best_model is None:
            best_model = RobustMLP(input_dim) if algo_name=="MLP" else RobustGRU(input_dim)
            
        best_model._best_cfg = best_cfg
        if return_cv_results: return best_model, tuning_results
        return best_model

    # --- C. STANDARD REGRESSORS ---
    mask = (y_train > 0) if (not is_direct) else np.ones(len(y_train), dtype=bool)
    X_sub, y_sub = X_train[mask], y_train[mask]
    
    # Safety: If mask resulted in zero rows (e.g. C1 with very few crashes), fall back
    if len(X_sub) < 10:
        # Fallback to training on top 50 rows just to return a valid object, 
        # though it will be terrible. Better than crash.
        X_sub, y_sub = X_train[:50], y_train[:50]

    if len(X_sub) > 200_000:
        idx = np.random.choice(len(X_sub), 200_000, replace=False)
        X_sub, y_sub = X_sub[idx], y_sub[idx]

    if algo_name == "HGB":
        base = HistGradientBoostingRegressor(loss="poisson", random_state=42)
        params = HGB_REG_DIST
    elif algo_name == "RF":
        base = RandomForestRegressor(n_jobs=1, random_state=42)
        params = RF_REG_DIST
    elif algo_name == "GLM":
        base = PoissonRegressor(max_iter=500)
        params = GLM_REG_DIST
    else: raise ValueError(f"Unknown Regressor: {algo_name}")
        
    search = HalvingRandomSearchCV(
        base, params, factor=2, cv=3, n_jobs=1,
        random_state=42, scoring="neg_mean_poisson_deviance",
        n_candidates=6 if light_mode else 20
    )
    
    try:
        search.fit(X_sub, y_sub)
        if return_cv_results: return search.best_estimator_, search.cv_results_
        return search.best_estimator_
    except:
        reg = base.fit(X_sub, y_sub)
        if return_cv_results: return reg, {}
        return reg
    
# End of scripts/model_definitions.py