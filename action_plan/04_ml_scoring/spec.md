# Phase 4: ML Scoring

## Objective

Build an XGBoost ensemble model that combines all signals into a calibrated insider probability score. The model learns from historical patterns to improve upon rule-based signal aggregation.

---

## Scope

| Item | Details |
|------|---------|
| Model | XGBoost ensemble classifier |
| Input Features | Rule-based signals from Phase 3 + raw features |
| Output | Calibrated probability score (0-1) |
| Training Data | Hybrid labels (profitability + pattern matching) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ML SCORING PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Signal     │───▶│   Feature    │───▶│   Feature    │       │
│  │   Data       │    │   Engineer   │    │   Store      │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│                                                  │               │
│                                                  ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Ground     │───▶│   Label      │───▶│   Training   │       │
│  │   Truth      │    │   Generator  │    │   Dataset    │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│                                                  │               │
│                                                  ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   XGBoost    │───▶│   Calibrator │───▶│   Ensemble   │       │
│  │   Training   │    │   (Platt)    │    │   Model      │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│                                                  │               │
│                                                  ▼               │
│                                          ┌──────────────┐       │
│                                          │   Scorer     │       │
│                                          │   Service    │       │
│                                          └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature Engineering

### Feature Categories

```python
class FeatureEngineer:
    """
    Transforms raw data and signals into ML features.

    Feature Categories:
    1. Rule-based signal scores (from Phase 3)
    2. Raw wallet features
    3. Raw trade features
    4. Market context features
    5. Temporal features
    6. Interaction features
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        markets_df: pd.DataFrame,
        wallet_profiles: Dict[str, WalletProfile],
        signals_df: pd.DataFrame
    ):
        self.trades = trades_df
        self.markets = markets_df
        self.profiles = wallet_profiles
        self.signals = signals_df

    def extract_features(self, trade: Trade) -> Dict[str, float]:
        """
        Extract all features for a single trade.

        Returns dict of feature_name -> value.
        """
        features = {}

        # 1. Signal scores
        features.update(self._get_signal_features(trade))

        # 2. Wallet features
        features.update(self._get_wallet_features(trade))

        # 3. Trade features
        features.update(self._get_trade_features(trade))

        # 4. Market features
        features.update(self._get_market_features(trade))

        # 5. Temporal features
        features.update(self._get_temporal_features(trade))

        # 6. Interaction features
        features.update(self._get_interaction_features(trade, features))

        return features
```

### 1. Signal Score Features

```python
def _get_signal_features(self, trade: Trade) -> Dict[str, float]:
    """
    Features from rule-based signal detection.

    These capture domain knowledge from Phase 3.
    """
    signal = self.signals.get(trade.trade_id, {})

    return {
        # Category scores
        "sig_freshness": signal.get("freshness_score", 0),
        "sig_timing": signal.get("timing_score", 0),
        "sig_sizing": signal.get("sizing_score", 0),
        "sig_funding": signal.get("funding_score", 0),
        "sig_cluster": signal.get("cluster_score", 0),

        # Aggregated scores
        "sig_raw_score": signal.get("raw_score", 0),
        "sig_negative_discount": signal.get("negative_discount", 0),
        "sig_market_boost": signal.get("market_boost", 0),

        # Individual signal presence (binary)
        "sig_has_zero_history": 1 if "zero_history" in str(signal) else 0,
        "sig_has_pre_news": 1 if "pre_news" in str(signal) else 0,
        "sig_has_whale_size": 1 if "whale_size" in str(signal) else 0,
        "sig_has_privacy_funding": 1 if "privacy_funding" in str(signal) else 0,
        "sig_has_cluster_coord": 1 if "cluster_coordination" in str(signal) else 0,
    }
```

### 2. Wallet Features

```python
def _get_wallet_features(self, trade: Trade) -> Dict[str, float]:
    """
    Features from wallet profile.
    """
    profile = self.profiles.get(trade.wallet)
    if not profile:
        return {f"wallet_{k}": 0 for k in [
            "freshness", "days_active", "total_trades", "total_volume",
            "unique_markets", "retail_likelihood", "win_rate",
            "holds_to_resolution", "funding_risk", "is_whale",
            "off_hours_ratio", "avg_trade_size", "cluster_size"
        ]}

    return {
        # Freshness
        "wallet_freshness": profile.freshness_score,
        "wallet_days_active": profile.days_on_polymarket,
        "wallet_is_new": 1 if profile.days_on_polymarket < 7 else 0,

        # Activity
        "wallet_total_trades": profile.total_trades,
        "wallet_total_volume": profile.total_volume,
        "wallet_unique_markets": profile.unique_markets,
        "wallet_trades_log": np.log1p(profile.total_trades),
        "wallet_volume_log": np.log1p(profile.total_volume),

        # Behavior
        "wallet_retail_likelihood": profile.retail_likelihood,
        "wallet_win_rate": profile.win_rate,
        "wallet_sharpe": profile.sharpe_ratio,
        "wallet_holds_to_resolution": profile.holds_to_resolution_ratio,
        "wallet_early_exit_ratio": profile.early_exit_ratio,

        # Funding
        "wallet_funding_risk": profile.funding_risk_score,
        "wallet_has_privacy_funding": 1 if profile.has_privacy_funding else 0,

        # Size
        "wallet_is_whale": 1 if profile.is_whale else 0,
        "wallet_avg_trade_size": profile.avg_trade_size,
        "wallet_max_trade_size": profile.max_trade_size,
        "wallet_size_log": np.log1p(profile.avg_trade_size),

        # Timing
        "wallet_off_hours_ratio": profile.off_hours_ratio,
        "wallet_burst_episodes": profile.burst_episodes,

        # Cluster
        "wallet_in_cluster": 1 if profile.cluster_id else 0,
        "wallet_cluster_size": len(profile.cluster_members) if profile.cluster_members else 1,
    }
```

### 3. Trade Features

```python
def _get_trade_features(self, trade: Trade) -> Dict[str, float]:
    """
    Features from the trade itself.
    """
    return {
        # Size
        "trade_size": trade.size,
        "trade_notional": trade.notional,
        "trade_size_log": np.log1p(trade.notional),

        # Price/odds
        "trade_price": trade.price,
        "trade_is_longshot": 1 if trade.price < 0.2 else 0,
        "trade_is_favorite": 1 if trade.price > 0.8 else 0,
        "trade_odds_ratio": trade.price / (1 - trade.price) if trade.price < 1 else 100,

        # Direction
        "trade_is_buy": 1 if trade.side == "BUY" else 0,
        "trade_is_yes": 1 if trade.outcome == "Yes" else 0,
    }
```

### 4. Market Features

```python
def _get_market_features(self, trade: Trade) -> Dict[str, float]:
    """
    Features from market context.
    """
    market = self.markets.get(trade.market_id)
    if not market:
        return {f"market_{k}": 0 for k in [
            "volume", "liquidity", "days_to_resolution",
            "is_high_insider_category", "is_niche", "participant_count"
        ]}

    # Compute market baselines
    market_trades = self.trades[self.trades.market_id == trade.market_id]
    participant_count = market_trades.wallet.nunique()
    median_size = market_trades.notional.median()

    return {
        # Volume
        "market_volume": market.volume,
        "market_volume_log": np.log1p(market.volume),
        "market_liquidity": market.liquidity,

        # Timeline
        "market_days_to_resolution": self._days_until(trade.timestamp, market.end_date),
        "market_pct_timeline": self._pct_through_timeline(trade.timestamp, market),
        "market_is_near_resolution": 1 if self._days_until(trade.timestamp, market.end_date) < 7 else 0,

        # Category
        "market_is_high_insider_category": 1 if market.category in HIGH_INSIDER else 0,
        "market_is_political": 1 if market.category == "Politics" else 0,
        "market_is_business": 1 if market.category == "Business" else 0,

        # Participation
        "market_participant_count": participant_count,
        "market_is_niche": 1 if participant_count < 50 else 0,

        # Trade context
        "market_trade_vs_median": trade.notional / median_size if median_size > 0 else 0,
        "market_pct_of_volume": trade.notional / market.volume if market.volume > 0 else 0,
    }
```

### 5. Temporal Features

```python
def _get_temporal_features(self, trade: Trade) -> Dict[str, float]:
    """
    Time-based features.
    """
    ts = trade.timestamp

    return {
        # Time of day
        "time_hour": ts.hour,
        "time_is_off_hours": 1 if ts.hour < 9 or ts.hour > 21 else 0,
        "time_is_weekend": 1 if ts.weekday() >= 5 else 0,

        # Cyclical encoding
        "time_hour_sin": np.sin(2 * np.pi * ts.hour / 24),
        "time_hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "time_dow_sin": np.sin(2 * np.pi * ts.weekday() / 7),
        "time_dow_cos": np.cos(2 * np.pi * ts.weekday() / 7),
    }
```

### 6. Interaction Features

```python
def _get_interaction_features(
    self,
    trade: Trade,
    base_features: Dict[str, float]
) -> Dict[str, float]:
    """
    Derived interaction features.

    Captures non-linear relationships between features.
    """
    return {
        # Fresh wallet + large trade
        "inter_fresh_whale": (
            base_features.get("wallet_freshness", 0) *
            base_features.get("trade_size_log", 0)
        ),

        # New wallet + niche market
        "inter_new_niche": (
            base_features.get("wallet_is_new", 0) *
            base_features.get("market_is_niche", 0)
        ),

        # High insider category + near resolution
        "inter_insider_cat_near_res": (
            base_features.get("market_is_high_insider_category", 0) *
            base_features.get("market_is_near_resolution", 0)
        ),

        # Privacy funding + zero history
        "inter_privacy_zero_hist": (
            base_features.get("wallet_has_privacy_funding", 0) *
            base_features.get("sig_has_zero_history", 0)
        ),

        # Cluster + unusual size
        "inter_cluster_unusual": (
            base_features.get("wallet_in_cluster", 0) *
            base_features.get("sig_has_whale_size", 0)
        ),

        # Off hours + longshot
        "inter_offhours_longshot": (
            base_features.get("time_is_off_hours", 0) *
            base_features.get("trade_is_longshot", 0)
        ),
    }
```

---

## Ground Truth Labeling

### Hybrid Labeling Approach

```python
class GroundTruthLabeler:
    """
    Creates ground truth labels using hybrid approach.

    Label Sources:
    1. Profitability-based: Wallet profited on resolved markets
    2. Pattern-based: Wallet exhibits multiple insider signals
    3. Manual review: Analyst-labeled examples (optional)

    Final label is weighted combination.
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        markets_df: pd.DataFrame,
        wallet_profiles: Dict[str, WalletProfile]
    ):
        self.trades = trades_df
        self.markets = markets_df
        self.profiles = wallet_profiles

    def compute_profitability_label(
        self,
        wallet: str,
        trade: Trade
    ) -> float:
        """
        Label based on trade profitability.

        Criteria for positive label:
        - Trade on market that resolved
        - Wallet was on winning side
        - Entry price was favorable (<30% for winning outcome)
        """
        market = self.markets.get(trade.market_id)
        if not market or not market.resolved:
            return 0.0

        # Check if trade was on winning side
        winning_outcome = market.resolution  # "Yes" or "No"
        on_winning_side = (
            (trade.outcome == winning_outcome and trade.side == "BUY") or
            (trade.outcome != winning_outcome and trade.side == "SELL")
        )

        if not on_winning_side:
            return 0.0

        # Check if entry was at favorable odds
        if trade.price < 0.3:  # Bought at <30% when it resolved to 100%
            return 1.0
        elif trade.price < 0.5:
            return 0.7
        else:
            return 0.3

    def compute_pattern_label(
        self,
        wallet: str,
        trade: Trade,
        signal: TradeSignal
    ) -> float:
        """
        Label based on signal pattern strength.

        Criteria:
        - Multiple high-confidence signals present
        - Low negative signals
        - Wallet shows consistent profitable pattern
        """
        score = 0.0

        # High raw signal score
        if signal.aggregated.raw_score > 0.7:
            score += 0.4

        # Low negative signals
        if signal.aggregated.negative_discount < 0.3:
            score += 0.3

        # Wallet has high win rate
        profile = self.profiles.get(wallet)
        if profile and profile.win_rate > 0.65:
            score += 0.3

        return min(score, 1.0)

    def generate_label(
        self,
        trade: Trade,
        signal: TradeSignal,
        profitability_weight: float = 0.6,
        pattern_weight: float = 0.4
    ) -> float:
        """
        Generate final hybrid label.

        Returns continuous label 0-1.
        For training, can threshold to binary.
        """
        profit_label = self.compute_profitability_label(trade.wallet, trade)
        pattern_label = self.compute_pattern_label(trade.wallet, trade, signal)

        return (
            profitability_weight * profit_label +
            pattern_weight * pattern_label
        )

    def generate_all_labels(
        self,
        trades: List[Trade],
        signals: Dict[str, TradeSignal],
        binary_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate labels for all trades.

        Returns DataFrame with:
        - trade_id
        - label_continuous (0-1)
        - label_binary (0 or 1)
        - label_source (profit/pattern/mixed)
        """
        labels = []
        for trade in trades:
            signal = signals.get(trade.trade_id)
            if not signal:
                continue

            label = self.generate_label(trade, signal)
            labels.append({
                "trade_id": trade.trade_id,
                "label_continuous": label,
                "label_binary": 1 if label >= binary_threshold else 0,
                "profit_component": self.compute_profitability_label(trade.wallet, trade),
                "pattern_component": self.compute_pattern_label(trade.wallet, trade, signal),
            })

        return pd.DataFrame(labels)
```

---

## Model Training

### XGBoost Ensemble

```python
class InsiderScoringModel:
    """
    XGBoost ensemble for insider probability scoring.

    Architecture:
    - Multiple XGBoost models with different hyperparameters
    - Calibrated probabilities using Platt scaling
    - Optional rule-based overrides for edge cases
    """

    def __init__(self):
        self.models = []
        self.calibrators = []
        self.feature_columns = None
        self.is_fitted = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_models: int = 5,
        test_size: float = 0.2,
        calibration_size: float = 0.1
    ) -> Dict[str, float]:
        """
        Train ensemble of XGBoost models.

        Returns evaluation metrics.
        """
        self.feature_columns = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # Further split for calibration
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train, y_train, test_size=calibration_size,
            stratify=y_train, random_state=42
        )

        # Define hyperparameter variations
        param_variations = [
            {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 100},
            {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 200},
            {"max_depth": 8, "learning_rate": 0.03, "n_estimators": 300},
            {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 150, "subsample": 0.8},
            {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 200, "colsample_bytree": 0.8},
        ]

        # Train models
        for params in param_variations[:n_models]:
            model = xgb.XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            self.models.append(model)

            # Calibrate
            calibrator = CalibratedClassifierCV(
                model, method="sigmoid", cv="prefit"
            )
            calibrator.fit(X_cal, y_cal)
            self.calibrators.append(calibrator)

        self.is_fitted = True

        # Evaluate
        return self._evaluate(X_test, y_test)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated probabilities.

        Returns ensemble average of calibrated probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X = X[self.feature_columns]

        # Get predictions from all calibrators
        probas = np.zeros((len(X), len(self.calibrators)))
        for i, calibrator in enumerate(self.calibrators):
            probas[:, i] = calibrator.predict_proba(X)[:, 1]

        # Ensemble average
        return probas.mean(axis=1)

    def _evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance.
        """
        y_proba = self.predict_proba(X_test)
        y_pred = (y_proba > 0.5).astype(int)

        return {
            "auc_roc": roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "log_loss": log_loss(y_test, y_proba),
            "brier_score": brier_score_loss(y_test, y_proba),
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance averaged across ensemble.
        """
        importances = np.zeros(len(self.feature_columns))
        for model in self.models:
            importances += model.feature_importances_

        importances /= len(self.models)

        return pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importances
        }).sort_values("importance", ascending=False)

    def save(self, path: Path):
        """Save model to disk."""
        joblib.dump({
            "models": self.models,
            "calibrators": self.calibrators,
            "feature_columns": self.feature_columns,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "InsiderScoringModel":
        """Load model from disk."""
        data = joblib.load(path)
        model = cls()
        model.models = data["models"]
        model.calibrators = data["calibrators"]
        model.feature_columns = data["feature_columns"]
        model.is_fitted = True
        return model
```

---

## Rule-Based Overrides

```python
class RuleBasedOverrides:
    """
    Hard-coded rules that override ML predictions for edge cases.

    These capture domain knowledge that may not be learnable.
    """

    def apply_overrides(
        self,
        trade: Trade,
        features: Dict[str, float],
        ml_score: float
    ) -> float:
        """
        Apply rule-based overrides to ML score.

        Returns adjusted score.
        """
        adjusted = ml_score

        # Override 1: Privacy funding + zero history = always high
        if (features.get("wallet_has_privacy_funding", 0) == 1 and
            features.get("sig_has_zero_history", 0) == 1):
            adjusted = max(adjusted, 0.9)

        # Override 2: Known entity = always low
        if features.get("wallet_is_known_entity", 0) == 1:
            adjusted = min(adjusted, 0.1)

        # Override 3: Retail behavior + long history = cap at 0.3
        if (features.get("wallet_retail_likelihood", 0) > 0.8 and
            features.get("wallet_days_active", 0) > 180):
            adjusted = min(adjusted, 0.3)

        # Override 4: Cluster coordination + whale = boost
        if (features.get("sig_has_cluster_coord", 0) == 1 and
            features.get("wallet_is_whale", 0) == 1):
            adjusted = min(adjusted * 1.2, 1.0)

        return adjusted
```

---

## Scoring Service

```python
class InsiderScoringService:
    """
    Main service for scoring trades.

    Combines:
    - Feature engineering
    - ML model prediction
    - Rule-based overrides
    - Confidence thresholding
    """

    def __init__(
        self,
        model: InsiderScoringModel,
        feature_engineer: FeatureEngineer,
        overrides: RuleBasedOverrides = None
    ):
        self.model = model
        self.features = feature_engineer
        self.overrides = overrides or RuleBasedOverrides()

    def score_trade(self, trade: Trade) -> ScoredTrade:
        """
        Score a single trade.
        """
        # Extract features
        features = self.features.extract_features(trade)
        features_df = pd.DataFrame([features])

        # ML prediction
        ml_score = self.model.predict_proba(features_df)[0]

        # Apply overrides
        final_score = self.overrides.apply_overrides(trade, features, ml_score)

        # Determine confidence tier
        confidence_tier = self._get_confidence_tier(final_score)

        return ScoredTrade(
            trade=trade,
            ml_score=ml_score,
            final_score=final_score,
            confidence_tier=confidence_tier,
            features=features,
            top_features=self._get_top_features(features)
        )

    def score_batch(self, trades: List[Trade]) -> List[ScoredTrade]:
        """
        Score multiple trades efficiently.
        """
        # Batch feature extraction
        features_list = [self.features.extract_features(t) for t in trades]
        features_df = pd.DataFrame(features_list)

        # Batch ML prediction
        ml_scores = self.model.predict_proba(features_df)

        # Apply overrides and build results
        results = []
        for i, trade in enumerate(trades):
            final_score = self.overrides.apply_overrides(
                trade, features_list[i], ml_scores[i]
            )
            results.append(ScoredTrade(
                trade=trade,
                ml_score=ml_scores[i],
                final_score=final_score,
                confidence_tier=self._get_confidence_tier(final_score),
                features=features_list[i],
                top_features=self._get_top_features(features_list[i])
            ))

        return results

    def _get_confidence_tier(self, score: float) -> str:
        """Map score to confidence tier."""
        if score >= 0.8:
            return "very_high"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.2:
            return "low"
        else:
            return "very_low"

    def _get_top_features(
        self,
        features: Dict[str, float],
        n: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top contributing features."""
        # Sort by absolute value
        sorted_features = sorted(
            features.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:n]


@dataclass
class ScoredTrade:
    trade: Trade
    ml_score: float
    final_score: float
    confidence_tier: str
    features: Dict[str, float]
    top_features: List[Tuple[str, float]]
```

---

## Model Evaluation

```python
class ModelEvaluator:
    """
    Comprehensive model evaluation.
    """

    def __init__(self, model: InsiderScoringModel):
        self.model = model

    def evaluate_on_test(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Full evaluation on test set.
        """
        y_proba = self.model.predict_proba(X_test)

        # Metrics at various thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        threshold_metrics = {}
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            threshold_metrics[t] = {
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "n_positive": y_pred.sum(),
            }

        return {
            "auc_roc": roc_auc_score(y_test, y_proba),
            "brier": brier_score_loss(y_test, y_proba),
            "threshold_metrics": threshold_metrics,
            "feature_importance": self.model.get_feature_importance(),
        }

    def plot_calibration(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ):
        """
        Plot calibration curve.
        """
        from sklearn.calibration import calibration_curve

        y_proba = self.model.predict_proba(X_test)
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

        return {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist()
        }

    def compute_lift_chart(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_bins: int = 10
    ) -> pd.DataFrame:
        """
        Compute lift chart data.

        Shows how well model ranks positive examples.
        """
        y_proba = self.model.predict_proba(X_test)

        # Create bins
        df = pd.DataFrame({
            "proba": y_proba,
            "actual": y_test
        })
        df["decile"] = pd.qcut(df["proba"], n_bins, labels=False)

        lift = df.groupby("decile").agg({
            "actual": ["mean", "sum", "count"],
            "proba": "mean"
        }).reset_index()

        return lift
```

---

## Output Schema

### `model_outputs.parquet`

```python
{
    "trade_id": str,
    "ml_score": float,
    "final_score": float,
    "confidence_tier": str,
    "top_features_json": str,  # JSON of top 5 features
}
```

### `model_metadata.json`

```python
{
    "model_version": str,
    "training_date": str,
    "n_training_samples": int,
    "n_features": int,
    "metrics": {
        "auc_roc": float,
        "precision": float,
        "recall": float,
        "f1": float,
    },
    "feature_importance": List[Dict],
    "hyperparameters": List[Dict],
}
```

---

## CLI Interface

```python
@app.command()
def train_model(
    features_file: Path,
    labels_file: Path,
    output_dir: Path,
    n_models: int = 5
):
    """Train XGBoost ensemble."""
    pass

@app.command()
def evaluate_model(
    model_path: Path,
    test_features: Path,
    test_labels: Path,
    output_file: Path
):
    """Evaluate trained model."""
    pass

@app.command()
def score_trades(
    model_path: Path,
    trades_file: Path,
    output_file: Path
):
    """Score trades using trained model."""
    pass

@app.command()
def export_feature_importance(
    model_path: Path,
    output_file: Path
):
    """Export feature importance analysis."""
    pass
```

---

## Dependencies

```python
# Additional requirements for Phase 4

xgboost>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
shap>=0.44.0           # For explainability (optional)
matplotlib>=3.8.0      # For visualizations
```

---

## Success Criteria

- [ ] Feature engineering produces 50+ features
- [ ] Ground truth labeling produces balanced dataset
- [ ] XGBoost ensemble achieves AUC > 0.75
- [ ] Calibrated probabilities are reliable (Brier < 0.2)
- [ ] Feature importance aligns with domain expectations
- [ ] Rule-based overrides catch edge cases
- [ ] Model inference <10ms per trade

---

## Next Phase

After completing Phase 4, proceed to:
→ **Phase 5: Backtest Engine** (`05_backtest_engine/spec.md`)

The scoring model will be used to:
- Generate signals for simulated trading
- Evaluate strategy performance
- Optimize confidence thresholds
