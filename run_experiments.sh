#!/bin/bash
# ============================================================
# MLOps Lab 1 — Hyperparameter Tuning Experiments
# ============================================================
# This script runs multiple training experiments to study
# the effect of the regularization parameter C on model
# performance, demonstrating overfitting analysis.
# ============================================================

echo "🚀 Starting Hyperparameter Tuning Experiments..."
echo "=================================================="
echo ""

# Experiment 1: Vary C (regularization) for Logistic Regression
echo "📊 Series 1: Logistic Regression — varying C"
echo "----------------------------------------------"
for C in 0.01 0.1 0.5 1.0 5.0 10.0; do
    echo ""
    echo "▶ Running with C=$C ..."
    python src/train.py --model_type logistic_regression --C $C --max_features 5000
done

echo ""
echo "=================================================="

# Experiment 2: Vary max_features for Logistic Regression
echo "📊 Series 2: Logistic Regression — varying max_features"
echo "----------------------------------------------"
for max_features in 1000 3000 5000 10000 15000; do
    echo ""
    echo "▶ Running with max_features=$max_features ..."
    python src/train.py --model_type logistic_regression --C 1.0 --max_features $max_features
done

echo ""
echo "=================================================="

# Experiment 3: Different model types
echo "📊 Series 3: Comparing model types"
echo "----------------------------------------------"
echo ""
echo "▶ Running Random Forest..."
python src/train.py --model_type random_forest --n_estimators 100

echo ""
echo "▶ Running SVM..."
python src/train.py --model_type svm --C 1.0

echo ""
echo "=================================================="
echo "✅ All experiments completed!"
echo "Run 'mlflow ui' and open http://127.0.0.1:5000 to view results."
