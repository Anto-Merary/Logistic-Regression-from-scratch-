#!/usr/bin/env python3
"""
Logistic Regression Example Script

This script demonstrates the usage of the object-oriented logistic regression
implementation with comprehensive evaluation and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression
from utils import DataPreprocessor, ModelEvaluator, split_data, load_sample_data, print_model_summary


def main():
    """Main function demonstrating logistic regression usage."""
    
    print("Logistic Regression Implementation Example")
    print("=" * 50)
    
    # 1. Load and prepare data
    print("\nLoading breast cancer dataset...")
    X, y = load_sample_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # 2. Split data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 3. Preprocess data
    print("\nPreprocessing data...")
    preprocessor = DataPreprocessor(scaler_type='standard')
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    print("Data scaled using StandardScaler")
    
    # 4. Create and train model
    print("\nTraining logistic regression model...")
    model = LogisticRegression(
        learning_rate=0.01,
        max_iterations=1000,
        tolerance=1e-6,
        random_state=42
    )
    
    # Train with verbose output
    model.fit(X_train_scaled, y_train, verbose=True)
    
    # 5. Make predictions
    print("\nMaking predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)
    
    # 6. Evaluate model
    print("\nEvaluating model performance...")
    
    # Calculate accuracies
    train_accuracy = ModelEvaluator.accuracy_score(y_train, y_train_pred)
    test_accuracy = ModelEvaluator.accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Generate classification report
    print("\nClassification Report:")
    print(ModelEvaluator.classification_report(y_test, y_test_pred, 
                                             target_names=['Benign', 'Malignant']))
    
    # 7. Print comprehensive model summary
    print_model_summary(model, X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 8. Visualizations
    print("\nGenerating visualizations...")
    
    # Plot training cost history
    print("  - Plotting cost history...")
    model.plot_cost_history()
    
    # Plot confusion matrix
    print("  - Plotting confusion matrix...")
    ModelEvaluator.plot_confusion_matrix(y_test, y_test_pred)
    
    # Plot ROC curve
    print("  - Plotting ROC curve...")
    ModelEvaluator.plot_roc_curve(y_test, y_test_proba)
    
    # Plot precision-recall curve
    print("  - Plotting precision-recall curve...")
    ModelEvaluator.plot_precision_recall_curve(y_test, y_test_proba)
    
    # Plot decision boundary (using first two features)
    print("  - Plotting decision boundary...")
    try:
        model.plot_decision_boundary(X_test_scaled, y_test, feature_indices=(0, 1))
    except Exception as e:
        print(f"    Note: Decision boundary plot requires 2D data. Error: {e}")
    
    # 9. Model parameters analysis
    print("\nModel Parameters Analysis:")
    params = model.get_params()
    print(f"Bias (Intercept): {params['bias']:.6f}")
    print(f"Number of coefficients: {len(params['coefficients'])}")
    print(f"Largest coefficient: {np.max(np.abs(params['coefficients'])):.6f}")
    print(f"Smallest coefficient: {np.min(np.abs(params['coefficients'])):.6f}")
    
    # 10. Feature importance (if feature names available)
    if 'feature_names' in params:
        print("\nTop 10 Most Important Features:")
        feature_importance = list(zip(params['feature_names'], np.abs(params['coefficients'])))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"  {i+1:2d}. {feature}: {importance:.6f}")
    
    print("\nExample completed successfully!")
    print("=" * 50)


def advanced_example():
    """Advanced example with different hyperparameters and comparison."""
    
    print("\nAdvanced Example: Hyperparameter Comparison")
    print("=" * 50)
    
    # Load data
    X, y = load_sample_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    
    # Preprocess
    preprocessor = DataPreprocessor(scaler_type='standard')
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Test different learning rates
    learning_rates = [0.001, 0.01, 0.1]
    results = {}
    
    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        
        model = LogisticRegression(
            learning_rate=lr,
            max_iterations=1000,
            tolerance=1e-6,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train, verbose=False)
        
        train_acc = model.score(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)
        
        results[lr] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'iterations': len(model.cost_history),
            'final_cost': model.cost_history[-1] if model.cost_history else None
        }
        
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Iterations: {len(model.cost_history)}")
        print(f"  Final Cost: {model.cost_history[-1]:.6f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    lrs = list(results.keys())
    train_accs = [results[lr]['train_accuracy'] for lr in lrs]
    test_accs = [results[lr]['test_accuracy'] for lr in lrs]
    
    plt.plot(lrs, train_accs, 'o-', label='Training Accuracy', linewidth=2)
    plt.plot(lrs, test_accs, 's-', label='Test Accuracy', linewidth=2)
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Iterations comparison
    plt.subplot(2, 2, 2)
    iterations = [results[lr]['iterations'] for lr in lrs]
    plt.plot(lrs, iterations, 'o-', color='green', linewidth=2)
    plt.xlabel('Learning Rate')
    plt.ylabel('Iterations')
    plt.title('Convergence Speed vs Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Cost comparison
    plt.subplot(2, 2, 3)
    costs = [results[lr]['final_cost'] for lr in lrs]
    plt.plot(lrs, costs, 'o-', color='red', linewidth=2)
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Cost')
    plt.title('Final Cost vs Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print("\nAdvanced example completed!")


if __name__ == "__main__":
    # Run basic example
    main()
    
    # Run advanced example
    advanced_example() 