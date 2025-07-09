import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Union


class DataPreprocessor:
    """
    Data preprocessing utilities for logistic regression.
    """
    
    def __init__(self, scaler_type: str = 'standard', random_state: Optional[int] = None):
        """
        Initialize DataPreprocessor.
        
        Args:
            scaler_type (str): Type of scaler ('standard' or 'minmax')
            random_state (int, optional): Random seed for reproducibility
        """
        self.scaler_type = scaler_type
        self.random_state = random_state
        self.scaler = None
        self.is_fitted = False
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit scaler and transform data.
        
        Args:
            X (np.ndarray or pd.DataFrame): Features to scale
            
        Returns:
            np.ndarray: Scaled features
        """
        scaled_X = self.scaler.fit_transform(X)
        self.is_fitted = True
        return scaled_X
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Args:
            X (np.ndarray or pd.DataFrame): Features to scale
            
        Returns:
            np.ndarray: Scaled features
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming")
        return self.scaler.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data.
        
        Args:
            X (np.ndarray): Scaled features
            
        Returns:
            np.ndarray: Original scale features
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transforming")
        return self.scaler.inverse_transform(X)


def split_data(X: Union[np.ndarray, pd.DataFrame], 
               y: Union[np.ndarray, pd.Series],
               test_size: float = 0.2,
               random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.
    
    Args:
        X (np.ndarray or pd.DataFrame): Features
        y (np.ndarray or pd.Series): Target values
        test_size (float): Proportion of test set (default: 0.2)
        random_state (int, optional): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


class ModelEvaluator:
    """
    Model evaluation utilities for logistic regression.
    """
    
    @staticmethod
    def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            float: Accuracy score
        """
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                            target_names: Optional[list] = None) -> str:
        """
        Generate classification report.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            target_names (list, optional): Names of target classes
            
        Returns:
            str: Classification report
        """
        return classification_report(y_true, y_pred, target_names=target_names)
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            np.ndarray: Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                            figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            figsize (tuple): Figure size (width, height)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, 
                      figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_proba (np.ndarray): Predicted probabilities
            figsize (tuple): Figure size (width, height)
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, 
                                   figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_proba (np.ndarray): Predicted probabilities
            figsize (tuple): Figure size (width, height)
        """
        from sklearn.metrics import precision_recall_curve, auc
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def load_sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load sample breast cancer dataset.
    
    Returns:
        tuple: (X, y) features and target
    """
    from sklearn.datasets import load_breast_cancer
    
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    return X, y


def print_model_summary(model, X_train: np.ndarray, X_test: np.ndarray, 
                       y_train: np.ndarray, y_test: np.ndarray) -> None:
    """
    Print comprehensive model summary.
    
    Args:
        model: Fitted logistic regression model
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Test features
        y_train (np.ndarray): Training labels
        y_test (np.ndarray): Test labels
    """
    print("=" * 60)
    print("LOGISTIC REGRESSION MODEL SUMMARY")
    print("=" * 60)
    
    # Model parameters
    params = model.get_params()
    print(f"\nModel Parameters:")
    print(f"  Learning Rate: {params['learning_rate']}")
    print(f"  Max Iterations: {params['max_iterations']}")
    print(f"  Tolerance: {params['tolerance']}")
    print(f"  Bias (Intercept): {params['bias']:.6f}")
    
    # Performance metrics
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"\nPerformance Metrics:")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    # Training history
    if model.cost_history:
        print(f"  Final Cost: {model.cost_history[-1]:.6f}")
        print(f"  Iterations: {len(model.cost_history)}")
    
    print("=" * 60) 