import numpy as np
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt


class LogisticRegression:
    """
    Logistic Regression implementation using gradient descent optimization.
    
    This class implements logistic regression from scratch using NumPy,
    with methods for training, prediction, and evaluation.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, 
                 tolerance: float = 1e-6, random_state: Optional[int] = None):
        """
        Initialize LogisticRegression model.
        
        Args:
            learning_rate (float): Learning rate for gradient descent (default: 0.01)
            max_iterations (int): Maximum number of training iterations (default: 1000)
            tolerance (float): Convergence tolerance (default: 1e-6)
            random_state (int, optional): Random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        
        # Model parameters
        self.theta = None
        self.bias = None
        self.feature_names = None
        self.is_fitted = False
        
        # Training history
        self.cost_history = []
        self.iteration_history = []
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid function.
        
        Args:
            z (np.ndarray): Input values
            
        Returns:
            np.ndarray: Sigmoid values between 0 and 1
        """
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """
        Add bias term (intercept) to feature matrix.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Feature matrix with bias term
        """
        return np.c_[np.ones((X.shape[0], 1)), X]
    
    def _compute_cost(self, X_b: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute logistic regression cost function.
        
        Args:
            X_b (np.ndarray): Feature matrix with bias term
            y (np.ndarray): Target values
            theta (np.ndarray): Model parameters
            
        Returns:
            float: Cost value
        """
        m = y.size
        h = self._sigmoid(X_b @ theta)
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        
        cost = -1/m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost
    
    def _compute_gradient(self, X_b: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute gradient for gradient descent.
        
        Args:
            X_b (np.ndarray): Feature matrix with bias term
            y (np.ndarray): Target values
            theta (np.ndarray): Model parameters
            
        Returns:
            np.ndarray: Gradient vector
        """
        m = y.size
        h = self._sigmoid(X_b @ theta)
        gradient = (X_b.T @ (h - y)) / m
        return gradient
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'LogisticRegression':
        """
        Train the logistic regression model using gradient descent.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Target values of shape (n_samples,)
            verbose (bool): Whether to print training progress
            
        Returns:
            LogisticRegression: Self for method chaining
        """
        # Validate inputs
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if len(y.shape) > 1:
            y = y.flatten()
        
        # Add bias term
        X_b = self._add_bias(X)
        
        # Initialize parameters
        n_features = X_b.shape[1]
        self.theta = np.zeros(n_features)
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = ['bias'] + list(X.columns)
        
        # Training history
        self.cost_history = []
        self.iteration_history = []
        
        # Gradient descent
        for iteration in range(self.max_iterations):
            # Compute predictions and cost
            cost = self._compute_cost(X_b, y, self.theta)
            self.cost_history.append(cost)
            self.iteration_history.append(iteration)
            
            # Compute gradient
            gradient = self._compute_gradient(X_b, y, self.theta)
            
            # Update parameters
            self.theta -= self.learning_rate * gradient
            
            # Check convergence
            if np.linalg.norm(gradient) < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
            
            # Print progress
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Cost: {cost:.6f}")
        
        # Extract bias and coefficients
        self.bias = self.theta[0]
        self.coefficients = self.theta[1:]
        
        self.is_fitted = True
        
        if verbose:
            print(f"Training completed. Final cost: {self.cost_history[-1]:.6f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X (np.ndarray): Features to predict on
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_b = self._add_bias(X)
        return self._sigmoid(X_b @ self.theta)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X (np.ndarray): Features to predict on
            threshold (float): Classification threshold (default: 0.5)
            
        Returns:
            np.ndarray: Predicted class labels
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): True labels
            
        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_params(self) -> dict:
        """
        Get model parameters.
        
        Returns:
            dict: Dictionary containing model parameters
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting parameters")
        
        params = {
            'bias': self.bias,
            'coefficients': self.coefficients,
            'learning_rate': self.learning_rate,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance
        }
        
        if self.feature_names:
            params['feature_names'] = self.feature_names[1:]  # Exclude bias
        
        return params
    
    def plot_cost_history(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot training cost history.
        
        Args:
            figsize (tuple): Figure size (width, height)
        """
        if not self.cost_history:
            raise ValueError("No training history available. Run fit() first.")
        
        plt.figure(figsize=figsize)
        plt.plot(self.iteration_history, self.cost_history, 'b-', linewidth=2)
        plt.title('Training Cost History', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray, 
                             feature_indices: Tuple[int, int] = (0, 1),
                             figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot decision boundary for 2D data.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): True labels
            feature_indices (tuple): Indices of features to plot (default: (0, 1))
            figsize (tuple): Figure size (width, height)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting decision boundary")
        
        if X.shape[1] < 2:
            raise ValueError("X must have at least 2 features for 2D plotting")
        
        # Extract features for plotting
        X_plot = X[:, feature_indices]
        
        # Create mesh grid
        x_min, x_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
        y_min, y_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))
        
        # Create full feature matrix for prediction
        X_mesh = np.zeros((xx.ravel().shape[0], X.shape[1]))
        X_mesh[:, feature_indices[0]] = xx.ravel()
        X_mesh[:, feature_indices[1]] = yy.ravel()
        
        # Make predictions
        Z = self.predict(X_mesh)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=figsize)
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, alpha=0.8, cmap='RdYlBu', edgecolors='black')
        
        feature_names = self.feature_names[1:] if self.feature_names else [f'Feature {i}' for i in range(X.shape[1])]
        plt.xlabel(feature_names[feature_indices[0]], fontsize=12)
        plt.ylabel(feature_names[feature_indices[1]], fontsize=12)
        plt.title('Logistic Regression Decision Boundary', fontsize=14, fontweight='bold')
        plt.colorbar(label='Predicted Class')
        plt.tight_layout()
        plt.show()
    
    def __repr__(self) -> str:
        """String representation of the model."""
        if self.is_fitted:
            return f"LogisticRegression(learning_rate={self.learning_rate}, fitted=True)"
        else:
            return f"LogisticRegression(learning_rate={self.learning_rate}, fitted=False)" 