"""
Neural network models and training functionality for baseball analytics.

Contains the PyTorch model definition, training procedures, and evaluation functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from collections import Counter
import logging
from typing import Tuple, List, Dict

from config import MODEL_CONFIG, TARGET_NAMES

logger = logging.getLogger(__name__)


class ImprovedPitchClassifier(nn.Module):
    """
    Feedforward neural network for pitch outcome classification.
    
    This model predicts pitch outcomes using a simplified but effective architecture
    designed to prevent overfitting while capturing complex feature interactions.
    The architecture uses batch normalization and dropout for regularization.
    
    Architecture:
    - Input normalization layer
    - 3 hidden layers with decreasing size (128 → 64 → 32 neurons)
    - Batch normalization and dropout after each layer
    - Final classification layer (3 outputs: Pitcher Win, Neutral, Hitter Win)
    
    Design Principles:
    - Batch normalization for stable training and faster convergence
    - Dropout for preventing overfitting to training data
    - ReLU activation for handling sparse features effectively
    - Progressive dimensionality reduction for feature hierarchy
    """
    
    def __init__(self, input_size: int, 
                 hidden_size: int = MODEL_CONFIG['hidden_size'], 
                 num_classes: int = MODEL_CONFIG['num_classes'], 
                 dropout: float = MODEL_CONFIG['dropout']):
        """
        Initialize the neural network architecture.
        
        Args:
            input_size (int): Number of input features (40 in our case)
            hidden_size (int): Size of first hidden layer (default: 128)
            num_classes (int): Number of output classes (3: Pitcher/Neutral/Hitter Win)
            dropout (float): Dropout probability for regularization (default: 0.4)
        """
        super().__init__()
        
        # Input normalization to stabilize training
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Sequential architecture with progressive size reduction
        self.layers = nn.Sequential(
            # First hidden layer: Full feature representation
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second hidden layer: Feature abstraction
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Third hidden layer: High-level patterns
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer: Classification predictions
            nn.Linear(hidden_size // 4, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input features, shape (batch_size, input_size) or
                            (batch_size, seq_len, features) for sequence data
                            
        Returns:
            torch.Tensor: Raw logits for each class, shape (batch_size, num_classes)
        """
        # Handle sequence data by flattening if necessary
        if len(x.shape) == 3:  # Convert sequences to single pitch features
            batch_size, seq_len, features = x.shape
            x = x.reshape(batch_size, seq_len * features)
        
        # Apply input normalization and forward pass
        x = self.input_norm(x)
        return self.layers(x)


def train_improved_model(X_train: torch.Tensor, y_train: torch.Tensor,
                        X_val: torch.Tensor, y_val: torch.Tensor,
                        input_size: int, 
                        num_classes: int = MODEL_CONFIG['num_classes'],
                        epochs: int = MODEL_CONFIG['epochs'], 
                        batch_size: int = MODEL_CONFIG['batch_size'],
                        learning_rate: float = MODEL_CONFIG['learning_rate']) -> Tuple[ImprovedPitchClassifier, List, List]:
    """
    Train the neural network with advanced techniques for optimal performance.
    
    This function implements a comprehensive training pipeline with:
    - Class balancing through weighted sampling
    - Early stopping to prevent overfitting
    - Learning rate scheduling for convergence
    - Gradient clipping for training stability
    
    Args:
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels  
        input_size (int): Number of input features
        num_classes (int): Number of output classes
        epochs (int): Maximum training epochs
        batch_size (int): Mini-batch size for training
        learning_rate (float): Initial learning rate
        
    Returns:
        Tuple containing:
        - Trained model (ImprovedPitchClassifier)
        - Training loss history (List[float])
        - Validation loss history (List[float])
    """
    
    # === CLASS BALANCING SETUP ===
    # Handle imbalanced classes by computing sample weights
    class_counts = Counter(y_train.numpy())
    total_samples = len(y_train)
    class_weights = {cls: total_samples / (len(class_counts) * count) 
                    for cls, count in class_counts.items()}
    
    # Create weighted sampler to balance training batches
    weights = torch.tensor([class_weights[int(y.item())] for y in y_train], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights, len(weights))
    
    # === MODEL AND OPTIMIZER SETUP ===
    model = ImprovedPitchClassifier(input_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()  # Appropriate for multi-class classification
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Data loaders
    train_dataset = TensorDataset(X_train, y_train.long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    
    best_val_acc = 0
    patience_counter = 0
    patience = MODEL_CONFIG['patience']
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val.long()).item()
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_accuracy = (val_predictions == y_val.long()).float().mean().item()
        
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def evaluate_classification_model(model: ImprovedPitchClassifier, 
                                 X_test: torch.Tensor, y_test: torch.Tensor) -> Dict:
    """Comprehensively evaluate the trained classification model on a held-out test set.

    Args:
        model (ImprovedPitchClassifier): Trained neural network classifier.
        X_test (torch.Tensor): Scaled test features of shape (n_samples, n_features).
        y_test (torch.Tensor): Ground-truth class labels for the test set.

    Returns:
        Dict: Dictionary containing:
            - accuracy (float): Overall accuracy on the test set.
            - f1_weighted (float): Weighted F1 score across classes.
            - classification_report (dict): Per-class precision/recall/F1 support metrics.
            - predictions (np.ndarray): Predicted class indices for each sample.
            - probabilities (np.ndarray): Softmax probabilities per class, per sample.
            - actuals (np.ndarray): True labels for comparison.

    Notes:
        - The function logs summary metrics and prints a formatted classification report.
        - Softmax is applied to logits to expose calibrated class probabilities.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1).numpy()
        probabilities = torch.softmax(outputs, dim=1).numpy()
        actuals = y_test.numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(actuals, predictions)
    f1 = f1_score(actuals, predictions, average='weighted')
    
    # Class-wise metrics
    class_report = classification_report(actuals, predictions, 
                                       target_names=TARGET_NAMES,
                                       output_dict=True)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Weighted F1: {f1:.4f}")
    logger.info("\nClassification Report:")
    print(classification_report(actuals, predictions, target_names=TARGET_NAMES))
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1,
        'classification_report': class_report,
        'predictions': predictions,
        'probabilities': probabilities,
        'actuals': actuals
    }