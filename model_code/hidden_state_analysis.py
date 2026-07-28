import random

from model_code.generate import _make_progress
from model_code.steering_extraction import generateSteering
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
import numpy as np



def build_cosine_matrix(vectors_by_emotion, emotions,layer_idx):
    """
    Built a distance matrix between all the different emotions

    EXPECTED INPUT:
    vectors_by_emotion: dict of emotion to list of vectors (one per layer)
    emotions: list of emotions to include in the matrix
    layer_idx: which layer to use for the vectors (if the vectors are lists of layers)
    """
    layer_vectors = {emotion: vectors_by_emotion[emotion][layer_idx] for emotion in emotions}
    matrix = np.zeros((len(emotions), len(emotions)))

    for i, emotion_i in enumerate(emotions):
        for j, emotion_j in enumerate(emotions):
            matrix[i, j] = F.cosine_similarity(
                layer_vectors[emotion_i].unsqueeze(0),
                layer_vectors[emotion_j].unsqueeze(0),
                dim=1
            ).item()

    return pd.DataFrame(matrix, index=emotions, columns=emotions)


def annotate_heatmap(ax, matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f'{matrix[i, j]:.2f}',
                ha='center', va='center',
                color='black', fontsize=10
            )

def mantel_test(matrix1, matrix2, num_permutations=1000, seed=42):
    """
    Perform a Mantel test to assess the correlation between two distance matrices.

    Parameters:
    - matrix1: First distance matrix (2D numpy array).
    - matrix2: Second distance matrix (2D numpy array).
    - num_permutations: Number of permutations for significance testing.
    - seed: Random seed for reproducibility.

    Returns:
    - correlation: The observed correlation between the two matrices.
    - p_value: The p-value from the permutation test.
    """
    # Flatten the upper triangle of the matrices (excluding the diagonal)
    def flatten_upper_triangle(matrix):
        return matrix[np.triu_indices_from(matrix, k=1)]

    flat1 = flatten_upper_triangle(matrix1)
    flat2 = flatten_upper_triangle(matrix2)

    # Compute the observed correlation
    correlation = np.corrcoef(flat1, flat2)[0, 1]

    # Permutation test
    random.seed(seed)
    count = 0
    for _ in range(num_permutations):
        permuted_flat2 = np.random.permutation(flat2)
        permuted_correlation = np.corrcoef(flat1, permuted_flat2)[0, 1]
        if abs(permuted_correlation) >= abs(correlation):
            count += 1

    p_value = count / num_permutations
    return correlation, p_value


# Simple MLP for probing task
class LogisticRegressionProbe(nn.Module):
    """
    Probe model for binary probing tasks.

    - Logistic regression mode: single linear layer (set use_logistic_regression=True).
    - Used to determine whether structure are similar across languages
    """
    def __init__(
        self,
        input_dim,
        output_dim,
    ):
        super().__init__()
        # Linear probe (logistic regression when used with BCEWithLogitsLoss).
        self.everything = nn.Sequential(nn.Linear(input_dim, output_dim))
    def forward(self, X):
        return self.everything(X)
    

# Load data into DataLoader
# A possible problem is that we shuffle incorrect vectors, so some emotion not listed might be 
# overrepresented
def create_dataloader(
    emotion_dict:dict,
    emotion:str,
    seed:int=42,
    batch_size:int=32,
    shuffle=True,
    val_ratio:float=0.2,
    multiclass=False,
    multiclass_emotions=None,
):
    """
    Expects an emotion dict, that consist of emotion keys, and values that are lists of vectors.
    The function will create a dataloader that can be used to train a probe
    """
    if multiclass:
        # For multiclass, we will create a dataset where each vector is labeled with its emotion index
        vector_blocks = []
        label_blocks = []
        if multiclass_emotions is None:
            multiclass_emotions = list(emotion_dict.keys())

        # assign emotion to a number and add the entire emotion text vectors 
        for idx, emo in enumerate(multiclass_emotions):
            vecs = emotion_dict[emo]
            vecs_np = np.asarray(vecs)
            vector_blocks.append(vecs_np)
            label_blocks.append(np.full(vecs_np.shape[0], idx, dtype=np.int64))

        all_vectors = np.concatenate(vector_blocks, axis=0)
        all_labels = np.concatenate(label_blocks, axis=0)

        X = torch.tensor(all_vectors, dtype=torch.float32)
        y = torch.tensor(all_labels, dtype=torch.long)

        # Split into training and validation sets
        dataset = torch.utils.data.TensorDataset(X, y)
        val_size = max(1, int(len(dataset) * val_ratio))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    correct_vectors = emotion_dict[emotion]
    correct_vectors = list(correct_vectors) if not isinstance(correct_vectors, list) else correct_vectors
    # Get all incorrect, stack them, shuffle them, then retrieve the same number as correct vectors.
    incorrect_vectors = [
        vec for emo, vecs in emotion_dict.items() if emo != emotion for vec in vecs
    ]
    random.Random(seed).shuffle(incorrect_vectors)
    incorrect_vectors = incorrect_vectors[:len(correct_vectors)]
    
    y_correct = torch.ones(len(correct_vectors))
    y_incorrect = torch.zeros(len(incorrect_vectors))
    
    all_vectors = correct_vectors + incorrect_vectors
    if isinstance(all_vectors[0], torch.Tensor):
        X = torch.stack([v.float() for v in all_vectors], dim=0)
    else:
        X = torch.tensor(np.asarray(all_vectors), dtype=torch.float32)
    y = torch.cat([y_correct, y_incorrect])

    # Split into training and validation sets
    dataset = torch.utils.data.TensorDataset(X, y)
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



# Train a probe 
def trainProbe(
    probe, 
    train_loader, 
    val_loader, 
    
    num_epochs=10, 
    learning_rate=1e-3, 
    device='cuda'
):
    probe.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        probe.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = probe(X_batch).squeeze()  # Get predictions
            loss = criterion(outputs, y_batch)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

        # Validation step 
        probe.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = probe(X_val).squeeze()
                predicted = (torch.sigmoid(outputs) > 0.5).float()  # Threshold at 0.5
                correct += (predicted == y_val).sum().item()
                total += y_val.size(0)

        accuracy = correct / total if total > 0 else 0
        print(f'Validation Accuracy: {accuracy:.4f}')

    return probe

def trainMultiClassProbe(
    probe, 
    train_loader, 
    val_loader, 
    num_epochs=10, 
    learning_rate=1e-3, 
    device='cuda'
):
    probe.to(device)
    criterion = nn.CrossEntropyLoss()  # Multi-class classification loss
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        probe.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = probe(X_batch)  # Get predictions
            loss = criterion(outputs, y_batch)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

        # Validation step 
        probe.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = probe(X_val)
                predicted = torch.argmax(outputs, dim=1)  # Get the class with highest probability
                correct += (predicted == y_val).sum().item()
                total += y_val.size(0)

        accuracy = correct / total if total > 0 else 0
        print(f'Validation Accuracy: {accuracy:.4f}')

    return probe

def predictProbe(probe, dataloader, device='cuda',multiclass=False):
    probe.eval()
    all_predictions = []

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:  # We don't need labels for prediction
            X_batch = X_batch.to(device)
            if multiclass:
                outputs = probe(X_batch)
                predicted = torch.argmax(outputs, dim=1)
            else:
                outputs = probe(X_batch).squeeze()
                predicted = (torch.sigmoid(outputs) > 0.5).float()  # Threshold at 0.5
            matches = (predicted.cpu() == Y_batch).float()  # Compare with true labels
            
            # Sum and divide by batch size to get the proportion of correct predictions in this batch
            batch_accuracy = matches.sum().item() / len(Y_batch)
            all_predictions.append(batch_accuracy)
    return np.array(all_predictions)
    