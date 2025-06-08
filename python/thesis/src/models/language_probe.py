import json
import os
import sys

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class LanguageIDProbe(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 100, output_dim: int = 9, dropout: float = 0.5):
        super(LanguageIDProbe, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

    def _get_loader(self, X: np.ndarray, y: np.ndarray, device, batch_size: int = 1):
        self.to(device)

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).long()

        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return data_loader
    
    def _check_and_save(self, results, layer_idx: int, output_dir: str):
        path = os.path.join(output_dir, 'language_id.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
        
        else:
            data = {}
        
        data[str(layer_idx)] = results
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int = 10,
            batch_size: int = 1,
            lr: float = 1e-3
        ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader = self._get_loader(X_train, y_train, device, batch_size=batch_size)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for e in range(1, epochs + 1):
            self.train()
            total_loss = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = self.forward(x_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x_batch.size(0)
            
            avg_train_loss = total_loss / len(train_loader.dataset)

            print(f'Epoch {e}/{epochs}\nTrain loss: {avg_train_loss:.4f}\n', file=sys.stderr, flush=True)
    
    def evaluate(self,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 batch_size: int=1,
                 labels: list=None,
                 output_dir: str=None,
                 layer_idx: int=-1
        ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()

        test_loader = self._get_loader(X_test, y_test, device, batch_size=batch_size)
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = self.forward(x_batch)
                predictions = logits.argmax(dim=1)

                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(y_batch.cpu().tolist())

        correct = sum(int(pred == gold) for pred, gold in zip(all_predictions, all_labels))
        total = len(all_labels)
        accuracy = correct / total if total > 0 else 0.0

        report_dict = classification_report(all_labels, all_predictions, target_names=labels, output_dict=True)
        cm = confusion_matrix(all_labels, all_predictions, labels=list(range(len(labels))))

        results = {
            'confusion_matrix': cm.tolist(),
            'classification_report': report_dict,
            'accuracy': accuracy
        }

        if output_dir is not None:
            self._check_and_save(results, layer_idx, output_dir)

        return accuracy
