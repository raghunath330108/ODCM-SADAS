"""Section 3.5 classifier: LSTM over the per-object observation sequence + linear softmax head over the 6 classes.

nn.LSTM implements Eq. 26-31 (input / forget / output gates, cell update, hidden state); Eq. 24-25 are its sigmoid / tanh.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SequenceClassifier(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout, n_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x, lengths):
        """x (B, N, D) right-padded; lengths (B,) valid steps. Class logits come from the last valid step."""
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(self.drop(h[-1]))
