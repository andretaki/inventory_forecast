import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class InventoryForecastModel:
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1):
        self.model = LSTMForecaster(input_size, hidden_size, num_layers, output_size)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> None:
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(X_tensor)
            loss = self.criterion(y_pred, y_tensor)
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item()}")

    def predict(self, data: np.ndarray, steps: int) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            predictions = []
            # Ensure input tensor has the correct shape: (1, sequence_length, input_size)
            input_tensor = torch.FloatTensor(data).unsqueeze(0)  # Adds batch dimension to (sequence_length, input_size)
            for _ in range(steps):
                output = self.model(input_tensor)  # Model expects (batch_size, sequence_length, input_size)
                predictions.append(output.item())
                # Shift the input tensor and append the output to the end
                input_tensor = torch.cat([input_tensor[:, 1:, :], output.unsqueeze(0).unsqueeze(2)], dim=1)
                # input_tensor shape remains (1, sequence_length, input_size)
        return np.array(predictions)
