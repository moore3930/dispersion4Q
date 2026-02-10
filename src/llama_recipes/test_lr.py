from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)  # A single-layer linear model

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = SimpleModel()

# Optimizer
optimizer = Adam(model.parameters(), lr=0.01)

# Scheduler
scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

# Training Loop
for epoch in range(5):
    # Training steps here...
    print(f"Epoch {epoch}, Current LR: {optimizer.param_groups[0]['lr']}")
    scheduler.step()  # Adjust learning rate
