import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

data_x = []
data_y = []
w = 30
h = 20

epochs = 200
samples = 500
cases = 100

def advect(field, vel_x, vel_y):
    rx = random.randint(2, w-3)
    ry = random.randint(2, h-3)
    field[ry-1:ry+1, rx-1:rx+1] = 1.0
    vel_x[ry-2:ry+2, rx-2:rx+2] = random.randint(-3, 3)
    vel_y[ry-2:ry+2, rx-2:rx+2] = random.randint(-3, 3)
    data_x.append(np.stack((field, vel_x, vel_y), axis=0))
    advfield = field.copy()
    for y in range(1, h-1):
        for x in range(1, w-1):
            xx = x - vel_x[y, x] * 0.8
            yy = y - vel_y[y, x] * 0.8
            xx = max(0.0, min(xx, w - 1.0))
            yy = max(0.0, min(yy, h - 1.0))
            x0 = int(np.floor(xx))
            x1 = min(x0 + 1, w - 1)
            y0 = int(np.floor(yy))
            y1 = min(y0 + 1, h - 1)
            x_frac = xx - x0
            y_frac = yy - y0
            advfield[y, x] = (
                field[y0, x0] * (1 - x_frac) * (1 - y_frac) +
                field[y0, x1] * x_frac * (1 - y_frac) +
                field[y1, x0] * (1 - x_frac) * y_frac +
                field[y1, x1] * x_frac * y_frac
            )
    data_y.append(advfield)

def adv_test(field, vel_x, vel_y):
    rx = random.randint(2, w-3)
    ry = random.randint(2, h-3)
    field[ry-1:ry+1, rx-1:rx+1] = 1.0
    vel_x[ry-2:ry+2, rx-2:rx+2] = random.randint(-3, 3)
    vel_y[ry-2:ry+2, rx-2:rx+2] = random.randint(-3, 3)
    test_x.append(np.stack((field, vel_x, vel_y), axis=0))
    advfield = field.copy()
    for y in range(1, h-1):
        for x in range(1, w-1):
            xx = x - vel_x[y, x] * 0.8
            yy = y - vel_y[y, x] * 0.8
            xx = max(0.0, min(xx, w - 1.0))
            yy = max(0.0, min(yy, h - 1.0))
            x0 = int(np.floor(xx))
            x1 = min(x0 + 1, w - 1)
            y0 = int(np.floor(yy))
            y1 = min(y0 + 1, h - 1)
            x_frac = xx - x0
            y_frac = yy - y0
            advfield[y, x] = (
                field[y0, x0] * (1 - x_frac) * (1 - y_frac) +
                field[y0, x1] * x_frac * (1 - y_frac) +
                field[y1, x0] * (1 - x_frac) * y_frac +
                field[y1, x1] * x_frac * y_frac
            )
    test_y.append(advfield)

for k in range(samples):
    field = np.zeros((h, w), dtype=np.float32)
    vel_x = np.zeros_like(field)
    vel_y = np.zeros_like(field)
    advect(field, vel_x, vel_y)
    if k % 100 == 99:
        print(f"Generated {k+1} samples.\n")
    
data_x = np.array(data_x)
data_y = np.array(data_y)
np.save("data_x.npy", data_x)
np.save("data_y.npy", data_y)

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
    def forward(self, x):
        return self.net(x)

den_max = np.max(np.abs(data_x[:, 0])) + 1e-7
velx_max = np.max(np.abs(data_x[:, 1])) + 1e-7
vely_max = np.max(np.abs(data_x[:, 2])) + 1e-7

data_x[:, 0] /= den_max
data_x[:, 1] /= velx_max
data_x[:, 2] /= vely_max

data_y /= den_max

X = torch.tensor(data_x, dtype=torch.float32)
Y = torch.tensor(data_y, dtype=torch.float32).unsqueeze(1)

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
X = X.to(device)
Y = Y.to(device)

for epoch in range(epochs):
    pred = model(X)
    loss = criterion(pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

resp = input("Sim model.")
mse, sse, rmse, r2, min_err, mae, max_err = 0, 0, 0, 0, 0, 0, 0
for i in range(cases):
    field = np.zeros((h, w), dtype=np.float32)
    vel_x = np.zeros_like(field)
    vel_y = np.zeros_like(field)
    test_x = []
    test_y = []
    adv_test(field, vel_x, vel_y)
    data_test_x = np.array(test_x)
    data_test_x[:, 0] /= den_max
    data_test_x[:, 1] /= velx_max
    data_test_x[:, 2] /= vely_max
    X_test = torch.tensor(data_test_x, dtype=torch.float32).to(device)
    with torch.no_grad():
        state = X_test[0:1]
        pred = model(state)
        density = pred[0, 0].cpu().numpy() * den_max
    sse += np.sum((density - test_y[0]) ** 2)
    mse += np.mean((density - test_y[0]) ** 2)
    rmse += np.sqrt(np.mean((density - test_y[0]) ** 2))
    mae += np.mean(np.abs(density - test_y[0]))
    r2 += 1 - np.sum((density - test_y[0]) ** 2) / (np.sum((test_y[0] - np.mean(test_y[0])) ** 2) + 1e-7)
    min_err += np.min(np.abs(density - test_y[0]))
    max_err += np.max(np.abs(density - test_y[0]))

plt.imshow(density, cmap='viridis')
plt.colorbar()
plt.title('Predicted Density')
plt.show()

plt.imshow(test_y[0], cmap='viridis')
plt.colorbar()
plt.title('True Density')
plt.show()

plt.imshow(np.abs(density - test_y[0]), cmap='viridis')
plt.colorbar()
plt.title('Absolute Error')
plt.show()

sse = sse / cases
mse = mse / cases
rmse = rmse / cases
mae = mae / cases
r2 = r2 / cases
min_err = min_err / cases
max_err = max_err / cases

print(f"SSE: {sse:.6f}")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R2: {r2:.6f}")
print(f"Min Error: {min_err:.6f}")
print(f"MAE: {mae:.6f}")
print(f"Max Error: {max_err:.6f}")
