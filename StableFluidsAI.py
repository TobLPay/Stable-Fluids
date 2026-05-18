import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

class FluidDL(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )
    
    def forward(self, x):
        return self.net(x)

X_data = np.load("X_data.npy")
Y_data = np.load("Y_data.npy")


density_max = np.max(np.abs(X_data[:, 0])) + 1e-7
velx_max = np.max(np.abs(X_data[:, 1])) + 1e-7
vely_max = np.max(np.abs(X_data[:, 2])) + 1e-7

X_data[:, 0:1] /= density_max
X_data[:, 1] /= velx_max
X_data[:, 2] /= vely_max

Y_data[:, 0:1] /= density_max
Y_data[:, 1] /= velx_max
Y_data[:, 2] /= vely_max

X = torch.tensor(X_data, dtype=torch.float32)
Y = torch.tensor(Y_data, dtype=torch.float32)

model = FluidDL()
creterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X = X.to(device)
Y = Y.to(device)

for epoch in range(100):
    pred = model(X)
    loss = creterion(pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

import pygame
pygame.init()
screen = pygame.display.set_mode((750, 500))
clock = pygame.time.Clock()
resp = input("시각화를 진행합니다. Enter를 누르세요.")

state = X[0:1]
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
    with torch.no_grad():
        pred = model(state)
        state = 0.5 * state + 0.5 * pred
    density = state[0, 0].cpu().numpy() * density_max
    img = np.sqrt(np.maximum(density, 0)) * 40
    img = np.clip(img, 0, 255).astype(np.uint8)

    rgb = np.stack((img, img, img), axis=-1)
    surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    surface = pygame.transform.scale(surface, (750, 500))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
    
with torch.no_grad():
    pred = model(state)
pred_np = pred[0].cpu().numpy()
density = pred_np[0]
vel_x = pred_np[1]
vel_y = pred_np[2]

import matplotlib.pyplot as plt
plt.imshow(density)
plt.title("Predicted Density")
plt.colorbar()
plt.show()