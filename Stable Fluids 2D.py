#학교 제출용 2612 여*음 Stable Fluids 모듈(공개 사이트라서 이름 가렸습니당)

import numpy as np
import pygame
import random
from numba import njit

@njit
def diffuse(field, diff, dt, iterations=4):
    a = diff * dt * field.shape[0] * field.shape[1]
    x = field.copy()
    b = field.copy()
    for _ in range(iterations):
        x[1:-1, 1:-1] = (b[1:-1, 1:-1] + a * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])) / (1 + 4*a)
    return x

@njit
def bilerp(field, x, y):
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, field.shape[1] - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, field.shape[0] - 1)
    x_frac = x - x0
    y_frac = y - y0
    return(
        field[y0, x0] * (1 - x_frac) * (1 - y_frac) +
        field[y1, x0] * y_frac * (1 - x_frac) +
        field[y0, x1] * (1 - y_frac) * x_frac +
        field[y1, x1] * x_frac * y_frac
    )

@njit
def advection(field, vel_x, vel_y, dt):
    advfield = field.copy()
    for yy in range(1, field.shape[0]-1):
        for xx in range(1, field.shape[1]-1):
            x = xx - vel_x[yy, xx] * dt
            y = yy - vel_y[yy, xx] * dt
            x = max(0.0, min(x, field.shape[1] - 1.0))
            y = max(0.0, min(y, field.shape[0] - 1.0))
            advfield[yy, xx] = bilerp(field, x, y)
    return advfield

@njit
def set_boundary(x, y):
    x[:, 0] = 0
    x[:, -1] = 0
    y[:, 0] = y[:, 1]
    y[:, -1] = y[:, -2]

    y[0, :] = 0
    y[-1, :] = 0
    x[0, :] = x[1, :]
    x[-1, :] = x[-2, :]

@njit
def set_boundary_density(density):
    density[0, :] = density[1, :]
    density[-1, :] = density[-2, :]
    density[:, 0] = density[:, 1]
    density[:, -1] = density[:, -2]

@njit
def projection(vel_x, vel_y, iterations=8):
    div = np.zeros(vel_x.shape[0:2])
    p = div.copy()
    div[1:-1, 1:-1] = -0.5 * (
        vel_x[1:-1, 2:] - vel_x[1:-1, :-2] +
        vel_y[2:, 1:-1] - vel_y[:-2, 1:-1]
    )

    for i in range(iterations):
        p[1:-1, 1:-1] = (div[1:-1, 1:-1] + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4
    vel_x[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2])
    vel_y[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1])
    return vel_x, vel_y

def main():
    global X_data
    global Y_data
    pygame.init()
    dt = 0.1
    diff = 0.00001
    w = 250
    h = 200
    fw = w // 4
    fh = h // 4

    screen = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()

    field = np.zeros((fh, fw), dtype=np.float32)
    vel_x = np.zeros_like(field)
    vel_y = np.zeros_like(field)

    field[25, 25] = 0.0

    for _ in range(1500):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        field[random.randint(23, 27), 25] += 30.0
        vel_x[20:30, 20:30] = 3.0

        X_data.append(np.stack([field, vel_x, vel_y], axis=0))

        vel_x = diffuse(vel_x, diff, dt)
        vel_y = diffuse(vel_y, diff, dt)
        set_boundary(vel_x, vel_y)
        vel_x, vel_y = projection(vel_x, vel_y)
        set_boundary(vel_x, vel_y)
        old_vel_x = vel_x.copy()
        old_vel_y = vel_y.copy()
        vel_x = advection(old_vel_x, old_vel_x, old_vel_y, dt)
        vel_y = advection(old_vel_y, old_vel_x, old_vel_y, dt)
        set_boundary(vel_x, vel_y)
        vel_x, vel_y = projection(vel_x, vel_y)
        set_boundary(vel_x, vel_y)
        density = diffuse(field, diff, dt)
        density = advection(density, vel_x, vel_y, dt)
        set_boundary_density(density)
        field = density

        screen.fill((0, 0, 0))
        img = np.sqrt(field) * 40
        img = np.clip(img, 0, 255).astype(np.uint8)
        rgb = np.stack((img, img, img), axis=-1)
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        surface = pygame.transform.scale(surface, (w, h))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(120)

        Y_data.append(np.stack([field, vel_x, vel_y], axis=0))

X_data = list()
Y_data = list()
if __name__ == "__main__":
    main()
Y_data = np.array(Y_data)
X_data = np.array(X_data)
#np.save("X_data.npy", X_data)
#np.save("Y_data.npy", Y_data)
