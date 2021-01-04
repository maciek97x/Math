import sys
import pygame
from pygame.locals import *
from time import perf_counter
import numpy as np
import ripser

window_height = 600
window_width = window_height*2

pygame.init()
window = pygame.display.set_mode((window_width, window_height), 0, 32)

pygame.display.set_caption('Persistent homology')

def terminate():
    pygame.quit()
    sys.exit()

points = []
hom = None

time_0 = perf_counter()

clock = pygame.time.Clock()
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            terminate()

        if event.type == KEYDOWN:
            if event.key == K_r:
                points = []
                hom = None

        if event.type == MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if 16 < mouse_pos[0] < window_height - 16 and 16 < mouse_pos[1] < window_height - 16:
                if event.button == BUTTON_LEFT:
                    points.append(mouse_pos)
                elif event.button == BUTTON_RIGHT:
                    for p in points:
                        if (p[0] - mouse_pos[0])**2 + (p[1] - mouse_pos[1])**2 < 8**2:
                            points.remove(p)
                if len(points) > 1:
                    hom = ripser.ripser(np.array(points))
                else:
                    hom = None
        
            
    delta_time, time_0 = perf_counter() - time_0, perf_counter()
    
    window.fill((255,)*3)
    
    pygame.draw.rect(window, (0,)*3, (16, 16, window_height - 32, window_height - 32), 1)
    pygame.draw.rect(window, (0,)*3, (window_height + 16, 16, window_height - 32, window_height - 32), 1)

    for i in range(16, window_height - 16, 16):
        pygame.draw.line(window, (63,)*3, (window_height + i, window_height - i), (window_height + i + 8, window_height - i - 8))

    for p in points:
        pygame.draw.circle(window, (0, 255, 0), p, 4)
    
    if hom is not None:
        if hom['dgms'][0].shape[0] > 1 and hom['dgms'][1].shape[0] > 0:
            m = max(hom['dgms'][0][:-1].max(), hom['dgms'][1].max())
        elif hom['dgms'][0].shape[0] > 1:
            m = hom['dgms'][0][:-1].max()
        elif hom['dgms'][1].shape[0] > 0:
            m = hom['dgms'][1].max()
        else:
            m = 1

        for p in hom['dgms'][0][:-1]:
            pygame.draw.circle(window, (0, 255, 0), (int(window_height + 16 + p[0]*(window_height - 32)/m), int(window_height - 16 - p[1]*(window_height - 32)/m)), 4)
        for p in hom['dgms'][1]:
            pygame.draw.circle(window, (0, 0, 255), (int(window_height + 16 + p[0]*(window_height - 32)/m), int(window_height - 16 - p[1]*(window_height - 32)/m)), 4)
    
    pygame.display.update()
    
    clock.tick(60)