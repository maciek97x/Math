import numpy as np
import numba as nb
import sys
import pygame
from pygame.locals import *
from time import perf_counter
from threading import Thread

u_range = (2, 4)
animate = False
animation_time = .5

# rozmiar okna
window_height = 256
window_width = window_height*(u_range[1] - u_range[0])

# inicjacja okna
pygame.init()
window = pygame.display.set_mode((window_width, window_height), 0, 32)

# tytul okna
pygame.display.set_caption('Chaos')

# funkcja do rysowania tekstu w oknie
def draw_text(text, size, color, surface, position, anchor=''):
    font = pygame.font.SysFont('consolas', size*3//4)
    textobj = font.render(text, 1, color)
    textrect = textobj.get_rect()
    if anchor == 'bottomleft':
        textrect.bottomleft = position
    elif anchor == 'topleft':
        textrect.topleft = position
    elif anchor == 'bottomright':
        textrect.bottomright = position
    elif anchor == 'topright':
        textrect.topright = position
    else:
        textrect.center = position
    surface.blit(textobj, textrect)

def terminate():
    pygame.quit()
    sys.exit()

if_quit = False
p_surf = pygame.Surface((window_width, window_height))
iteration = 0
def computation_loop():
    global p_surf
    global iteration
    global u_range
    global animation_time
    n = 1000
    x = np.linspace(0, 1, window_height).astype(np.float64)
    u = np.linspace(u_range[0], u_range[1], window_width).astype(np.float64)
    u, x = np.meshgrid(u, x)
    y = np.linspace(0, 1, window_height)
    p = np.zeros((window_height, window_width, 3))
    if animate:
        x_prev = np.copy(x)
        x_draw = np.copy(x)
        time_step = animation_time
        time_0 = perf_counter()
        while not if_quit:
            if perf_counter() - time_0 > time_step:
                x_prev = np.copy(x)
                x = u*x*(1-x)
                iteration += 1
                time_0 = perf_counter()
            x_draw = x_prev*(time_step - (perf_counter() - time_0))/time_step + x*(perf_counter() - time_0)/time_step
            for i in range(window_height - 1):
                p[i,:,0] = np.sum((y[i] < x_draw) & (x_draw <= y[i+1]), axis=0)
                #p[i,:,0] = 1/(1. + np.sum(np.square(x - y[i]), axis=0))
            for i in range(window_width):
                p[:,i,0] = p[:,i,0]*255/np.max(p[:,i,0])
                p[:,i,0][p[:,i,0] > 255] = 255
            p[:,:,2] = p[:,:,1] = p[:,:,0]
            p = p[::-1,]
            p_surf = pygame.surfarray.make_surface(p.astype(np.uint8).transpose(1, 0, 2))
        
    else:
        while not if_quit:
            x = u*x*(1-x)
            iteration += 1
            p[:] = 0
            for i in range(window_height - 1):
                p[i,:,0] = np.sum((y[i] < x) & (x <= y[i+1]), axis=0)
                #p[i,:,0] = 1/(1. + np.sum(np.square(x - y[i]), axis=0))
            for i in range(window_width):
                p[:,i] = p[:,i]*255/np.max(p[:,i])
                p[:,i][p[:,i] > 255] = 255
            p[:,:,2] = p[:,:,1] = p[:,:,0]
            p = p[::-1,]
            p_surf = pygame.surfarray.make_surface(p.astype(np.uint8).transpose(1, 0, 2))
        
Thread(target=computation_loop).start()

time_0 = perf_counter()

# glowna petla
pygame.event.get_grab()
clock = pygame.time.Clock()
while True:
    # obsluga zdarzen
    for event in pygame.event.get():
        if event.type == QUIT:
            if_quit = True
            terminate()
            
    # obliczanie czasu miedzy klatkami
    delta_time, time_0 = perf_counter() - time_0, perf_counter()
    
    # wypelnienie okna czarnym kolorem
    window.fill((0, 0, 0))
    
    # narysowanie wszystkich obiektow
    window.blit(p_surf, (0, 0))
    draw_text(f' Iteration: {iteration:6d}', 16, (255,)*3, window, (8, 8), 'topleft')
    # odswiezenie okna
    pygame.display.update()
    
    # ograniczenie liczby klatek do 60 na sekunde
    clock.tick(60)