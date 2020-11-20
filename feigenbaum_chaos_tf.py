import numpy as np
import numba as nb
import sys
import pygame
from pygame.locals import *
from time import perf_counter
from threading import Thread
import tensorflow as tf

u_range = (3, 4)

# rozmiar okna
window_height = 512
window_width = int(window_height*(u_range[1] - u_range[0]))

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


@tf.function
def compute_x(x, y, u):
    x_neg = tf.compat.v2.negative(x)
    one_m_x = tf.compat.v2.add(x_neg, 1.)
    new_x = tf.compat.v2.multiply(x, one_m_x)
    new_x = tf.compat.v2.multiply(new_x, u)
    return new_x
    
@tf.function
def compute_p(x, y, p):
    p = tf.zeros(p.shape)
    out_p = []
    for i, p_0 in enumerate(tf.unstack(p)):
        p_0 = tf.compat.v2.reduce_sum(tf.multiply(tf.cast((y[i] < x), tf.int32),
                                                  tf.cast((x <= y[i+1]), tf.int32)), axis=0)
        out_p.append(p_0)
    p = tf.stack(out_p)
    p = tf.transpose(p)
    p = tf.cast(p, tf.float64)
    out_p = []
    for p_0 in tf.unstack(p):
        p_0 = p_0*2*255./tf.compat.v2.reduce_max(p_0)
        out_p.append(p_0)
    p = tf.stack(out_p)
    p = tf.clip_by_value(p, 0, 255)
    p = tf.cast(p, tf.uint8)
    return p

def computation_loop():
    global p_surf
    global iteration
    global u_range
    n = 1000
    x = np.linspace(0, 1, window_height*2).astype(np.float64)
    u = np.linspace(u_range[0], u_range[1], window_width).astype(np.float64)
    u, x = np.meshgrid(u, x)
    y = np.linspace(0, 1, window_height + 1).astype(np.float64)
    p_3 = np.zeros((window_width, window_height, 3), dtype=np.uint8)
    p_0 = np.zeros((window_width, window_height), dtype=np.uint8)
    while not if_quit:
        p = tf.zeros((window_height, window_width), dtype=tf.int64)
        x = compute_x(x, y, u)
        #with tf.compat.v1.Session() as sess:
        #    x = sess.run(cx)
        p = compute_p(x, y, p)
        #with tf.compat.v1.Session() as sess:
        #    p = sess.run(cp)
        iteration += 1
        p_0 = p
        p_3[:,:,2] = p_3[:,:,1] = p_3[:,:,0] = p_0#//2 + p_3[:,:,0]//2
        p_surf = pygame.surfarray.make_surface(p_3[:,::-1,:].astype(np.uint8))
        
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