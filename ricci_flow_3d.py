import sys
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
from time import perf_counter
from math import pi, sin, cos
import numpy as np

import geometry3d

window_width = 1200
window_height = 800

pygame.init()
window = pygame.display.set_mode((window_width, window_height), 0, 32)

pygame.display.set_caption('Ricci flow')

def draw_text(text, size, color, surface, position, anchor=''):
    font = pygame.font.SysFont('Source Code Pro', size*3//4)
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

# osie ukladu wspolrzednych
x_axis = geometry3d.Line([0, 0, 0], [100, 0, 0])
x_axis.color = (255, 0, 0)
y_axis = geometry3d.Line([0, 0, 0], [0, 100, 0])
y_axis.color = (0, 255, 0)
z_axis = geometry3d.Line([0, 0, 0], [0, 0, 100])
z_axis.color = (0, 0, 255)
axes = [x_axis, y_axis, z_axis]

cube = geometry3d.Cube((0, 0, 0), (100, 100, 100))

surf = geometry3d.Surface('D:/Workspace/3dmodels/_moje/h.stl')

# utworzenie kamery
cam = geometry3d.Camera(position=(-256, -256, 128), angle=(pi/4, -pi/16))

cam_move = np.zeros(3)
cam_move_speed = 100
cam_move_boost = 1
cam_rotate = [0, 0]
cam_rotate_speed = 0.5

clock = pygame.time.Clock()
delta_time = 0

free_move = False
skip_mouse_event = False

time_0 = perf_counter()

# glowna petla
pygame.event.get_grab()
while True:
    pygame.event.set_grab(True)
    # obsluga zdarzen
    for event in pygame.event.get():
        if event.type == QUIT:
            terminate()
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                terminate()
            if event.key == K_KP5:
                cam.change_proj_type()
            if event.key == K_x:
                free_move = not free_move
                pygame.mouse.set_visible(not free_move)
                skip_mouse_event = True
            if free_move:
                if event.key == K_w:
                    cam_move[0] = cam_move_speed
                if event.key == K_s:
                    cam_move[0] = -cam_move_speed
                if event.key == K_a:
                    cam_move[1] = cam_move_speed
                if event.key == K_d:
                    cam_move[1] = -cam_move_speed
                if event.key == K_LSHIFT:
                    cam_move[2] = -cam_move_speed
                if event.key == K_SPACE:
                    cam_move[2] = cam_move_speed
                if event.key == K_LCTRL:
                    cam_move_boost = 10
        if event.type == KEYUP:
            if event.key in (K_w, K_s):
                cam_move[0] = 0
            if event.key in (K_a, K_d):
                cam_move[1] = 0
            if event.key in (K_LSHIFT, K_SPACE):
                cam_move[2] = 0
            if event.key == K_LCTRL:
                cam_move_boost = 1
        if event.type == MOUSEMOTION:
            if free_move and not skip_mouse_event:
                cam_rotate[0] = -cam_rotate_speed*event.rel[0]
                cam_rotate[1] = -cam_rotate_speed*event.rel[1]
                #pygame.mouse.set_pos(window_width//2, window_height//2)
            skip_mouse_event = False
    
    delta_time, time_0 = perf_counter() - time_0, perf_counter()
    
    cam.rotate(*map(lambda x: x*delta_time, cam_rotate))
    cam_rotate = [0, 0]
    cam.move(*map(lambda x: x*delta_time, cam_move*cam_move_boost))
    
    for i, obj in enumerate(axes + [cube, surf]):
        obj.set_cam(cam)
        obj.compute(window.get_size())
    
    window.fill((0, 0, 0))
    
    geometry3d.draw(window, *axes, cube, surf)
    
    draw_text(f'fps: {int(1/delta_time): 03d}', 24, (255, 255, 255), window, (8, 8), 'topleft')
    
    pygame.display.update()
    clock.tick(60)