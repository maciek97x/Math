import sys
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
from time import perf_counter, sleep
import numpy as np
import numba as nb
from scipy import interpolate
from threading import Thread, Lock
from matplotlib import cm

window_width = 1200
window_height = 800

pygame.init()
window = pygame.display.set_mode((window_width, window_height), 0, 32)

pygame.display.set_caption('Ricci flow')

scale = 32

base_curve_delta = 32
dens = 2
curve_delta = base_curve_delta

run = True

class Options(object):
    flow = False
    flow_started = False
    
    draw_curvature = True
    draw_mode = 0
    color_mode = 0

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

def clamp(a, b, c):
    if a < b:
        return b
    if a > c:
        return c
    return a

def ctop(c):
    return (int(c.real*scale), int(c.imag*scale))

def ctofp(c):
    return (c.real*scale, c.imag*scale)

def ptoc(p):
    return complex(*p)/scale

def terminate():
    global compute_thread
    global run
    run = False
    compute_thread.join()
    pygame.quit()
    sys.exit()
    
def compute_curvature(curve):
    x = np.zeros_like(curve)
    y = np.zeros_like(curve)
    z = np.zeros_like(curve)

    x[0] = curve[-1]
    x[1:] = curve[:-1]

    y[:] = curve[:]

    z[-1] = curve[0]
    z[:-1] = curve[1:]

    w = z-x
    w /= y-x
    c = (x-y)*(w-np.abs(w)**2)/2j/w.imag-x
    vec = c + y

    curvature = vec/(vec.real**2 + vec.imag**2)

    curvature[np.isnan(curvature)] = 0

    return curvature

def reduce_curve(curve):
    global curve_delta
    global scale
    i = 0
    deleted = True
    while deleted:
        deleted = False
        while i < curve.shape[0]:
            x = curve[(i - 1)%curve.shape[0]]
            y = curve[i]
            z = curve[(i + 1)%curve.shape[0]]
            if abs(x - y) < curve_delta/scale and abs(y - z) < curve_delta/scale:
                curve = curve[curve != y]
                deleted = True
            i += 1
    return curve

def smoothen(curve, d):
    global curve_delta
    interp_x = [p.real for p in curve]
    interp_y = [p.imag for p in curve]
    
    tck, u = interpolate.splprep([interp_x, interp_y], s=0, per=True)
    x, y = interpolate.splev(np.linspace(0, 1, 10000), tck)
    x = x[::(10000//d)//curve.shape[0]]
    y = y[::(10000//d)//curve.shape[0]]

    curve_delta /= d
    
    new_curve = np.array(x[:-1], dtype=np.complex128) + 1j*np.array(y[:-1], dtype=np.complex128)
    
    print(new_curve.shape)
    return new_curve

def flow(curve, curvature, delta_time):
    velocity = -2*curvature
    curve += velocity*min(delta_time, .01)*base_curve_delta/dens/16

def compute():
    global curve
    global curvature
    global run
    global options
    global compute_steps_per_sec
    global do_update

    time_0_ = perf_counter()

    c_curve = None
    c_curvature = None

    while run:
        delta_time_, time_0_ = perf_counter() - time_0_, perf_counter()
        compute_steps_per_sec = 1./delta_time_

        if not options.flow_started:
            with mutex:
                c_curve = np.copy(curve) if curve is not None else None

        if c_curve is not None and c_curve.shape[0] > 3:
            c_curvature = compute_curvature(c_curve)

        if options.flow and c_curve is not None and c_curvature is not None and c_curve.shape[0] > 10:
            flow(c_curve, c_curvature, delta_time_)

            if c_curve is not None:
                c_curve = reduce_curve(c_curve)

        if do_update:
            with mutex:
                curve = np.copy(c_curve) if c_curve is not None else None
                curvature = np.copy(c_curvature) if c_curvature is not None else None
                do_update = False

        sleep(.001)

curve_list = []
curve = None
curve_ = None
curvature = None
curvature_ = None
smoothened = False
compute_steps_per_sec = 1
do_update = False

options = Options()

mutex = Lock()
compute_thread = Thread(target=compute)

compute_thread.start()

time_0 = perf_counter()

clock = pygame.time.Clock()
while True:
    delta_time, time_0 = perf_counter() - time_0, perf_counter()
    for event in pygame.event.get():
        if event.type == QUIT:
            terminate()
        elif event.type == KEYDOWN:
            if event.key == K_d:
                if not smoothened:
                    curve = smoothen(curve, dens)
                    smoothened = True

            if event.key == K_f:
                options.flow = not options.flow
                options.flow_started = True

            if event.key == K_c:
                options.color_mode += 1
            
            if event.key in (K_1, K_2, K_3):
                options.draw_mode = (K_1, K_2, K_3).index(event.key)
            
            if event.key == K_r:
                curve_list = []
                curve = None
                curve_ = None
                curvature = None
                curvature_ = None
                smoothened = False
                options.flow = False
                options.flow_started = False

        if pygame.mouse.get_pressed()[0] and not options.flow_started:
            mouse_pos = ptoc(pygame.mouse.get_pos())
            if len(curve_list) == 0 or abs(mouse_pos - curve_list[-1]) > base_curve_delta/scale:
                curve_list.append(mouse_pos)
                smoothened = False

                with mutex:
                    curve = np.array(curve_list)

    window.fill((255,)*3)

    with mutex:
        if type(curve) is np.ndarray and type(curvature) is np.ndarray and curve.shape == curvature.shape:
            curve_ = np.copy(curve)
            curvature_ = np.copy(curvature)
        do_update = True

    if options.draw_mode == 2:
        if curve_ is not None and curvature_ is not None and curve_.shape[0] > 3 and options.draw_curvature:
            for i in range(curve_.shape[0]):
                c = curvature_[i]
                cn = curvature_[(i+1)%curvature_.shape[0]]
                cd = (abs(curvature_[i]) + abs(curvature_[(i+1)%curvature_.shape[0]]))/2

                pygame.draw.polygon(window, (0, int(255*(1 - clamp(cd, 0, 1))), int(255*clamp(cd, 0, 1))),
                                    (ctop(curve_[i]), ctop(curve_[i] + 2*c),
                                    ctop(curve_[(i+1)%curve_.shape[0]] + 2*cn), ctop(curve_[(i+1)%curve_.shape[0]])), 0)

    if options.draw_mode in (0, 2):
        if curve_ is not None and curve_.shape[0] > 3:
            pygame.draw.lines(window, (0,)*3, True, np.array(list(map(ctop, curve_)), dtype=np.int32), 4)

    if options.draw_mode == 1:
        if curve_ is not None and curvature_ is not None and curve_.shape[0] > 3 and options.draw_curvature:
            for i in range(curve_.shape[0]):
                c = curvature_[i]
                cn = curvature_[(i+1)%curvature_.shape[0]]
                cd = (abs(curvature_[i]) + abs(curvature_[(i+1)%curvature_.shape[0]]))/2
                
                pygame.draw.line(window, (0, int(255*(1 - clamp(cd, 0, 1))), int(255*clamp(cd, 0, 1))),
                                 ctop(curve_[i]), ctop(curve_[(i+1)%curve_.shape[0]]), 4)

    #draw_text(f'{1/delta_time:10.0f}', 24, (0,)*3, window, (0, 0), anchor='topleft')
    #draw_text(f'{compute_steps_per_sec:10.0f}', 24, (0,)*3, window, (0, 24), anchor='topleft')

    pygame.display.update()
    clock.tick(60)