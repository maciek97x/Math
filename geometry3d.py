import numpy as np
import numba as nb
import pygame
from math import sin, cos, pi
from stl import mesh

def clamp(a, b, c):
    if a < b:
        return b
    if a > c:
        return c
    return a

# funkcja do rysowania wszystkich obiektow
def draw(surface, *objects):
    # rozmiat okna
    surface_size = np.array(surface.get_size())
    # ustalamy kolejnosc rysowania w zaleznosci od odleglosci od obserwatora
    draw_order = []
    for obj_i, obj in enumerate(objects):
        for d, t, i in obj.draw_order:
            draw_order.append((d, t, obj_i, i))
    draw_order.sort(key=lambda x: x[0], reverse=True)
    
    # rysujemy wszystko po kolei
    for _, t, obj_i, i in draw_order:
        # pierwszy element to odleglosc, nie jest potrzebny, wiec pomijamy
        # t - typ, obj_i - indeks obiektu do ktorego nalezy, i - indeks elementu na liscie obiektu
        if t == 'p':
            # rysowanie punktu
            p = np.array(objects[obj_i].proj_points[i,:2], dtype='int64') + surface_size//2
            d = objects[obj_i].proj_points[i,2]
            pygame.draw.circle(surface,
                               (np.array(objects[obj_i].color)*clamp(256/d, 0, 1)).astype(np.int32),
                               p, 2)
        elif t == 'l':
            # rysowanie odcinka
            p1 = np.array(objects[obj_i].proj_lines[i,0,:2], dtype='int64') + surface_size//2
            p2 = np.array(objects[obj_i].proj_lines[i,1,:2], dtype='int64') + surface_size//2
            d = (objects[obj_i].proj_lines[i,0,2] + objects[obj_i].proj_lines[i,1,2])/2
            pygame.draw.aaline(surface,
                               (np.array(objects[obj_i].color)*clamp(256/d, 0, 1)).astype(np.int32),
                               p1, p2)

class Camera(object):
    def __init__(self, position=[0, 0, 0], angle=[0, 0]):
        self.__origin = np.array(position, dtype='float64')
        self.__angle = np.array(angle, dtype='float64')
        self.__proj_type = 0
        self.__proj_types = ['orthographic', 'perspective']
    
    def __getattr__(self, name):
        if name == 'origin':
            return self.__origin
        if name == 'dir_vector':
            # wektor, w kierunku ktorego patrzy kamera
            return np.array([cos(self.__angle[1])*cos(self.__angle[0]),
                             cos(self.__angle[1])*sin(self.__angle[0]),
                             sin(self.__angle[1])])
        if name == 'proj_type':
            return self.__proj_type
        if name == 'proj_types':
            return self.__proj_types
    
    def change_proj_type(self):
        self.__proj_type += 1
        self.__proj_type %= len(self.__proj_types)
    
    def move(self, dx, dy, dz):
        # dx - poruszanie do przodu/tylu
        # dy - w boki
        # dz - gora/dol
        vx = self.dir_vector
        vy = np.zeros(3)
        vy[0] = -vx[1]
        vy[1] = vx[0]
        self.__origin += dx*vx
        self.__origin += dy*vy
        self.__origin[2] += dz
    
    def rotate(self, dx, dy):
        self.__angle += np.array([dx, dy])
        if self.__angle[1] > pi/2:
            self.__angle[1] = pi/2
        if self.__angle[1] < -pi/2:
            self.__angle[1] = -pi/2
        self.__angle[0] %= 2*pi
    
class Solid(object):
    def __init__(self, origin=np.zeros(3), points=[], lines=[]):
        self.__origin = np.array(origin, dtype='float64')
        
        # elementy, wspolrzedne wzgledem srodka
        self.__points = np.array(points)
        self.__lines = np.array(lines)
        
        # elementy po rzucie, trzecia wspolrzedna to odleglosc od obserwatora
        self.__proj_points = np.zeros((self.__points.shape[0], 3))
        self.__proj_lines = np.zeros((self.__lines.shape[0], 2, 3))
        
        # parametry do rzutu
        self.__proj_origin = -10*np.ones(3, dtype='float64')
        self.__proj_vector = np.ones(3, dtype='float64')
        self.__proj_type = 0
        self.__proj_types = ['orthographic', 'perspective']
        
        # kolejnosc rysowania elementow
        self.__draw_order = []
        
        self.__color = np.ones(3)*255
        
        self.__cam_changed = True
        self.__obj_changed = True
    
    def __setattr__(self, name, value):
        if name == 'color':
            self.__color = np.array(value)
        super(Solid, self).__setattr__(name, value)
    
    def __getattr__(self, name):
        if name == 'origin':
            return self.__origin
        if name == 'color':
            return self.__color
        if name == 'draw_order':
            return self.__draw_order
        if name == 'proj_points':
            return self.__proj_points
        if name == 'proj_lines':
            return self.__proj_lines
    
    # ustawienie kamery
    def set_cam(self, cam):
        if np.any(self.__proj_origin != cam.origin) or\
           np.any(self.__proj_vector != cam.dir_vector) or\
           np.any(self.__proj_type != cam.proj_type):
            self.__cam_changed = True
            self.__proj_origin = np.copy(cam.origin)
            self.__proj_vector = cam.dir_vector
            self.__proj_type = cam.proj_type
            self.__proj_types = cam.proj_types

    def rotate(self, angle):
        if self.__points.shape[0] > 0:
            self.__points = Solid.__rotate_points(self.__points, angle)
        if self.__lines.shape[0] > 0:
            self.__lines[:,0,:] = self.__rotate_points(self.__lines[:,0,:], angle)
            self.__lines[:,1,:] = self.__rotate_points(self.__lines[:,1,:], angle)
        if angle % (2*pi):
            self.__obj_changed = True
    
    def translate(self, vector):
        self.__origin += vector
        if Solid.__d2(vector) > 0:
            self.__obj_changed = True
    
    # obliczanie rzutu i kolejnosci rysowania
    def compute(self, surface_size):
        # obliczamy tylko jak poruszy sie kamera lub obiekt
        if self.__cam_changed or self.__obj_changed:
            self.__compute_projection()
            self.__compute_draw_order(surface_size)
            self.__cam_changed = False
            self.__obj_changed = False
    
    def draw(self, surface):
        surface_size = np.array(surface.get_size())
        for _, t, i in self.__draw_order:
            try:
                if t == 'p':
                    p = np.array(self.__proj_points[i,:2], dtype='int64') + surface_size//2
                    pygame.draw.circle(surface, self.__color.astype(np.int32), p, 2)
                elif t == 'l':
                    p1 = np.array(self.__proj_lines[i,0,:2], dtype='int64') + surface_size//2
                    p2 = np.array(self.__proj_lines[i,1,:2], dtype='int64') + surface_size//2
                    pygame.draw.aaline(surface, self.__color.astype(np.int32), p1, p2)
            except:
                pass
    
    # odlegosc eulidesowa
    @staticmethod
    def __d2(p1, p2=None):
        if p2 is None:
            p2 = np.zeros(p1.shape)
        return np.linalg.norm(p1 - p2)
    
    # obrot wokol osi X
    @staticmethod
    @nb.guvectorize(['float64[:,:], float64, float64[:,:]'], '(n, k),()->(n, k)', nopython=True)
    def __rotate_points(points, angle, rotated_points):
        for i in range(points.shape[0]):
            rotated_points[i,0] = points[i,0]
            rotated_points[i,1] = points[i,1]*cos(angle) + points[i,2]*sin(angle)
            rotated_points[i,2] = -points[i,1]*sin(angle) + points[i,2]*cos(angle)
    
    @staticmethod
    @nb.guvectorize(['float64[:,:], float64[:], float64[:], float64[:,:]'], '(n, k),(k),(k)->(n, k)', nopython=True)
    def __perspective_projection(points, origin, vector, proj_points):
        for i in range(points.shape[0]):
            p = np.copy(points[i,:])
            # przesuniecie do srodka
            p -= origin
            # unormowanie wektora normalnego do plaszczyzny rzutu
            unit_vector = vector/((vector[0]**2 + vector[1]**2 + vector[2]**2)**.5)
            # odleglosc do plaszczyzny rzutu
            dist_to_plane = unit_vector[0]*p[0] + unit_vector[1]*p[1] + unit_vector[2]*p[2]
            # obliczamy wsporzedne punktu przeciecia plaszczyzny z odcinkiem p - srodek rzutu
            p -= p*(dist_to_plane - 500)/dist_to_plane
            
            # baza plaszczyzny
            e1 = np.array([vector[1], -vector[0], 0])
            e1 /= (e1[0]**2 + e1[1]**2 + e1[2]**2)**.5
            
            e2 = np.zeros(3)
            e2[0] = unit_vector[1]*e1[2] - unit_vector[2]*e1[1]
            e2[1] = -unit_vector[0]*e1[2] + unit_vector[2]*e1[0]
            e2[2] = unit_vector[0]*e1[1] - unit_vector[1]*e1[0]
            
            # rzut na wektory bazy plaszczyzny
            proj_points[i,0] = e1[0]*p[0] + e1[1]*p[1] + e1[2]*p[2]
            proj_points[i,1] = e2[0]*p[0] + e2[1]*p[1] + e2[2]*p[2]
            proj_points[i,2] = dist_to_plane

    @staticmethod
    @nb.guvectorize(['float64[:,:], float64[:], float64[:], float64[:,:]'], '(n, k),(k),(k)->(n, k)', nopython=True)
    def __orthographic_projection(points, origin, vector, proj_points):
        for i in range(points.shape[0]):
            p = np.copy(points[i,:])
            # tak samo jak wyzej ale nie obliczamy punktu przeciecia z plaszczyzna
            # tylko rzutujemy p na baze plaszczyzny
            p -= origin
            #unit_vector = vector/np.linalg.norm(vector)
            unit_vector = vector/((vector[0]**2 + vector[1]**2 + vector[2]**2)**.5)
            #dist_to_plane = unit_vector.dot(p)
            dist_to_plane = unit_vector[0]*p[0] + unit_vector[1]*p[1] + unit_vector[2]*p[2]
            
            e1 = np.array([vector[1], -vector[0], 0])
            #e1 /= np.linalg.norm(e1)
            e1 /= (e1[0]**2 + e1[1]**2 + e1[2]**2)**.5
            
            #e2 = np.cross(unit_vector, e1)
            e2 = np.zeros(3)
            e2[0] = unit_vector[1]*e1[2] - unit_vector[2]*e1[1]
            e2[1] = -unit_vector[0]*e1[2] + unit_vector[2]*e1[0]
            e2[2] = unit_vector[0]*e1[1] - unit_vector[1]*e1[0]
            
            proj_points[i,0] = e1[0]*p[0] + e1[1]*p[1] + e1[2]*p[2]
            proj_points[i,1] = e2[0]*p[0] + e2[1]*p[1] + e2[2]*p[2]
            proj_points[i,2] = dist_to_plane
    
    def __compute_projection(self):
        if self.__proj_types[self.__proj_type] == 'orthographic':
            if self.__points.shape[0] > 0:
                # rzutowanie wierzcholkow
                self.__proj_points = self.__orthographic_projection(self.__points + self.__origin, self.__proj_origin, self.__proj_vector)
            if self.__lines.shape[0] > 0:
                # rzutowanie krawedzi, najpierw punkt poczatkowy, potem koncowy
                self.__proj_lines[:,0,:] = self.__orthographic_projection(self.__lines[:,0,:] + self.__origin, self.__proj_origin, self.__proj_vector)
                self.__proj_lines[:,1,:] = self.__orthographic_projection(self.__lines[:,1,:] + self.__origin, self.__proj_origin, self.__proj_vector)
            
        elif self.__proj_types[self.__proj_type] == 'perspective':
            if self.__points.shape[0] > 0:
                self.__proj_points = self.__perspective_projection(self.__points + self.__origin, self.__proj_origin, self.__proj_vector)
            if self.__lines.shape[0] > 0:
                self.__proj_lines[:,0,:] = self.__perspective_projection(self.__lines[:,0,:] + self.__origin, self.__proj_origin, self.__proj_vector)
                self.__proj_lines[:,1,:] = self.__perspective_projection(self.__lines[:,1,:] + self.__origin, self.__proj_origin, self.__proj_vector)
    
    def __compute_draw_order(self, surface_size):
        draw_order = []
        # dodanie punktow
        for i in range(self.__proj_points.shape[0]):
            d = self.__proj_points[i,2]
            if d > 0 and -surface_size[0]/2 <= self.__proj_points[i,0] < surface_size[0]/2 and\
               -surface_size[1]/2 <= self.__proj_points[i,1] < surface_size[1]/2:
                draw_order.append((d, 'p', i))
        # dodanie krawedzi
        for i in range(self.__proj_lines.shape[0]):
            d = np.min(self.__proj_lines[i,:,2])
            if d > 0 and ((-surface_size[0]/2 <= self.__proj_lines[i,0,0] < surface_size[0]/2 and\
                           -surface_size[1]/2 <= self.__proj_lines[i,0,1] < surface_size[1]/2) or\
                          (-surface_size[0]/2 <= self.__proj_lines[i,1,0] < surface_size[0]/2 and\
                           -surface_size[1]/2 <= self.__proj_lines[i,1,1] < surface_size[1]/2)):
                draw_order.append((d, 'l', i))
        # sortowanie po odleglosci
        draw_order.sort(key=lambda x: x[0], reverse=True)
        self.__draw_order = draw_order
        
    def __repr__(self):
        return f'Solid({self.__origin}, {self.__points}, {self.__lines})'

class Cube(Solid):
    def __init__(self, origin, dims):
        self.__dims = dims
        origin = np.array(origin)
        dims = np.array(dims)
        points = []
        lines = []
        for dx in [-dims[0]//2, dims[0]//2]:
            for dy in [-dims[1]//2, dims[1]//2]:
                for dz in [-dims[2]//2, dims[2]//2]:
                    points.append(np.array([dx, dy, dz], dtype='float64'))
        for i in range(len(points)):
            for j in range(i):
                # jezeli punkty maja 2 takie same wspolrzedne to je laczymy
                if (points[i] == points[j]).sum() == 2:
                    line = np.zeros((2, 3))
                    line[0,:] = points[i]
                    line[1,:] = points[j]
                    lines.append(line)
        super(Cube, self).__init__(origin=origin, points=points, lines=lines)
        
    def __repr__(self):
        return f'Cube({self.__origin}, {self.__dims})'

class Line(Solid):
    def __init__(self, start, end):
        self.__start = start
        self.__end = end
        points = []
        lines = [[np.zeros(3), np.array(end) - np.array(start)]]
        super(Line, self).__init__(origin=start, points=points, lines=lines)
    
    def __repr__(self):
        return f'Line({self.__start}, {self.__end})'

class Stars(Solid):
    def __init__(self, origin, dims, star_count):
        self.__origin = origin
        self.__dims = dims
        self.__star_cound = star_count
        origin = np.array(origin)
        dims = np.array(dims)
        points = []
        lines = []
        for _ in range(star_count):
            points.append((np.random.random(3) - .5)*dims)
        super(Stars, self).__init__(origin=origin, points=points, lines=lines)
    
    def __repr__(self):
        return f'Stars({self.__origin}, {self.__dims}, {self.__star_cound})'     

class Surface(Solid):
    def __init__(self, stl_filename):
        m = mesh.Mesh.from_file(stl_filename)

        points = np.concatenate([m.v0, m.v1, m.v2], axis=0)
        points = np.unique(points, axis=0)

        #points = points[np.random.randint(0, points.shape[0] - 1, 1000)]

        origin = np.zeros(3)

        lines = []
        
        super(Surface, self).__init__(origin=origin, points=points, lines=lines)