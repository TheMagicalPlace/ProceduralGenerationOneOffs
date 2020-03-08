from dataclasses import dataclass
from tkinter import *
from tkinter.ttk import *
from typing import List
import time
import random
from itertools import chain,product
from math import sin,cos
@dataclass
class NBody:
    mass : int
    radius : int
    x : int
    y : int
    vx : int = 0
    vy : int = 0


class BHTreeNode:

    def __init__(self,center_of_mass,cords :List[List[int]],parent=None,nbodies=None):
        self.center_of_mass : int = center_of_mass
        x,y = list(zip(*cords))
        self.x0,self.x = x
        self.y0,self.y = y

        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.parent = parent
        self.childs = []
        self.bodies = []
        self.content_mass = 0
    def _check_membership(self,bodies):

        for body in bodies:
            self.parent.content_mass += body.mass
            if self.x0 <= body.x < self.x:
                if self.y0 <= body.y < self.y:
                    self.bodies.append(body)

        if len(self.bodies) > 1:
            return True
        elif len(self.bodies) == 1:
            self.content_mass = self.bodies[0].mass
            self._find_com()
        else:
            self._find_com()


    def _find_com(self,depth=None):

        if len(self.bodies) == 1:
            body = self.bodies[0]
            self.center_of_mass = [body.x,body.y]
        elif len(self.childs) == 0:
            self.center_of_mass = [self.x0 + self.dx, self.y0 + self.dy]

        else:
            m_sum = 0
            cm_x = 0
            cm_y = 0
            for c in self.childs:
                cm_x += c.center_of_mass[0]*c.content_mass
                cm_y += c.center_of_mass[1]*c.content_mass
                m_sum += c.content_mass
            self.center_of_mass = [cm_x // m_sum, cm_y // m_sum]

class BHTree:

    def __init__(self,canvas,bodies,mass_range = None):
        self.debug = True
        self.mass_range = mass_range
        self.max_depth = 0
        self.canvas = canvas
        self.canvas.update()
        self.bodies = bodies
        for b in self.bodies:
            self._show_body(b)
        canvas.update()
        cords = [[0,0],[canvas.winfo_width(),canvas.winfo_height()]]
        self.root = BHTreeNode(0,cords)
        self.root.bodies = bodies

        self.subdivide(self.root,0)
        self.root._find_com()
        self._show_center_of_mass(self.root)


    def _show_body(self,body):
        if self.mass_range:
            mass_delta = self.mass_range[1]-self.mass_range[0]
            md = body.mass-self.mass_range[0]
            if mass_delta > 511:
                rgb_change = 511/mass_delta
                rgb = int((md)*rgb_change)
            else:
                rgb_incr_per_mass = 511/mass_delta
                rgb = int((md)*rgb_incr_per_mass)

            if rgb > 256:
                g = 511 - rgb if 511 - rgb > 0 else 0
            else:
                g = 255
            r = rgb if rgb < 255 else 255

            color = f"#{r:02x}{g:02x}{0:02x}"

        else:
            color = '#1f0'

        self.canvas.create_oval(body.x-body.radius,body.y-body.radius,body.x+body.radius,body.y+body.radius,outline=color, fill=color, width=6)
        if self.debug:
            canvas.update()
            time.sleep(0.01)
    def _show_center_of_mass(self,node,depth=0):
        if node.content_mass == 0:
            return
        if node == self.root:
            w = 40
        else:
            w = (self.max_depth-depth)*4+1
        self.canvas.create_oval(node.center_of_mass[0]-w,
                                node.center_of_mass[1]-w,
                                node.center_of_mass[0]+w,
                                node.center_of_mass[1]+w,
                                width=1)
        self.canvas.update()
    def subdivide(self,node,depth):


        self.max_depth = depth

        if self.debug:
            #self.canvas.create_oval(node.x0 + node.dx //2 - 2, node.y0 + node.dy//2 - 2, node.x0 + node.dx//2 + 2, node.y0 + node.dy//2 + 2,outline='#f11',fill='#1f1',width=1)
            self.canvas.create_line(node.x0 + node.dx//2, node.y0, node.x0 + node.dx//2, node.y,width=1)
            self.canvas.create_line(node.x0, node.y0 + node.dy // 2, node.x , node.y0+ node.dy // 2, width=2)
            self.canvas.update()

        midx = node.x0 +node.dx//2;midy=node.y0+node.dy // 2
        nw = [node.x0, node.y0], [midx,midy]
        ne = [[midx, node.y0], [node.x, midy]]
        sw = [[node.x0, midy], [midx, node.y]]
        se = [[midx, midy], [node.x, node.y]]
        next = [nw,ne,sw,se]
        for c in next:
            n = BHTreeNode(0,c,parent=node)
            if n._check_membership(node.bodies):
                self.subdivide(n,depth+1)
            time.sleep(.01)
            node.childs.append(n)
        node._find_com()
        self._show_center_of_mass(node,depth)






def callback(event):
    tree = BHTree(canvas,bodies,(min_mass,max_mass))
    while True:
        break
if __name__ == '__main__':
    bodies = []
    res = []
    ranges = random.sample(range(500),8)
    ranges.sort()
    range_gens = []
    for i in range(0,len(ranges),2):
        range_gens.append(range(ranges[i],ranges[i+1]))
    print(range_gens)
    sets = product(chain(*range_gens),range(0,360))
    for r,theta in sets:
        res.append((int(r*cos(theta)+401),int(r*sin(theta)+401)))
    res = set(res)

    max_mass = 0
    min_mass = 1000
    for aa,bb in random.sample(res,500):
        mass = random.randint(0,1000)
        max_mass = mass if mass > max_mass else max_mass
        min_mass = mass if mass < min_mass else min_mass
        bodies.append(NBody(mass,random.randint(1,3),aa,bb,aa+5,bb+5))
    print(max_mass,min_mass)
    root = Tk()
    main = Frame(root)
    canvas = Canvas(master=main,width=800,height=800)
    canvas.bind('<1>',callback)
    canvas.pack()
    main.pack()


    mainloop()