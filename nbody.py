from dataclasses import dataclass
from tkinter import *
from tkinter.ttk import *
from typing import List
import time
import random
from itertools import chain, product
from math import sin, cos
import numpy as np

from math import sqrt


class NBody:
    G = 6.674 * (1 / 10 ** (11))

    id = (i for i in range(100000))

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        if self._id == other:
            return True
        else:
            return False

    def __init__(self, mass, radius, x, y, static=False):
        self.static = static
        self._id = next(NBody.id)
        self.mass: int = mass
        self.radius: int = radius
        self.x: int = x
        self.y: int = y
        self.velocity = np.array([random.randint(-100,100)/100,random.randint(-100,100)/100],dtype='float64')
        self.acceleration = np.array([0,0],dtype='float64')
        self.center_of_mass = np.array([x, y],dtype='float64')

        self.F = np.array([0, 0], dtype='float64')

    def get_magnitude(self, body):
        getr = lambda body: self.center_of_mass - body.center_of_mass

    def get_distance_scalar(self, body):
        getr = lambda body: self.center_of_mass - body.center_of_mass
        return sqrt(sum(np.vectorize(getr)(body) ** 2))

    def calculate_force(self, body):
        getr = lambda body: self.center_of_mass - body.center_of_mass

        r = self.get_distance_scalar(body)
        ru = getr(body) / r

        F = -NBody.G * self.mass * body.mass * ru / r ** 2
        self.F += F


    def calculate_new_motion_properties(self,t=1):
        # t = 1
        if self.static:
            return
        self.acceleration = self.F/self.mass
        self.velocity = self.acceleration*t+self.velocity
        self.center_of_mass = self.center_of_mass+self.velocity*t-(0.5)*self.acceleration*t**2
        self.x,self.y = self.center_of_mass

class BHTreeNode:

    def __init__(self, cords: List[List[int]], parent=None, nbodies=None):

        x, y = list(zip(*cords))
        self.x0, self.x = x
        self.y0, self.y = y
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.center_of_mass = np.array([self.x0 + self.dx, self.y0 + self.dy])
        self.s = self.dx
        self.parent = parent
        self.childs = []
        self.bodies = []
        self.mass = 0

    def _check_membership(self, bodies):

        for body in bodies:
            self.parent.mass += body.mass
            if self.x0 <= body.x < self.x:
                if self.y0 <= body.y < self.y:
                    self.bodies.append(body)

        if len(self.bodies) > 1:
            return True
        elif len(self.bodies) == 1:
            self.mass = self.bodies[0].mass
            self._find_com()
        else:
            self._find_com()

    def _find_com(self, depth=None):

        if len(self.bodies) == 1:
            body = self.bodies[0]
            self.center_of_mass = np.array([body.x, body.y])
        elif len(self.childs) == 0:
            self.center_of_mass = np.array([self.x0 + self.dx, self.y0 + self.dy])

        else:
            m_sum = 0
            cm_x = 0
            cm_y = 0
            for c in self.childs:
                cm_x += c.center_of_mass[0] * c.mass
                cm_y += c.center_of_mass[1] * c.mass
                m_sum += c.mass
            self.center_of_mass = np.array([cm_x // m_sum, cm_y // m_sum])



class BHTree:

    def __init__(self, canvas, bodies, mass_range=None):
        self.debug = False
        self.mass_range = mass_range
        self.max_depth = 0
        self._body_canvas_objs = {}
        self._tree_line_objs = []
        self.canvas = canvas
        self.canvas.update()
        self.bodies = bodies
        for b in self.bodies:
            self._show_body(b)
        canvas.update()
        cords = [[0, 0], [canvas.winfo_width(), canvas.winfo_height()]]
        self.s = sqrt(canvas.winfo_width() ** 2 + canvas.winfo_height() ** 2)
        self.root = BHTreeNode(cords)
        self.root.bodies = bodies
        self.ratio = 0.5  # min ratio of s/d
        self.subdivide(self.root, 0)
        self.root._find_com()
        self._show_debug_info(self.root)

    def _update_body_positions(self,body):
        bd,tx = self._body_canvas_objs[body]
        self.canvas.coords(bd,body.x - body.radius, body.y - body.radius, body.x + body.radius, body.y + body.radius)
        self.canvas.coords(tx, body.x,body.y)

    def examine_bodies(self):
        for body in self.bodies:
            if body.static: continue
            self.active_body = body
            self.run_nbody(self.root)
            body.calculate_new_motion_properties()
            self._update_body_positions(body)
        self.canvas.update()

    def run_nbody(self, parent):

        last_colors = []
        s = parent.s
        d = self.active_body.get_distance_scalar(parent)

        # highlights all bodies in examined node
        if self.debug:
            for body in parent.bodies:
                bd,_ = self._body_canvas_objs[body]
                last_colors.append(self.canvas.itemcget(bd,'outline'))
                self.canvas.itemconfig(bd,fill ='#ff69b4',outline='#ff69b4')
            self.canvas.update()

        # external node i.e. node only contains one body
        if len(parent.bodies) == 1:
            # body does not act on itself
            if parent.bodies[0] == self.active_body:
                pass
            else:
                if parent.bodies[0].static: return
                self.active_body.calculate_force(parent.bodies[0])

        # Empty nodes exert no force
        elif len(parent.bodies) == 0:
            pass
        # node is already at center of mass
        elif d == 0:
            pass
        # if distance is far enough to approximate treat nodes content bodies as one for force
        elif s / d < self.ratio:
            self.active_body.calculate_force(parent)

        # else examine children
        else:
            # returns to normal color
            if self.debug:
                for i, body in enumerate(parent.bodies):
                    bd, _ = self._body_canvas_objs[body]
                    self.canvas.itemconfig(bd, outline=last_colors[i])
                self.canvas.update()
            for child in parent.childs:
                self.run_nbody(child)
        # returns to normal color
        if self.debug:
            for i,body in enumerate(parent.bodies):
                bd,_ = self._body_canvas_objs[body]
                self.canvas.itemconfig(bd,outline=last_colors[i])
            self.canvas.update()

    def _show_body(self, body):
        if self.mass_range:
            mass_delta = self.mass_range[1] - self.mass_range[0]
            md = body.mass - self.mass_range[0]
            if mass_delta > 511:
                rgb_change = 511 / mass_delta
                rgb = int((md) * rgb_change)
            else:
                rgb_incr_per_mass = 511 / mass_delta
                rgb = int((md) * rgb_incr_per_mass)

            if rgb > 256:
                g = 511 - rgb if 511 - rgb > 0 else 0
            else:
                g = 255
            r = rgb if rgb < 255 else 255

            color = f"#{r:02x}{g:02x}{0:02x}"

        else:
            color = '#1f0'

        bd = self.canvas.create_oval(body.x - body.radius, body.y - body.radius, body.x + body.radius, body.y + body.radius,
                                outline=color, fill=color, width=6)
        tx = self.canvas.create_text(body.x, body.y, text=str(body._id))

        self._body_canvas_objs[body] = (bd,tx)

        if self.debug:
            canvas.update()
            time.sleep(0.01)

    def _show_debug_info(self, node, depth=0):

        dp = 0
        a = self.canvas.create_line(node.x0 + node.dx // 2, node.y0, node.x0 + node.dx // 2, node.y, width=1)
        b = self.canvas.create_line(node.x0, node.y0 + node.dy // 2, node.x, node.y0 + node.dy // 2, width=2)
        if node.mass == 0:
            self._tree_line_objs+= [a,b]
        else:
            n = node
            while n.parent:
                dp += 1
                n =n.parent
            w = 40-(dp*8)
            c = self.canvas.create_oval(node.center_of_mass[0] - w,
                                    node.center_of_mass[1] - w,
                                    node.center_of_mass[0] + w,
                                    node.center_of_mass[1] + w,
                                    width=1)
            self._tree_line_objs += [a, b,c]
        self.canvas.update()

    def subdivide(self, node, depth):

        midx = node.x0 + node.dx // 2;
        midy = node.y0 + node.dy // 2
        nw = [node.x0, node.y0], [midx, midy]
        ne = [[midx, node.y0], [node.x, midy]]
        sw = [[node.x0, midy], [midx, node.y]]
        se = [[midx, midy], [node.x, node.y]]
        next = [nw, ne, sw, se]
        for c in next:
            n = BHTreeNode(c, parent=node)
            if n._check_membership(node.bodies):
                self.subdivide(n, depth + 1)
            if n.bodies:
                node.childs.append(n)
            time.sleep(.01)
        node._find_com()

        if self.debug:
            self._show_debug_info(node, depth)


def callback(event):
    tree = BHTree(canvas, bodies, (min_mass, max_mass))
    while True:
        tree.examine_bodies()
        time.sleep(0.05)

if __name__ == '__main__':
    bodies = []
    res = []
    ranges = random.sample(range(500), 8)
    ranges.sort()
    range_gens = []
    for i in range(0, len(ranges), 2):
        range_gens.append(range(ranges[i], ranges[i + 1]))
    # range_gens = [range(400)]
    print(range_gens)
    sets = product(chain(*range_gens), range(0, 360))
    for r, theta in sets:
        res.append((int(r * cos(theta) + 501), int(r * sin(theta) + 501)))
    res = set(res)

    max_mass = 0
    min_mass = 10 ** 20
    mrange = [10 ** 4, 10 ** 7]
    for aa, bb in random.sample(res, 50):
        mass = random.randint(*mrange)
        max_mass = mass if mass > max_mass else max_mass
        min_mass = mass if mass < min_mass else min_mass
        bodies.append(NBody(mass, random.randint(1, 3), aa, bb))
    for aa, bb in random.sample(list(product(range(500), range(0, 360))), 00):
        mass = random.randint(*mrange)
        aa, bb = (int(aa * cos(bb) + 501), int(aa * sin(bb) + 501))
        max_mass = mass if mass > max_mass else max_mass
        min_mass = mass if mass < min_mass else min_mass
        bodies.append(NBody(mass, random.randint(1, 3), aa, bb))

    bodies.append(NBody(10**8,20,508,508,True))

    root = Tk()
    main = Frame(root)
    canvas = Canvas(master=main, width=1000, height=1000)
    canvas.bind('<1>', callback)
    canvas.pack()
    main.pack()
    callback('')
    bodies[0].calculate_force(bodies[1])

    mainloop()