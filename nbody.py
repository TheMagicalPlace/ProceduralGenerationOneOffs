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

SCALE = 1/10 # scaling factor




class NBody:
    """Class for the creation and data tracking of particle objects"""

    # Gravitational Constant
    G = 6.674 * (1 / 10 ** (11))

    # Used to generate a unique id for each body
    __id = (i for i in range(100000))


    def __hash__(self):
        return self._id

    def __eq__(self, other):
        if self._id == other:
            return True
        else:
            return False

    def __init__(self, mass, radius, x, y, static=False):
        self.static = static            # specifies if the body should not have momentum calculations done on it
        self._id = next(NBody.__id)     # unique identifier
        self.mass: int = mass
        self.radius: int = radius
        self.x: int = x                 # canvas x position
        self.y: int = y                 # canvas y position

        # Initial velocity vector is determined randomly
        self.velocity = np.array([random.randint(-100, 100) / 100*SCALE,
                                  random.randint(-100, 100) / 100*SCALE],
                                 dtype='float64')

        self.acceleration = np.array([0, 0], dtype='float64')
        self.center_of_mass = np.array([x, y], dtype='float64')
        self.F = np.array([0, 0], dtype='float64')



    def get_distance_scalar(self, body):
        """square distance scalar between """
        getr = lambda body:  body.center_of_mass-self.center_of_mass
        return sqrt(sum(np.vectorize(getr)(body) ** 2))

    def calculate_force(self, body):
        """ Calculates the vector force on the object body by another body using Newtons law
        of gravitation. Force is calculated as a sum of each other body/node and is reset to zero each new round.

        F = G*m1*m2/|r^2| <x2-x1,y2-y1>
        """

        # returns distance vector between bodies
        getr = lambda body:  body.center_of_mass-self.center_of_mass

        # distance scalar between bodies
        r = self.get_distance_scalar(body)

        # position vector
        ru = getr(body) / r

        F = NBody.G * self.mass * body.mass * ru / r ** 2
        self.F += F

    def calculate_new_motion_properties(self, t=.010):
        """Calculates the current acceleration,velocity,and position of the center of mass for the body"""

        # static bodies do not have meaningful amounts of force exerted on them, so no calculations are done
        if self.static:
            return
        self.acceleration = self.F / self.mass
        self.velocity = self.acceleration * t + self.velocity
        self.center_of_mass = self.center_of_mass + self.velocity * t - (0.5) * self.acceleration * t ** 2
        self.x, self.y = self.center_of_mass


class BHTreeNode:
    """Node used for approximating acting force using the Barnes-Hut approximation"""


    def __init__(self, cords: List[List[int]], parent=None, nbodies=None):

        # determining values used to update canvas
        x, y = list(zip(*cords))
        self.x0, self.x = x
        self.y0, self.y = y
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]

        self.center_of_mass = np.array([self.x0 + self.dx, self.y0 + self.dy])

        # size of node
        self.s = self.dx
        self.parent = parent
        self.childs = []
        self.bodies = []
        self.mass = 0

    def _check_membership(self, bodies):
        """ Finding the bodies contained within the boundaries of the node and finding the nodes center of mass"""
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
        """ Find the nodes center of mass"""
        # center of mass is just the bodies position if only one body
        if len(self.bodies) == 1:
            body = self.bodies[0]
            self.center_of_mass = np.array([body.x, body.y], dtype='float64')
        # if no bodies are contained, the center of mass is just taken as the center of the node (exerts no force)
        elif len(self.childs) == 0:
            self.center_of_mass = np.array([self.x0 + self.dx, self.y0 + self.dy], dtype='float64')

        # for multiple bodies, calculate the center of mass normally
        else:
            m_sum = 0
            cm_x = 0
            cm_y = 0
            for c in self.childs:
                cm_x += c.center_of_mass[0] * c.mass
                cm_y += c.center_of_mass[1] * c.mass
                m_sum += c.mass
            self.center_of_mass = np.array([cm_x // m_sum, cm_y // m_sum], dtype='float64')


class BHTree:


    info_canvas = None
    tracked_bodies = None

    def __init__(self, canvas, bodies, mass_range=None):
        self.debug = True  # enables drawing of tree boxes and centers of mass
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
        self.ratio = 0.2  # min ratio of s/d
        self.subdivide(self.root, 0)
        self.root._find_com()
        self._show_debug_info(self.root)

    def _update_body_positions(self, body):
        bd, tx, momentum_radius = self._body_canvas_objs[body]
        self.canvas.coords(momentum_radius,
                           body.x + body.radius,
                           body.y + body.radius,
                           body.x + body.radius + body.velocity[0],
                           body.y + body.radius + body.velocity[1]
                           )
        self.canvas.coords(bd, body.x - body.radius, body.y - body.radius, body.x + body.radius, body.y + body.radius)
        if self.debug:
            if body._id in list(self.tracked_bodies.keys()):
                txid= self.tracked_bodies[body._id]
                self.info_canvas.itemconfigure(txid, text=f"Body #{body._id}\n"
                                                 f"Velocity : {body.velocity[0]:.2e},{body.velocity[1]:.2e}\n"
                                                 f"Mass : {body.mass:.2e}\n",anchor=CENTER)

    def examine_bodies(self):
        for body in self.bodies:
            if body.static: continue                        # 'static' bodies are those massive enough by comparison
                                                            # that negligable force is exerted on them by the other bodies

            body.F = np.array([0,0],dtype='float64')        # need to clear out force from last round
            self.active_body = body
            self.run_nbody(self.root)
            body.calculate_new_motion_properties()
            self._update_body_positions(body)
            if body._id in self.tracked_bodies.keys():
                _,tx,_ = self._body_canvas_objs[body._id]
                self.canvas.coords(tx,body.x,body.y)
                self.canvas.itemconfigure(tx ,text=f"id : {body._id}")

        self.canvas.update()

    def run_nbody(self, parent):
        individual_debug = False  # set manually to debug individual bodies

        last_colors = []
        s = parent.s
        d = self.active_body.get_distance_scalar(parent)

        # highlights all bodies in examined node

        ## DEBUG CODE ##
        if self.debug and individual_debug:  # only used for examining individual bodies, needs to be set manually
            for body in parent.bodies:
                bd, _,_ = self._body_canvas_objs[body]
                last_colors.append(self.canvas.itemcget(bd, 'outline'))
                self.canvas.itemconfig(bd, fill='#ff69b4', outline='#ff69b4')
        ## END DEBUG CODE ##

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
            for child in parent.childs:
                self.run_nbody(child)

        ## DEBUG CODE ##
        # returns to normal color
        if self.debug and individual_debug:
            for i, body in enumerate(parent.bodies):
                bd, _,_ = self._body_canvas_objs[body]
                self.canvas.itemconfig(bd, outline=last_colors[i])
            self.canvas.update()
        ## END DEBUG CODE ##

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

        bd = self.canvas.create_oval(body.x - body.radius, body.y - body.radius, body.x + body.radius,
                                     body.y + body.radius,
                                     outline=color, fill=color, width=6)

        if body._id in self.tracked_bodies.keys():
            tx = self.canvas.create_text(body.x, body.y, text=f"id : {body._id}\nvelocity : {body.velocity}")
        else:
            tx = []

        momentum_arrow = self.canvas.create_line(body.x + body.radius,
                                                 body.y + body.radius,
                                                 body.x + body.radius + body.velocity[0],
                                                 body.y + body.radius + body.velocity[1],
                                                 arrow=LAST
                                                 )

        self._body_canvas_objs[body] = (bd, tx, momentum_arrow)

    def _show_debug_info(self, node, show_mass=True):

        dp = 0
        a = self.canvas.create_line(node.x0 + node.dx // 2, node.y0, node.x0 + node.dx // 2, node.y, width=1)
        b = self.canvas.create_line(node.x0, node.y0 + node.dy // 2, node.x, node.y0 + node.dy // 2, width=2)

        if show_mass:
            if node.mass == 0:
                self._tree_line_objs += [a, b]
            else:
                n = node
                while n.parent:
                    dp += 1
                    n = n.parent
                w = 40 - (dp * 8)
                c = self.canvas.create_oval(node.center_of_mass[0] - w,
                                            node.center_of_mass[1] - w,
                                            node.center_of_mass[0] + w,
                                            node.center_of_mass[1] + w,
                                            width=1)
                self._tree_line_objs += [a, b, c]
        else:
            self._tree_line_objs += [a, b]

        self.canvas.update()

    def subdivide(self, node, depth):

        if self.debug:
            self._show_debug_info(node)
            time.sleep(0.05)

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
        node._find_com()

        if self.debug:
            self._show_debug_info(node)
            time.sleep(0.05)


def callback(event):
    canvas.delete('all')

    tree = BHTree(canvas, bodies, (10**12,10**15))

    def run(event):
        for _ in range(250000):
            tree.examine_bodies()

    canvas.bind('<1>', run)

def generate_bodies(no_bodies : int, proportion : float,mass_range = (10**12,10**15),star=True):

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

    max_mass,min_mass = mass_range
    mrange = [10 ** 12, 10 ** 15]
    for aa, bb in random.sample(res, int(no_bodies*proportion)):
        mass = random.randint(*mrange)
        max_mass = mass if mass > max_mass else max_mass
        min_mass = mass if mass < min_mass else min_mass
        bodies.append(NBody(mass, random.randint(1, 3), aa, bb))
    for aa, bb in random.sample(list(product(range(500), range(0, 360))), int(no_bodies*(1-proportion))):
        mass = random.randint(*mrange)
        aa, bb = (int(aa * cos(bb) + 501), int(aa * sin(bb) + 501))
        max_mass = mass if mass > max_mass else max_mass
        min_mass = mass if mass < min_mass else min_mass
        bodies.append(NBody(mass, random.randint(1, 3), aa, bb))

    if star:
        bodies.append(NBody(10**25,20,508,508,True))

    return bodies

if __name__ == '__main__':


    root = Tk()
    root.geometry("1300x1000")
    root.resizable(True,False)
    main = Frame(root,height=1000)
    canvas = Canvas(master=main, width=1000, height=1000)

    canvas.bind('<1>', callback)

    canvas.grid(column=0,row=0,rowspan=2,sticky=N)

    contframe = Frame(master=main)
    info = Canvas(master=contframe,width=300,height=1000,)

    bodies = generate_bodies(40,0.2)

    hbar = Scrollbar(contframe, orient=VERTICAL)

    hbar.config(command=info.yview)
    info.config(yscrollcommand=hbar.set)

    info.pack(side=LEFT)
    hbar.pack(side=RIGHT,fill=Y)
    info.create_text(150,10,text="Tracked Bodies",anchor=CENTER)
    infoboxes = {}
    tracked = random.sample(bodies,10)
    tracked = sorted(tracked,key=lambda body : body._id)
    info.create_line(0, 0, 0, 1000, width=2)
    for ycord,body in enumerate(tracked,start=1):
        frm = info.create_text(150,ycord*75,text=f"Body #{body._id}\n"
                                                 f"Velocity : {body.velocity[0]:.2e},{body.velocity[1]:.2e}\n"
                                                 f"Mass : {body.mass:.2e}\n",anchor=CENTER)
        infoboxes[body._id] = frm
    info.config(scrollregion=(0,0,0,len(tracked)*100+110))

    ## info frame setup

    contframe.grid(row=0,column=1)
    main.pack()

    BHTree.info_canvas = info
    BHTree.tracked_bodies = infoboxes


    mainloop()
