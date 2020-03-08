from tkinter import *
from tkinter.ttk import *
import queue
import threading
from time import sleep,time
from math import log1p
from random import randrange,randint
from collections import deque
class box:
    ident = (i for i in range(100000))
    def __init__(self,xy0,xy1,canvas,canvas_size,box_container):
        self.canvas_size = canvas_size
        self._box_container = box_container
        self._id = next(box.ident)
        self.canvas = canvas
        self.width = abs(xy0[0]-xy1[0])
        self.height = abs(xy0[1]-xy1[1])
        self.boxid = canvas.create_rectangle(xy0[0], xy0[1], xy1[0], xy1[1])
        self.cords = canvas.coords(self.boxid)
        self.text = None
        self.last = [xy0[0], xy0[1]]
        self.momentum = [randrange(-5, 5), randrange(-5, 5)]
        self.logs = []
        self.iteration = 0
        self._box_info_debug()
        self._debug_flag = False

    def _debug(self,x,y):
        if not self._debug_flag:
            return
        if not self.iteration%100:
            for log in self.logs:
                print(log)
        else:
            if x <= 0 or y <= 0:
                self.logs.append(f' box #{self._id} (x : {x}, y: {y})')
                self.logs.append(self.momentum)
            if x > self.canvas_size[0] or y > self.canvas_size[1] - self.height:
                self.logs.append(f' box #{self._id} (x : {x}, y: {y})')
                self.logs.append(self.momentum)

    def _box_info_debug(self):
        canvas = self.canvas
        a,b,c,d = canvas.coords(self.boxid)
        xy0 = [0,0]
        xy1 = [0,0]
        xy0[0],xy0[1],xy1[0],xy1[1] = [int(a),int(b),int(c),int(d)]
        if self.text is None:
            nw = canvas.create_text(xy0[0] ,
                          xy0[1],
                          text=f'({xy0[0]},{xy0[1]})',font = 'times 8 bold',anchor='nw')
            sw = canvas.create_text(xy0[0] ,
                          xy1[1],
                          text=f'({xy0[0]},{xy1[1]})',font = 'times 8 bold',anchor='sw')
            se = canvas.create_text(xy1[0] ,
                          xy1[1],
                          text=f'({xy1[0]},{xy1[1]})',font = 'times 8 bold',anchor='se')
            ne = canvas.create_text(xy1[0] ,
                          xy0[1],
                          text=f'({xy1[0]},{xy0[1]})',font = 'times 8 bold',anchor='ne')

            id = canvas.create_text(xy0[0] +self.width/2,
                          xy0[1] + self.height/2,
                          text=self._id,font = 'times 15 bold',anchor='center')

            self.text = [nw,ne,sw,se,id]
        else:
            look = [[0,0],[1,0],[0,1],[1,1],[2,2]]
            txt = [f'({xy0[0]},{xy0[1]})',
                   f'({xy1[0]},{xy0[1]})',
                   f'({xy0[0]},{xy1[1]})',
                   f'({xy1[0]},{xy1[1]})',
                   self._id

            ]
            xy = [xy0,xy1,[xy0[0] +self.width/2,
                          xy0[1] + self.height/2]]
            i = 0
            for lk,tx in zip(look,self.text):
                canvas.itemconfig(tx,text=txt[i])
                canvas.coords(tx,xy[lk[0]][0],xy[lk[1]][1])
                i +=1
    def update(self):

        self.iteration +=1

        x, y, w, h = self.canvas.coords(self.boxid)


        self._debug(x,y)
        if self.momentum[0] < 0:
            x += self.momentum[0]
        elif self.momentum[0] > 0:
            x += self.momentum[0]
        else:

            x += self.momentum[0]
        if self.momentum[1] < 0:
            y += self.momentum[1]
        elif self.momentum[1] > 0:
            y += self.momentum[1]
        else:

            y += self.momentum[1]

        if x <= 0:
            self.canvas.itemconfigure(self.boxid,fill='green')
            x = 0
            self.momentum[0] = -self.momentum[0]
        elif x >= self.canvas_size[0]-self.width:
            self.canvas.itemconfigure(self.boxid, fill='red')
            x = self.canvas_size[0]-self.width
            self.momentum[0] = -self.momentum[0]


        if y <= 0:
            self.canvas.itemconfigure(self.boxid,fill='blue')
            y = 0
            self.momentum[1] = -self.momentum[1]
        elif y >= self.canvas_size[1]-self.height+1:
            self.canvas.itemconfigure(self.boxid, fill='yellow')
            y = self.canvas_size[1]-self.height
            self.momentum[1] = -self.momentum[1]



        self.last = [x,y]
        self.canvas.coords(self.boxid, x, y, x+self.width, y+self.height)
        self._box_info_debug()
        self.momentum = [m/abs(m)*2*log1p(abs(m)) if abs(m*0.99) > 2 else randint(-1,2) for m in self.momentum ]
        print(self.momentum)
class Test:

    def __init__(self,root):
        self.root = root

    def _check_collisions(self):
        for debug_val in [5]:
            t0 = time()

            def collision_fun():
                col_box = self.boxes[comp_id]
                for i in [0,1]:
                    if abs(col_box.momentum[i]+box.momentum[i]) == abs(col_box.momentum[i])+abs(box.momentum[i]):

                        by_momentum = sorted([col_box,box],key= lambda b: b.momentum[i])
                        by_momentum[0].momentum[i] *= 1
                        by_momentum[1].momentum[i] *= 1.5
                    else:

                        col_box.momentum[i], box.momentum[i] = box.momentum[i], \
                                                           col_box.momentum[i]




                if debug_val > 0:
                    self.canvas_main.itemconfigure(box.boxid, fill='red')
                    self.canvas_main.itemconfigure(col_box.boxid, fill='yellow')
                    collisions.append(comp_id)
                if debug_val > 0:
                    box._box_info_debug()
                if debug_val  == 0 or debug_val > 2:
                    box_ids.remove(self.boxes[comp_id].boxid)




            box_ids = list(self.boxes.keys())
            for id,box in self.boxes.items():
                if len(box_ids) == 0:
                    break
                if id != box_ids[0]:
                    continue
                collisions = []
                x, y, x2, y2 = self.canvas_main.coords(box.boxid)

                for comp_id in box_ids:

                    if comp_id == id:
                        continue
                    xc, yc, x2c, y2c = self.canvas_main.coords(self.boxes[comp_id].boxid)
                    if xc <= x <= x2c:
                        if yc <= y <= y2c:
                            collision_fun()
                            continue
                        elif yc <=y2 <= y2c:
                            x, y = x2c, yc
                            collision_fun()
                            continue
                    elif xc <= x2 <= x2c:
                        if yc <= y <= y2c:
                            collision_fun()
                            continue
                        elif yc <= y2 <= y2c:
                            collision_fun()
                            continue
                    if x <= xc <= x2:
                        if y <= yc <= y2:
                            collision_fun()
                            continue
                        elif y <=y2c <= y2:
                            collision_fun()
                            continue
                    elif x <= x2c <= x2:
                        if y <= yc <= y2:
                            collision_fun()
                            continue
                        elif y <=y2c <= y2:
                            collision_fun()
                            continue

                if debug_val == 0 or debug_val >= 1:
                    box_ids.remove(box.boxid)
            self.canvas_main.coords(box,x,y,x2,y2)
            print(time()-t0)
        print('\n')
        self.canvas_main.update()



    def callback_fun(self,event):
        n = 0

        while True:

            for box in self.boxes.values():
                box.update()
            self._check_collisions()
            sleep(.005)
            self.canvas_main.update()

    def start(self,boxes,canvas_size=[600,600]):
        self.main = Frame(root)
        self.boxes = {}
        self.canvas_main = Canvas(master=self.main,width=canvas_size[0],height=canvas_size[1])
        self._create_box_positions(boxes)
        for i in range(len(self._positions)):
            a,b,c,d = self._positions[i]
            bx = box([a,b],[c,d],self.canvas_main,canvas_size,self.boxes)
            self.boxes[bx.boxid] = bx
        self.canvas_main.bind("<1>", self.callback_fun)
        self.canvas_main.pack()
        self.main.pack()
        mainloop()

    def _create_box_positions(self,boxes):
        self._positions = []
        x,y = 0,0

        for i in range(boxes):
            size = randrange(15, 75)
            x0,y0 = randrange(0,601),randrange(0,601)
            self._positions.append([x0,y0,x0+size,y0+size])

if __name__ == '__main__':
    root = Tk()
    import random
    c = Test(root)
    c.start(5)