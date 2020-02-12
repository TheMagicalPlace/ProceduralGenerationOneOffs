from tkinter import *
from tkinter.ttk import *
import queue
import threading
from time import sleep

from random import randrange,randint

class box:
    ident = (i for i in range(1000))
    def __init__(self,xy0,xy1,canvas,canvas_size):
        self.canvas_size = canvas_size
        self._id = next(box.ident)
        self.canvas = canvas
        self.width = abs(xy0[0]-xy1[0])
        self.height = abs(xy0[1]-xy1[1])
        self.boxid = canvas.create_rectangle(xy0[0], xy0[1], xy1[0], xy1[1])
        self.last = [xy0[0], xy0[1]]
        self.momentum_direction = [randrange(-1,2),randrange(-1,2)]
        self.logs = []
        self.iteration = 0
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
                self.logs.append(self.momentum_direction)
            if x > self.canvas_size[0] or y > self.canvas_size[1] - self.height:
                self.logs.append(f' box #{self._id} (x : {x}, y: {y})')
                self.logs.append(self.momentum_direction)

    def update(self):

        self.iteration +=1

        x, y, w, h = self.canvas.coords(self.boxid)

        x += randint(-3,3)
        y += randint(-3,3)

        self._debug(x,y)
        if self.momentum_direction[0] < 0:
            x -= 2
        elif self.momentum_direction[0] > 0:
            x += 2
        else:
            self.momentum_direction[0] = randint(-1,1)
        if self.momentum_direction[1] < 0:
            y -= 2
        elif self.momentum_direction[1] > 0:
            y += 2
        else:
            self.momentum_direction[1] = randint(-2,2)

        if x <= 0:
            self.canvas.itemconfigure(self.boxid,fill='green')
            x = 0
            self.momentum_direction[0] = 1
        elif x >= self.canvas_size[0]-self.width:
            self.canvas.itemconfigure(self.boxid, fill='red')
            x = self.canvas_size[0]-self.width
            self.momentum_direction[0] = -1
        else:

            if x < self.last[0]:
                if self.momentum_direction[0] > 0:
                    self.momentum_direction[0] -= 1
                elif self.momentum_direction[0] <= 0:
                    self.momentum_direction[0] -= randint(0,1)
            elif x > self.last[0]:
                if self.momentum_direction[0] > 0:
                    self.momentum_direction[0] += 1
        if y <= 0:
            self.canvas.itemconfigure(self.boxid,fill='blue')
            y = 0
            self.momentum_direction[1] = 1
        elif y >= self.canvas_size[1]-self.height+1:
            self.canvas.itemconfigure(self.boxid, fill='yellow')
            y = self.canvas_size[1]-self.height
            self.momentum_direction[1] = -1
        else:
            if y < self.last[1]:
                if self.momentum_direction[1] > 0:
                    self.momentum_direction[1] -= 1
                elif self.momentum_direction[1] <= 0:
                    self.momentum_direction[1] -= randint(0,1)
            elif y > self.last[1]:
                if self.momentum_direction[1] > 0:
                    self.momentum_direction[1] += 1


        self.last = [x,y]
        self.canvas.coords(self.boxid, x, y, x+self.width, y+self.height)

class Test:

    def __init__(self,root):
        self.root = root

    def callback_fun(self,event):
        n = 0
        while True:
            for box in self.boxes:

                box.update()
            sleep(.005)
            self.canvas_main.update()

    def start(self,boxes,canvas_size=[600,600]):
        self.main = Frame(root)
        self.boxes = []
        self.canvas_main = Canvas(master=self.main,width=canvas_size[0],height=canvas_size[1])
        self._create_box_positions(boxes)
        for i in range(len(self._positions)):
            a,b,c,d = self._positions[i]
            self.boxes.append(box([a,b],[c,d],self.canvas_main,canvas_size))
        self.canvas_main.bind("<1>", self.callback_fun)
        self.canvas_main.pack()
        self.main.pack()
        mainloop()

    def _create_box_positions(self,boxes):
        self._positions = []
        x,y = 0,0

        for i in range(boxes):
            size = randrange(10, 41)
            x0,y0 = randrange(0,601),randrange(0,601)
            self._positions.append([x0,y0,x0+size,y0+size])

if __name__ == '__main__':
    root = Tk()
    c = Test(root)
    c.start(100)