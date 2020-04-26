import random
from tkinter import *
class Cell:


    def __init__(self,position,living,tile_coords):
        self.pos = position
        self.last_state = living
        self.alive = living
        self.adjacent_tiles = tile_coords
        self.adjacent_nodes = {k: None for k in tile_coords.keys()}
        self.id = canvas.create_text(position[0]*6+5,position[1]*6+5,text=str(self))

    def check_state(self):
        alive = 0
        dead = 0
        self.last_state =self.alive
        for node in self.adjacent_nodes.values():
            if node.last_state:
                alive+=1
            else:
                dead+=1
        if self.alive:
            if alive < 2 or alive > 3:
                self.alive = False
        else:
            if alive ==3:
                self.alive = True
        canvas.itemconfigure(self.id,text=str(self))
    def assign_nodes(self, tile_list):
        for direction, coords in self.adjacent_tiles.items():
            if coords == None:
                del self.adjacent_nodes[direction]
                continue
            self.adjacent_nodes[direction] = tile_list[coords[1]][coords[0]]

    def __str__(self):
        if self.alive:
            return "o"
        else:
            return " "

class Conway:
    def __init__(self):
        self.data = None
        self.raw = []
    def setup(self):
        data = []

        rnge = 50
        next = lambda x, y: {'N': [x, y - 1], 'S': [x, y + 1], 'E': [x + 1, y], 'W': [x - 1, y],
                             'NW' : [x-1, y - 1],'NE':[x+1, y - 1],'SW':[x-1, y +1],'SE':[x+1, y + 1]}
        for y in range(rnge):
            data.append([])
            for x in range(rnge):
                next_coords = next(x, y)

                if x == 0:
                    next_coords['W'] = None
                    next_coords['NW'] = None
                    next_coords['SW'] = None
                if y == 0:
                    next_coords['N'] = None
                    next_coords['NW'] = None
                    next_coords['NE'] = None
                if y == rnge - 1:
                    next_coords['S'] = None
                    next_coords['SW'] = None
                    next_coords['SE'] = None
                if x == rnge - 1:
                    next_coords['E'] = None
                    next_coords['NE'] = None
                    next_coords['SE'] = None
                alive = True if random.randint(-10,1) > 0 else False
                if x > 50 or y > 50:
                    alive=False
                cell = Cell((x,y),alive,next_coords)
                data[y].append(cell)
                self.raw.append(cell)
        for _ in data:
            for node in _:
                node.assign_nodes(data)
        self.data = data

    def run(self):
        for _ in range(1):
            for c in self.raw:
                c.check_state()


def callback(event):
    canvas.bind("<Button-1>", reset)
    while True:
        for c in dT.raw:
            c.check_state()
        canvas.update()
        time.sleep(0.01)

def reset(event):
    canvas.bind("<Button-1>", callback)
    for c in dT.raw:
        alive = True if random.randint(-15, 1) > 0 else False
        c.alive = alive


if __name__ == '__main__':
    import time
    main = Tk()
    win =Frame(master=main)
    canvas = Canvas(master=win,width=700,height=700)
    canvas.bind("<Button-1>", callback)
    canvas.pack()
    win.pack()
    dT = Conway()
    dT.setup()


    mainloop()