from random import randrange,randint,choice
from statistics import mean
from typing import List,TypeVar
from math import sqrt,ceil

Node = TypeVar('Node')

class LocationNode:

    def __init__(self,position : List[int],max_height : int =None):
        self.max_height = max_height
        self.xcord,self.ycord = position
        self.N,self.S,self.E,self.W = None,None,None,None
        self.NW,self.NE,self.SW,self.SE = None,None,None,None
        self.height = None

    def setup_nodes(self,adjacent_pieces : List[Node]):

        for node in [node for node in adjacent_pieces if node is not None]:
            if node.xcord < self.xcord:
                if node.ycord < self.ycord:
                    self.SW = node
                elif node.ycord > self.ycord:
                    self.NW = node
                else:
                    self.W = node
            elif node.xcord > self.xcord:
                if node.ycord < self.ycord:
                    self.SE = node
                elif node.ycord > self.ycord:
                    self.NE = node
                else:
                    self.E = node
            else:
                if node.ycord < self.ycord:
                    self.S = node
                else:
                    self.N = node
        self.adjacent_nodes = tuple([pos for pos in
                                     [self.N, self.S, self.E, self.W, self.NW, self.NE, self.SW, self.SE]
                                     if pos is not None])

    def generate_adj_heights(self):
        heights = []
        for node in self.adjacent_nodes:
            if node.height:
                heights.append(node.height)
        else:
            if heights:
                avgheight = mean(heights)
                print(avgheight)
            else:
                for node in self.adjacent_nodes:
                    node.height = randrange(self.max_height-2,self.max_height)

        return self.adjacent_nodes


class MapContainer:


    def __init__(self,xy : List[int],peak_height : int,ismaxheight = True):
        self.peak_height = peak_height
        self.nodecontainer = [[None for y in range(xy[1])] for x in range(xy[0])]

        if ismaxheight:
            max = peak_height
        else:
            max = None
        for x in range(xy[1]):
            for y in range(xy[0]):
                print(x,y)
                self.nodecontainer[y][x] = LocationNode([y,xy[1]-x-1],max)
        for x in range(xy[1]):
            for y in range(xy[0]):
                working_node = self.nodecontainer[y][x]
                adj_nodes = []
                for zx,zy in [(1,1),(1,0),(0,1),(-1,-1),(-1,0),(0,-1),(1,-1),(-1,1)]:
                    try:
                        adj_node = self.nodecontainer[y+zx][x+zy]
                    except IndexError:
                        continue
                    else:
                        adj_nodes.append(adj_node)
                working_node.setup_nodes(adj_nodes)
        self.nodecontainer = list(zip(*self.nodecontainer))

    def set_initial_nodes(self,peaks):
        self.starting_nodes = []
        for _ in range(peaks):
            node = choice(choice(self.nodecontainer))
            node.height = self.peak_height
            self.starting_nodes.append(node)

    def _generate_heightmap(self):
        unexamined_nodes = [node for nodes in self.nodecontainer for node in nodes if node not in self.starting_nodes]
        open_nodes = self.starting_nodes
        examined_nodes = []
        while open_nodes:
            current_node = open_nodes[0]
            _ = [[open_nodes.append(node),unexamined_nodes.remove(node)]
                 for node in current_node.generate_adj_heights
                 if node not in examined_nodes and node not in open_nodes]
            examined_nodes.append(open_nodes.pop(0))


    def __call__(self, peaks : int = 5):
        self.set_initial_nodes(peaks+1)

    def __str__(self):
        display = "\n".join(["".join([f'({pc.xcord},{pc.ycord},{pc.height})' for pc in row]) for row in self.nodecontainer])
        return display

if __name__ == '__main__':
    test = MapContainer([10,10],5)
    test()
    print(test)


