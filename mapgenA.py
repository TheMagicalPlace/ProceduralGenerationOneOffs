from random import randrange,randint,choice
from statistics import mean
from typing import List,TypeVar,Container
from math import sqrt,ceil,log10
from collections import defaultdict
Node = TypeVar('Node')
from colorama import Back,Style,Fore
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
class LocationNode:
    reverse_directions = {'N':'S','S':'N','E':'W','W':'E'}

    def __init__(self,position : List[int],max_height : int =None,weights=(0.5,0.45,0.05)):
        self.weights = weights
        self.max_height = max_height
        self.xcord,self.ycord = position
        self.adj_coords = {}
        self.height = None

    def setup_nodes(self,adjacent_pieces : List[Node]):

        for node in [node for node in adjacent_pieces if node is not None]:
            if node.xcord < self.xcord:
                if node.ycord < self.ycord:
                    self.adj_coords['SW'] = node
                elif node.ycord > self.ycord:
                    self.adj_coords['NW'] = node
                else:
                    self.adj_coords['W'] = node
            elif node.xcord > self.xcord:
                if node.ycord < self.ycord:
                    self.adj_coords['SE'] = node
                elif node.ycord > self.ycord:
                    self.adj_coords['NE'] = node
                else:
                    self.adj_coords['E'] = node
            else:
                if node.ycord < self.ycord:
                    self.adj_coords['S'] =  node
                else:
                    self.adj_coords['N'] =  node

    def generate_adj_heights(self,custom_weights):
        if custom_weights: self.weights = custom_weights
        has_height = {}
        heightval = []
        no_height = {}

        for rel_pos,node in self.adj_coords.items():
            if node.height:
                has_height[rel_pos] = node
                heightval.append(node.height)
            else:
                no_height[rel_pos] = node

        if has_height:
            wavg,eavg,savg,navg = [],[],[],[]
            directional_averages = [wavg,eavg,savg,navg]
            for rel,node in has_height.items():
                if 'N' in rel:
                    navg.append(node.height)
                if 'S' in rel:
                    savg.append(node.height)
                if 'E' in rel:
                    eavg.append(node.height)
                if 'W' in rel:
                    wavg.append(node.height)
            else:

                directional_averages= {direction:(0 if not avg else mean(avg)) for direction,avg
                                        in zip(['N','S','E','W'],directional_averages)}
                for position,node in no_height.items():

                    # weighted average  = [weight from directional slope if present,
                    #                      weight from local avg, random for variability]

                    weighted_average = [0,0,0]
                    for cardinal_dir in ['N','S','E','W']:
                        if cardinal_dir in position:
                            if weighted_average[0]:
                                # if there is a gradient in one direction this will favor it
                                weighted_average[0] += abs(directional_averages[cardinal_dir])
                                weighted_average[0] = weighted_average[0]/4
                            else:
                                weighted_average[0] += abs(directional_averages[cardinal_dir])
                    # factoring in directional slope, matters in lower areas
                    weighted_average[0] = weighted_average[0]
                    # local average,matters less when the local area varies heavily
                    weighted_average[1] = mean(heightval)
                    weighted_average[2] = randint(int(min(heightval)-5),int(max(heightval)+5)) # random factor

                    new_height = sum([value*weight for value,weight in zip(weighted_average,self.weights)])

                    if new_height - int(new_height) < 0.5:
                        node.height = new_height
                    else:
                        node.height = new_height
                    if node.height > self.max_height: node.height = self.max_height
        else:
            for node,node in self.adj_coords.items():
                    node.height = randrange(self.height-2,self.max_height)

        return [value for value in self.adj_coords.values()]

    def __str__(self,comparison_height = None):
        if self.max_height:
            height_ratio = self.height/self.max_height
        elif comparison_height:
            height_ratio = self.height/comparison_height
        else:
            height_ratio = 1
        return "".join((Fore.WHITE if height_ratio > 0.9 else
                        Fore.RED if height_ratio > 0.75 else
                        Fore.LIGHTRED_EX if height_ratio > 0.65 else
                        Fore.YELLOW if height_ratio > 0.5 else
                        Fore.LIGHTYELLOW_EX if height_ratio > 0.4
                        else Fore.GREEN if height_ratio > 0.1
                        else Fore.BLUE)
                        +'('  +f'{self.height}'.ljust(2).ljust(1)+')' +Fore.RESET)

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

    def _generate_heightmap(self,custom_weights):
        unexamined_nodes = [node for nodes in self.nodecontainer for node in nodes if node not in self.starting_nodes]
        open_nodes = self.starting_nodes
        examined_nodes = []
        while open_nodes:
            current_node = open_nodes[0]
            _ = [[open_nodes.append(node),unexamined_nodes.remove(node)]
                 for node in current_node.generate_adj_heights(custom_weights)
                 if node not in examined_nodes and node not in open_nodes]
            examined_nodes.append(open_nodes.pop(0))

    def scatter(self):
        nodes = [nodes for nodes in self.nodecontainer]
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        x = [] ; y = [] ; z = []
        for node in nodes:
            for node in node:

                x.append(node.xcord)
                y.append(node.ycord)
                z.append(node.height)
        X,Y = x,y
        Z= z
        ax.scatter(X,Y,Z,c=Z,cmap='viridis')
        ax.plot_trisurf(X,Y,Z,cmap='viridis')
        #plt.show()

    def __call__(self, peaks : int = 5,custom_weights : List[float] = None):
        self.set_initial_nodes(peaks)
        self._generate_heightmap(custom_weights)
        self.scatter()
    def __str__(self):
        display = "\n".join(["".join([str(pc) for pc in row]) for row in self.nodecontainer])
        return display

if __name__ == '__main__':
    test = MapContainer([15,15],5)
    test(custom_weights=[1,0.5,0.2])
    print(test)


