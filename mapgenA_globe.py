from math import sin,cos,log,ceil
from typing import List

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class WrapList:

    def __init__(self,iterator):
        self.iterator = iterator
        self.perm = tuple(iterator)
        self.length = len(iterator)

    def __len__(self):
        return len(self.iterator)
    def __iter__(self):
        return next(self)

    def __next__(self):
        for item in self.iterator:
            yield item

    def __getitem__(self,index):
        try:
            return self.iterator[index]
        except IndexError:
            if index >= self.length:
                return self.perm[0]
            else:
                return self.perm[-1]

    def __setitem__(self, index, value):
        self.iterator[index] = value

from typing import List
from random import randint,randrange,choice
from statistics import mean
from collections import defaultdict
class LocationNode:
    reverse_directions = {'N':'S','S':'N','E':'W','W':'E'}

    def __init__(self,cordinates : List[int],radius : int,max_height : int = None):

        self.weights = (0.5,0.3,0.2)
        self.max_height = max_height

        self.polar,self.radial = cordinates
        self.radius = abs(radius*sin(self.polar))

        self.xcord,self.ycord,self.zcord = self.radius*cos(self.radial),self.radius*sin(self.radial),radius*cos(self.polar)
        self.adj_coords = {}
        self.height = None

    def setup_nodes(self,adjacent_pieces : List):
        nodes = [node for node in adjacent_pieces if node is not None]
        maxrad = nodes[-1].radial
        for index,node in enumerate(nodes):
            if node.polar == self.polar:
                if node.radial == maxrad:
                    self.adj_coords['W'] = node
                else:
                    self.adj_coords['E'] = node
            elif node.radial == self.radial:
                if node.polar > self.polar:
                    self.adj_coords['N'] = node
                else:
                    self.adj_coords['S'] = node
            else:
                if node.polar > self.polar:
                    if node.radial > self.radial:
                        if node.radial == maxrad:
                            self.adj_coords['NW'] = node
                        else:
                            self.adj_coords['NE'] = node
                    else:
                        self.adj_coords['NW'] = node
                elif node.polar < self.polar:
                    if node.radial > self.radial:
                        if node.radial == maxrad:
                            self.adj_coords['SW'] = node
                        else:
                            self.adj_coords['SE'] = node
                    else:
                        self.adj_coords['SW'] = node





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
                                weighted_average[0] += abs(directional_averages[self.reverse_directions[cardinal_dir]]
                                                                                - directional_averages[cardinal_dir])
                                weighted_average[0] = weighted_average[0]/2
                            else:
                                weighted_average[0] += abs(directional_averages[self.reverse_directions[cardinal_dir]]
                                                                                - directional_averages[cardinal_dir])
                    # factoring in directional slope, matters less when local area is mostly uniform
                    weighted_average[0] = weighted_average[0]*(min(heightval)/max(heightval))
                    # local average,matters less when the local area varies heavily
                    weighted_average[1] = mean(heightval)*(min(heightval)/max(heightval))
                    weighted_average[2] = randint(int(min(heightval)-3),int(max(heightval)+5)) # random factor

                    new_height = sum([value*weight for value,weight in zip(weighted_average,self.weights)])

                    if new_height - int(new_height) < 0.5:
                        node.height = int(new_height)
                    else:
                        node.height = ceil(new_height)
                    if node.height > self.max_height: node.height = self.max_height
        else:
            for node,node in self.adj_coords.items():
                    node.height = randrange(self.height-2,self.max_height)

        return [value for value in self.adj_coords.values()]



class MapContainer:


    def __init__(self,radius,levels,peak_height : int,ismaxheight = True):

        self.peak_height = peak_height
        multipliers = [1,2,4,6,9,10,12,18,20,24,36,72,144,90,120,180][::-1]
        if len(multipliers) < int(ceil(4*log(levels, 2))):
            nodes_per_level = multipliers[-1]
        else:
            nodes_per_level = multipliers[int(ceil(4*log(levels, 2)))]
        nodes_per_level = 18
        polar = [180-90/(levels/2)*x for x in range(levels+1)]
        radial = [x for x in range(0,360,nodes_per_level)]

        self.nodecontainer = [WrapList([None for y in range(len(radial))]) for x in range(len(polar) )]
        if ismaxheight:
            max = peak_height
        else:
            max = None
        for zindex,pol in enumerate(polar):
            for yindex,rad in enumerate(radial):
                self.nodecontainer[zindex][yindex] = LocationNode([pol,rad],radius)
        for zindex, pol in enumerate(polar):
            for yindex, rad in enumerate(radial):
                working_node = self.nodecontainer[zindex][yindex]
                adj_nodes = []
                for zx,zy in [(1,1),(1,0),(0,1),(-1,-1),(-1,0),(0,-1),(1,-1),(-1,1)]:
                    try:
                        if zindex == 0 and zx == -1:
                            continue
                        adj_node = self.nodecontainer[zindex+zx][yindex+zy]
                    except IndexError:
                        print(zx,zy)
                        continue
                    else:
                        adj_nodes.append(adj_node)
                working_node.setup_nodes(adj_nodes)
        print('none')
        t = [nodes for nodes in self.nodecontainer]
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        for nodes in t:
            x,y,z = [],[],[]
            for node in nodes:
                print(node.radius,node.xcord)
                x.append(node.xcord);y.append(node.ycord);z.append(node.zcord)
            ax.scatter(xs=x,ys=y,zs=z)
        plt.show()

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
        nodes = [node for nodes in self.nodecontainer]
    def __call__(self, peaks : int = 5,custom_weights : List[int] = None):
        self.set_initial_nodes(peaks+1)
        self._generate_heightmap(custom_weights)
    def __str__(self):
        display = "\n".join(["".join([str(pc) for pc in row]) for row in self.nodecontainer])
        return display

if __name__ == '__main__':
    test = MapContainer(radius=20,levels=10,peak_height=5)
    test()
    print(test)


