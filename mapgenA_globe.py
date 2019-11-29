from math import sin,cos,log,ceil,radians,sqrt
from typing import List

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

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
        self.circradius = radius
        self.polar,self.radial = [radians(c) for c in cordinates]
        self.radius = radius
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

        # modifying default weights for height gen
        if custom_weights: self.weights = custom_weights


        has_height = {} # nodes that already have a height
        heightval = [] # corresponding heights
        no_height = {} # nodes without a height

        for rel_pos,node in self.adj_coords.items():
            if node.height:
                has_height[rel_pos] = node
                heightval.append(node.height)
            else:
                no_height[rel_pos] = node

        # if none of the adj nodes have heights this is skipped, otherwise determines the heigh position relative to
        # the node
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
            # this should always occur
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
                                weighted_average[0] += directional_averages[cardinal_dir]
                                weighted_average[0] = weighted_average[0]/2
                            else:
                                weighted_average[0] += directional_averages[cardinal_dir]

                    # factoring in directional slope, matters less when local area is mostly uniform
                    weighted_average[0] = weighted_average[0]

                    # local average,matters less when the local area varies heavily
                    weighted_average[1] = mean(heightval)

                    weighted_average[2] = randint(int(min(heightval)-self.circradius//5),int(max(heightval)+self.circradius//5)) # random factor


                    new_height = sum([value*weight for value,weight in zip(weighted_average,self.weights)])
                    node.height = new_height
                    node.radius = node.radius+new_height
                    if self.max_height and node.height > self.max_height:
                        node.height = self.max_height
                        node.radius +=self.max_height
        # if there is no height for the node, it is randomly generated
        else:
            for _,node in self.adj_coords.items():
                if self.max_height:
                    node.height = randint(self.height-2,self.max_height)
                    node.radius = node.radius+node.height
                else:
                    node.height = randint(int(self.height-self.height//2),int(self.height-self.height//2))
                    node.radius+=node.height

        return [value for value in self.adj_coords.values()]



class MapContainer:


    def __init__(self,radius,z_levels,nodes_per_level : int,max_height=None):

        self.max_height = max_height



        self.polar = [x-90 for x in np.linspace(90,-90,z_levels,endpoint=True)]
        self.radial = [x for x in np.linspace(0,360-360/nodes_per_level,nodes_per_level,endpoint=True)]
        self.radius = radius
        self.nodecontainer = [WrapList([None for y in range(len(self.radial))]) for x in range(len(self.polar) )]
        self._setup_map()


    def _setup_map(self):
        if self.max_height:
            max = self.max_height
        else:
            max = None
        for zindex,pol in enumerate(self.polar):
            for yindex,rad in enumerate(self.radial):
                self.nodecontainer[zindex][yindex] = LocationNode([pol,rad],self.radius)
        for zindex, pol in enumerate(self.polar):
            for yindex, rad in enumerate(self.radial):
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

    def set_initial_nodes(self,peaks,initialheight=None):
        if initialheight and not self.max_height:
            height = initialheight
        else:
            height = self.max_height
        self.starting_nodes = []

        for _ in range(peaks):
            node = choice(choice(self.nodecontainer))
            node.height = height
            node.radius += height
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

    def _convert_to_cartesian(self,radius,radial,polar):

        x = radius*np.sin(polar)*np.cos(radial)
        y = radius*np.sin(polar)*np.sin(radial)
        z = radius*np.cos(polar)
        print(z)

        return x,y,z

    def scatter(self):

        t = [nodes for nodes in self.nodecontainer]
        fig = plt.figure()
        fig2 = plt.figure()
        ax3d = fig2.add_subplot(111,projection='3d')
        gs = fig.add_gridspec(2, 2)
        ax2 = fig.add_subplot(gs[1,0])
        ax = fig.add_subplot(gs[0,:])
        ax3 = fig.add_subplot(gs[1,1])

        radius,radial,polar = [],[],[]

        for nodes in t:
            r,rad,pol = [],[],[]
            for node in nodes:
                r.append(node.radius);rad.append(node.radial);pol.append(node.polar)
            r.append(r[0]);rad.append(rad[0]);pol.append(pol[0])
            radial.append(rad),polar.append(pol),radius.append(r)
        radial = np.array(radial)
        polar = np.array(polar)
        X,Y,Z = self._convert_to_cartesian(radius,radial,polar)
        print(X.shape,Y.shape,Z.shape)
        ax3d.plot_wireframe(X,Y,Z,cmap='viridis')
        #ax3d.plot(X,Y,Z)
        #ax.scatter(xs=z, ys=x, zs=y, c=z, cmap='viridis')
        #X,_ = np.meshgrid(xx,xx)
        #Y,_ = np.meshgrid(yy,yy)
        #_,Z = np.meshgrid(zz,zz)


        #ax3d.plot_surface(Y,Z,X)
        plt.show()



    def __call__(self, peaks : int = 5,initial_height= 5,custom_weights : List[int] = None):
        self.set_initial_nodes(peaks+1,initialheight=initial_height)
        self._generate_heightmap(custom_weights)
    def __str__(self):
        #display = "\n".join(["".join([str(pc) for pc in row]) for row in self.nodecontainer])
        self.scatter()
        return ""


if __name__ == '__main__':
    test = MapContainer(radius=500,z_levels=60,nodes_per_level=60)
    test(peaks=100,initial_height=100,custom_weights=[.6,.3,.2])
    print(test)
    print('yeet')

