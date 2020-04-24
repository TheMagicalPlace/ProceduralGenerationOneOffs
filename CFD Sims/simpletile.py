from math import log10
import random
from collections import defaultdict
from copy import copy,deepcopy
from colorama import Fore,Style
class Tile:
    tile_types = ['coast','forest','sea']
    weights = {'coast' : 0.05,
               'forest' : 0.33,
               'sea' : 0.33}

    rules = {'coast' : {'required_by':[],'required':['sea','forest'],'disallowed':[]},
             'forest' : {'required_by':['coast'],'required':[],'disallowed':['sea']},
             'sea': {'required_by': ['coast'], 'required': [], 'disallowed': ['forest']}
             }

    def __init__(self,pos,tile_coords):
        self.pos = pos
        self.adjacent_tiles = tile_coords
        self.adjacent_nodes = {k:None for k in tile_coords.keys()}
        self.tile_type = None
        self.valid_tiles = ['coast','forest','sea']
        self.required_adjacent = None

    def get_entropy(self):

        count = defaultdict(int)
        if len(self.valid_tiles) == 0:
            print('oops')
            self.valid_tiles = ['sea']
            return 0
        weights = [Tile.weights[biome] for biome in self.valid_tiles]
        entropy = log10(sum(weights))-(sum([w*log10(w) for w in weights]))/sum(weights)
        return entropy

    def update_adjacent(self):
        if self.required_adjacent:
            unset = None
            for node in self.adjacent_nodes.values():
                if node.tile_type is None:
                    if unset is None:
                        for required in self.required_adjacent:
                            if required in node.valid_tiles:
                                unset = node
                    else:
                        break

                elif node.tile_type in self.required_adjacent:
                    self.required_adjacent.remove(node.tile_type)
            else:
                if self.required_adjacent and unset:
                    unset.valid_tiles = [self.required_adjacent[0]]

    def check_valid_types(self):
        if self.valid_tiles is None:
            self.update_adjacent()
            return
        if len(self.valid_tiles) == 1:
            return
        adj_biomes = []
        nodes = []
        needed_tiles = []
        disallowed = []
        for direction,node in self.adjacent_nodes.items():
            if node.tile_type is not None:
                adj_biomes.append(node.tile_type)
                nodes.append(node)
        if not adj_biomes:
            return
        for tile in self.valid_tiles[::]:
            rule = deepcopy(Tile.rules[tile])
            for i,biome in enumerate(adj_biomes):
                if biome in rule['disallowed'] or(nodes[i].valid_tiles is not None and len(nodes[i].valid_tiles) == 1 and nodes[i].valid_tiles[0] in rule['disallowed']):
                    self.valid_tiles.remove(tile)
                    break


            else:
                if len(adj_biomes) == 4:
                    for req_biome in rule['required']:
                        if req_biome not in adj_biomes and tile in self.valid_tiles:
                            self.valid_tiles.remove(tile)

    def set_tile_type(self):

            lottery = []
            count = defaultdict(int)
            for biome in self.valid_tiles:
                count[biome] += 1
            weights = {k:v*count[k] for k,v in Tile.weights.items() if k in self.valid_tiles}
            weight_scale = min([v for v in weights.values()])
            weights = {k: round(v /weight_scale) for k, v in weights.items()}
            for type,scaled_weight in weights.items():
                lottery += [type]*scaled_weight
            self.valid_tiles = None
            self.tile_type = random.choice(lottery)

            for node in self.adjacent_nodes.values():
                for biome in Tile.rules[self.tile_type]['disallowed']:
                    if node.valid_tiles and biome in node.valid_tiles:
                        node.valid_tiles.remove(biome)
            self.required_adjacent = deepcopy(Tile.rules[self.tile_type]['required'])
            if self.tile_type == 'coast':
                if len(self.required_adjacent) < 2:
                    print('')

    def assign_nodes(self,tile_list):
        for direction,coords in self.adjacent_tiles.items():
            if coords == None:
                del self.adjacent_nodes[direction]
                continue
            self.adjacent_nodes[direction] = tile_list[coords[1]][coords[0]]

    def __str__(self):
        spaceing = len("|" + "/".join([t[0] for t in Tile.tile_types]) + '|')-2
        spaceing = max(spaceing,max([len(_) for _ in Tile.tile_types]))
        if self.tile_type is not None:
            res = ""
            if self.tile_type == 'coast':
                res +=Fore.YELLOW
            if self.tile_type == 'sea':
                res +=Fore.BLUE
            if self.tile_type == 'forest':
                res +=Fore.GREEN
            return res + str(self.tile_type[0]).center(0)+Style.RESET_ALL
        else:
            return "/".join([t[0] for t in self.valid_tiles]).center(3)



class TileCollapse:

    def __init__(self):
        self.unassigned = []
        self.assigned_ignore = []
        self.assigned_examine = []
    def generate_initial_model(self):

        assigned_coords = []
        for i in range(1):
            assigned_coords.append([random.randrange(50),random.randrange(50)])
        data  = []
        rnge = 30
        next = lambda x,y : {'N':[x,y-1],'S':[x,y+1] ,'E':[x+1,y],'W':[x-1,y]}
        for y in range(rnge):
            data.append([])
            for x in range(rnge):
                next_coords = next(x,y)


                if x == 0:
                    next_coords['W'] = None
                if y == 0:
                    next_coords['N'] = None
                if y == rnge - 1:
                    next_coords['S'] = None
                if x == rnge - 1:
                    next_coords['E'] = None

                tile = Tile([x,y],next_coords)
                if [x,y] in assigned_coords:
                    tmp_node = tile
                    tmp_node.tile_type = 'sea'
                    tmp_node.valid_tiles = None
                    self.assigned_ignore.append(tmp_node)
                else:
                    self.unassigned.append(tile)
                data[y].append(tile)
        for _ in data:
            for node in _:
                node.assign_nodes(data)
        self.data = data
    def collapse(self):
        while self.unassigned:
            entropy = 10000000
            next_to_collapse = None
            to_remove = []


            for node in self.unassigned:
                        if node.tile_type is not None:
                            self.unassigned.remove(node)
                            continue
                        node.check_valid_types()
                        if node.valid_tiles is not None and node.get_entropy() < entropy:
                            entropy = node.get_entropy()
                            next_to_collapse = node
            next_to_collapse.set_tile_type()
            self.unassigned.remove(next_to_collapse)
            if next_to_collapse.required_adjacent is None:
                self.assigned_ignore.append(next_to_collapse)
            else:
                self.assigned_examine.append(next_to_collapse)
            for i,node in enumerate(self.assigned_examine):
                node.update_adjacent()
                if not node.required_adjacent:
                    to_remove.append(i)

            [self.assigned_ignore.append(self.assigned_examine[x]) for x in to_remove]
            self.assigned_examine = [examine for i,examine in enumerate(self.assigned_examine) if i not in to_remove]

if __name__ == '__main__':
    dT = TileCollapse()

    dT.generate_initial_model()
    for i in range(1):
        dT.collapse()
    [print("  ".join([str(__) for __ in _])) for _ in dT.data]
    print('\n')
    [print("  ".join([str(__) for __ in _])) for _ in dT.data]
    print('\n')