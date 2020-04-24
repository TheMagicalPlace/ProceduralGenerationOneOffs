from math import log10
import random


class Tile:
    tile_types = ['coast','forest','sea']
    weights = {'coast' : 0.1,
               'forest' : 0.34,
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
        weights = [Tile.weights[biome] for biome in self.valid_tiles]
        entropy = log10(sum(weights))-(sum([w*log10(w) for w in weights]))/sum(weights)
        return entropy


    def check_valid_types(self):
        if self.valid_tiles is None:
            return
        adj_biomes = []
        nodes = []
        needed_tiles = []
        disallowed = []
        for direction,node in self.adjacent_nodes.items():
            adj_biomes.append(node.tile_type)
            nodes.append(node)
        for tile in self.valid_tiles[::]:
            rule = Tile.rules[tile]
            for i,biome in enumerate(adj_biomes):
                if biome in rule['disallowed']:
                    self.valid_tiles.remove(tile)
                    disallowed.append(biome)
                    break
                elif biome in rule['required_by']:
                    index = i
                    tmp = nodes[i]
                    needed_tiles = tmp.required_adjacent[::]
            else:
                needed_tiles = [n for n in needed_tiles if n in self.valid_tiles] if needed_tiles is not None and self.valid_tiles is not None else None
                if needed_tiles and self.valid_tiles:
                    self.tile_type = random.choice(needed_tiles)
                    self.valid_tiles = None
                    nodes[index].required_adjacent.remove(self.tile_type)
                if None not in adj_biomes:
                    for req_biome in rule['required']:
                        if req_biome not in adj_biomes:
                            self.valid_tiles.remove(tile)

    def set_tile_type(self):
        if self.tile_type is not None:
            return
        elif len(self.valid_tiles) == len(Tile.tile_types):
            if not any([node.tile_type for node in self.adjacent_nodes.values()]):
                return

        else:
            lottery = []
            weights = {k:v for k,v in Tile.weights.items() if k in self.valid_tiles}
            weight_scale = min([v for v in weights.values()])
            weights = {k: round(v /weight_scale) for k, v in weights.items()}
            for type,scaled_weight in weights.items():
                lottery += [type]*scaled_weight
            self.valid_tiles = None
            self.tile_type = random.choice(lottery)
            self.required_adjacent = Tile.rules[self.tile_type]['required']

    def assign_nodes(self,tile_list):
        for direction,coords in self.adjacent_tiles.items():
            if coords == None:
                del self.adjacent_nodes[direction]
                continue
            self.adjacent_nodes[direction] = tile_list[coords[1]][coords[0]]

    def __str__(self):
        spaceing = len("|" + "/".join([t[0] for t in Tile.tile_types]) + '|')-2
        if self.tile_type is not None:

            return '|'+ str(self.tile_type).center(spaceing)+'|'
        else:
            return "|"+"/".join([t[0] for t in self.valid_tiles]).center(spaceing)+'|'



class TileCollapse:

    def __init__(self):
        pass

    def generate_initial_model(self):

        assigned_coords = []
        for i in range(1):
            assigned_coords.append([random.randrange(10),random.randrange(10)])
        data  = []
        rnge = 10
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

                data[y].append(Tile([x,y],next_coords))
                if [x,y] in assigned_coords:
                    tmp_node = data[y][-1]
                    tmp_node.tile_type = 'sea'
                    tmp_node.valid_tiles = None

        for _ in data:
            for node in _:
                node.assign_nodes(data)
        self.data = data

    def update_nodes(self):
        for row in self.data:
            for node in row:
                if node.valid_tiles != None:
                    node.check_valid_types()
                    node.set_tile_type()
        [print(" ".join([str(__) for __ in _])) for _ in dT.data]
        print('\n')
if __name__ == '__main__':
    dT = TileCollapse()

    dT.generate_initial_model()
    for i in range(10):
        dT.update_nodes()
