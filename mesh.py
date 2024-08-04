class Material:
    #type
    DIFFUSE = 0
    MIRROR = 1
    METAL = 2

    def __init__(self, colour, emit_strength, emit_colour, mat_type, options):
        self.colour = colour
        self.emit_strength = emit_strength
        self.emit_colour = emit_colour
        self.mat_type = mat_type
        self.options = options

    def data_dict(self):
        return {
            "colour" : self.colour,
            "emission_strength" : [self.emit_strength],
            "emission_colour" : self.emit_colour,
            "type" : [self.mat_type],
            "options" : self.options
        }


class Mesh:
    SPHERE_ID = 0

    def __init__(self, material):
        self.material = material

    def check_list(self, value):
        #ensure it is a list not a tuple
        if type(value) == list:
            return value
        else:
            return list(value)


class Sphere(Mesh):
    def __init__(self, material, x, y, z, r):
        super().__init__(material)

        self.x = x
        self.y = y
        self.z = z
        self.radius = r

    def data_dict(self):
        return {
            "type" : [Mesh.SPHERE_ID],
            "center" : [self.x, self.y, self.z],
            "radius" : [self.radius],
            "material" : self.material.data_dict()
        }