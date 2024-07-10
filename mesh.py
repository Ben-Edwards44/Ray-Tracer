class Material:
    def __init__(self, colour, emit_strength, emit_colour):
        self.colour = colour
        self.emit_strength = emit_strength
        self.emit_colour = emit_colour

    def data_dict(self):
        return {
            "colour" : self.colour,
            "emission_strength" : [self.emit_strength],
            "emission_colour" : self.emit_colour
        }


class Mesh:
    SPHERE_ID = 0

    def __init__(self, colour, emit_strength, emit_colour):
        self.material = Material(self.check_list(colour), emit_strength, self.check_list(emit_colour))

    def check_list(self, value):
        #ensure it is a list not a tuple
        if type(value) == list:
            return value
        else:
            return list(value)


class Sphere(Mesh):
    def __init__(self, colour, emit_strength, emit_colour, x, y, z, r):
        super().__init__(colour, emit_strength, emit_colour)

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