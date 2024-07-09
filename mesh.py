class Mesh:
    SPHERE_ID = 0

    def __init__(self, colour):
        if type(colour) == list:
            self.colour = colour
        else:
            self.colour = list(colour)


class Sphere(Mesh):
    def __init__(self, colour, x, y, z, r):
        super().__init__(colour)

        self.x = x
        self.y = y
        self.z = z
        self.radius = r

    def data_dict(self):
        return {
            "type" : [Mesh.SPHERE_ID],
            "center" : [self.x, self.y, self.z],
            "radius" : [self.radius],
            "colour" : self.colour
        }