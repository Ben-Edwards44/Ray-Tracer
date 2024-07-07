class Mesh:
    SPHERE_ID = 0

    def __init__(self):
        pass


class Sphere(Mesh):
    def __init__(self, x, y, z, r):
        super().__init__()

        self.x = x
        self.y = y
        self.z = z
        self.radius = r

    def data_dict(self):
        return {
            "type" : [Mesh.SPHERE_ID],
            "center" : [self.x, self.y, self.z],
            "radius" : [self.radius]
        }