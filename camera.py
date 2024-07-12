import math


class Camera:
    def __init__(self, pos, fov, focal_length, img_width, img_height):
        self.x, self.y, self.z = pos

        self.fov = math.radians(fov)
        self.focal_length = focal_length

        self.image = Image(img_width, img_height)
        self.viewport = self.get_viewport()

    def get_viewport(self):
        plane_height = 2 * self.focal_length * math.tan(self.fov / 2)
        viewport = Viewport(plane_height, self.image)

        return viewport

    def data_dict(self):
        return {
            "position" : [self.x, self.y, self.z],
            "focal_length" : [self.focal_length],
            "viewport" : {
                "dimensions" : self.viewport.dimensions()
            },
            "image" : {
                "dimensions" : self.image.dimensions()
            }
        }
    

class Image:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.aspect_ratio = self.width / self.height

    dimensions = lambda self: [self.width, self.height]


class Viewport:
    def __init__(self, height, image):
        self.height = height
        self.width = height * image.aspect_ratio

    dimensions = lambda self: [self.width, self.height]