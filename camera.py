import math


class Camera:
    def __init__(self, pos, fov, aspect_ratio, focal_length, img_width):
        self.x, self.y, self.z = pos

        self.fov = math.radians(fov)
        self.aspect_ratio = aspect_ratio
        self.focal_length = focal_length

        self.image = Image(img_width, aspect_ratio)
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
    def __init__(self, width, ideal_aspect_ratio):
        self.width = width
        self.height = int(width / ideal_aspect_ratio)
        self.actual_aspect_ratio = self.width / self.height

    dimensions = lambda self: [self.width, self.height]


class Viewport:
    def __init__(self, height, image):
        self.height = height
        self.width = height * image.actual_aspect_ratio

    dimensions = lambda self: [self.width, self.height]