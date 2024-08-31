from os import listdir
from PIL import Image


OUTPUT_FILE = "parsed_textures.txt"

IMAGE_EXTENSIONS = ("png", "jpg", "jpeg")


class Texture:
    def __init__(self, filename):
        self.filename = filename

        self.image = Image.open(self.filename)
        self.width, self.height = self.image.size

        self.pixel_colours = self.get_rgb()

    def get_rgb(self):
        rgb_image = self.image.convert("RGB")

        pixels = []
        for y in range(self.height):
            for x in range(self.width):
                pixels.append(rgb_image.getpixel((x, y)))

        return pixels
    
    def get_string(self):
        plain_rgb = ""
        for i in self.pixel_colours:
            for x in i:
                plain_rgb += f"{x / 256} "

        return f"{self.filename}\n{self.width}\n{self.height}\n{plain_rgb}\n"
    

def get_textures():
    files = listdir(".")

    textures = []
    for i in files:
        _, extension = i.split(".")

        if extension in IMAGE_EXTENSIONS:
            textures.append(Texture(i))

    return textures


def write_file(textures):
    write_string = f"{len(textures)}\n"

    for i in textures:
        write_string += i.get_string()

    with open(OUTPUT_FILE, "w") as file:
        file.write(write_string)


def main():
    textures = get_textures()

    write_file(textures)


main()