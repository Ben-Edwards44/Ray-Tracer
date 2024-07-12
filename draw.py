import pygame
import numpy


def create_window(screen_width, screen_height):
    pygame.init()
    window = Screen(screen_width, screen_height)

    return window


class Screen:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.window = pygame.display.set_mode((width, height))
        self.pixel_array = ScreenPixels(self.window, width, height)

    def draw(self):
        self.window.fill((0, 0, 0))
        self.pixel_array.draw_to_screen()

        pygame.display.update()


class ScreenPixels:
    def __init__(self, screen_window, width, height):
        self.window = screen_window
        self.width = width
        self.height = height

        self.pixel_array = numpy.zeros((width, height, 3), numpy.uint)

    def set_pixel_array(self, new_pixel_array):
        #NOTE: new_pixel array will be a flattened array containing rgb values from 0.0 - 1.0
        for i in range(self.width):
            for x in range(self.height):
                flattened_inx = (x * self.width + i) * 3  #i and x are swapped because the incoming array will have been transposed

                for y in range(3):
                    pixel_colour = int(new_pixel_array[flattened_inx + y] * 255)

                    if pixel_colour > 255:
                        pixel_colour = 255
                    elif pixel_colour < 0:
                        pixel_colour = 0

                    self.pixel_array[i][x][y] = pixel_colour

    def draw_to_screen(self):
        pygame.surfarray.blit_array(self.window, self.pixel_array)


def check_user_input():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()


def draw_screen(window, pixel_data):
    window.pixel_array.set_pixel_array(pixel_data)
    window.draw()