import draw
import camera
from gpu import api, cuda


SCREEN_WIDTH = 500
ASPECT_RATIO = 2

CUDA_FILEPATH = "gpu/main.cu"


def run_raytracer(cuda_script, cam):
    data_to_send = {
        "camera" : cam.data_dict()
    }

    api.send_to_cuda(data_to_send)

    cuda_script.run()
    
    recieved = api.recieve_from_cuda()

    return recieved


def main():
    cuda_script = cuda.CudaScript(CUDA_FILEPATH, False)
    cam = camera.Camera((0, 0, 0), 1.5, ASPECT_RATIO, 0.1, SCREEN_WIDTH)
    window = draw.create_window(cam.image.width, cam.image.height)

    while True:
        raytracing_data = run_raytracer(cuda_script, cam)
        pixels = raytracing_data["pixel_data"]

        draw.draw_screen(window, pixels)
        draw.check_user_input()


main()