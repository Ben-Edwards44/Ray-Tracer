import draw
import mesh
import camera
from gpu import api, cuda


SCREEN_WIDTH = 500
ASPECT_RATIO = 2

CUDA_FILEPATH = "gpu/main.cu"

RAY_REFLECT_LIMIT = 5
RAYS_PER_PIXEL = 1


def run_raytracer(cuda_script):
    #assumes all the appropriate data has already been sent
    cuda_script.run()
    recieved = api.recieve_from_cuda()

    return recieved


def send_data(cam, meshes):
    mesh_data = {}
    for i, x in enumerate(meshes):
        mesh_data[f"{i}"] = x.data_dict()

    mesh_data["num_meshes"] = [len(mesh_data)]

    data_to_send = {
        "camera" : cam.data_dict(),
        "mesh_data" : mesh_data,
        "ray_data" : {
            "reflect_limit" : [RAY_REFLECT_LIMIT],
            "rays_per_pixel" : [RAYS_PER_PIXEL]
        }
    }

    api.send_to_cuda(data_to_send)


def setup_scene():
    light = mesh.Sphere((0.2, 0.2, 0.2), 2, (1, 1, 1), 0, 2.4, 1, 2)

    sphere1 = mesh.Sphere((0, 0, 0.75), 0, (0, 0, 0), -0.5, 0, 3, 0.5)
    sphere2 = mesh.Sphere((0, 0.6, 0), 0, (0, 0, 0), 1.2, -0.1, 2, 0.4)
    sphere3 = mesh.Sphere((1, 0, 0), 0, (0, 0, 0), 0, -5, 4, 5)

    return [light, sphere1, sphere2, sphere3]


def main():
    cuda_script = cuda.CudaScript(CUDA_FILEPATH, False)
    cam = camera.Camera((0, 0, 0), 50, ASPECT_RATIO, 0.1, SCREEN_WIDTH)
    window = draw.create_window(cam.image.width, cam.image.height)
    meshes = setup_scene()

    while True:
        send_data(cam, meshes)

        raytracing_data = run_raytracer(cuda_script)
        pixels = raytracing_data["pixel_data"]

        draw.draw_screen(window, pixels)
        draw.check_user_input()


main()