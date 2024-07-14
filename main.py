import draw
import mesh
import camera
from gpu import api, cuda


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 500

CUDA_FILEPATH = "gpu/main.cu"

RAY_REFLECT_LIMIT = 3
RAYS_PER_PIXEL = 100

STATIC_SCENE = True


def run_raytracer(cam, meshes, cuda_script, is_initial_run):
    #assumes all the appropriate data has already been sent
    needs_block = STATIC_SCENE and is_initial_run

    if needs_block:
        #we need to (re)send the data (it has either changed or never been sent at all)
        send_data(cam, meshes)

    cuda_script.run(needs_block)

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
        },
        "static_scene" : [1] if STATIC_SCENE else [0]
    }

    api.send_to_cuda(data_to_send)


def setup_scene():
    light1 = mesh.Sphere((0, 0, 0), 3, (1, 1, 1), 0, 2.6, 2.5, 2)

    sphere1 = mesh.Sphere((0, 0, 1), 0, (1, 1, 1), -0.5, 0, 3, 0.3)
    sphere2 = mesh.Sphere((0, 1, 0), 0, (0, 0, 0), 1.2, -0.1, 2, 0.4)
    sphere3 = mesh.Sphere((1, 0, 0), 0, (0, 0, 0), 0, -5, 4, 5)

    return [light1, sphere1, sphere2, sphere3]


def main():
    api.clear_files()

    cuda_script = cuda.CudaScript(CUDA_FILEPATH, False)
    cam = camera.Camera((0, 0, 0), 50, 0.1, SCREEN_WIDTH, SCREEN_HEIGHT)
    window = draw.create_window(cam.image.width, cam.image.height)
    meshes = setup_scene()

    first_render = True

    while True:
        raytracing_data = run_raytracer(cam, meshes, cuda_script, first_render)
        pixels = raytracing_data["pixel_data"]

        if first_render:
            first_render = False

        draw.draw_screen(window, pixels)

        exit = draw.check_user_input()

        if exit:
            cuda_script.kill_process()
            break


main()