#include <cmath>
#include <vector>
#include <SFML/Graphics.hpp>
#include "gpu/main.cu"


const int WIDTH = 800;
const int HEIGHT = 500;
const float ASPECT = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);

const std::string CAPTION = "ray tracer";

const float PI = 3.141592653589793;


class Camera {
    public:
        CamData gpu_struct;

        Camera() {
            assign_default();

            gpu_struct = create_gpu_struct();
        }

    private:
        Vec3 cam_pos;
        float fov;
        float focal_len;

        void assign_default() {
            cam_pos.x = 0;
            cam_pos.y = 0;
            cam_pos.z = 0;

            fov = 60 * (PI / 180);
            focal_len = 0.1;
        }

        CamData create_gpu_struct() {
            float viewport_width = 2 * focal_len * tan(fov / 2);
            float viewport_height = viewport_width / ASPECT;

            Vec3 tl_pos(cam_pos.x - viewport_width / 2, cam_pos.y + viewport_height / 2, cam_pos.z + focal_len);
        
            float delta_u = viewport_width / WIDTH;
            float delta_v = viewport_height / HEIGHT;

            return CamData{cam_pos, tl_pos, focal_len, delta_u, delta_v, WIDTH, HEIGHT};
        }
};


class Meshes {
    public:
        std::vector<Sphere> spheres;

        Meshes() {
            create_gpu_struct();
        }

    private:
        const int DIFFUSE = 0;
        const int MIRROR = 1;
        const int METAL = 2;

        void create_gpu_struct() {
            Material dif_mat{Vec3(1, 0.2, 0.1), 0, Vec3(0, 0, 0), DIFFUSE};
            Material dif_mat2{Vec3(0.2, 0.1, 1), 0, Vec3(0, 0, 0), DIFFUSE};
            Material ref_mat{Vec3(0.8, 0.8, 0.8), 0, Vec3(0, 0, 0), MIRROR};
            Material met_mat{Vec3(0.8, 0.8, 0.8), 0, Vec3(0, 0, 0), METAL, 0.3};

            Sphere s1(Vec3(-1, 0, 2.2), 0.5, ref_mat);
            Sphere s2(Vec3(1, 0, 2), 0.5, met_mat);
            Sphere s3(Vec3(0, 0, 2.1), 0.5, dif_mat2);
            Sphere s4(Vec3(0, -5, 4), 5, dif_mat);

            spheres.push_back(s1);
            spheres.push_back(s2);
            spheres.push_back(s3);
            spheres.push_back(s4);
        }
};


class RenderSettings {
    public:
        RenderData gpu_struct;

        RenderSettings(int num_spheres) {
            assign_default();

            gpu_struct = create_gpu_struct(num_spheres);
        }

    private:
        int reflect_limit;
        int rays_per_pixel;

        bool static_scene;

        Vec3 sky_colour;

        void assign_default() {
            reflect_limit = 5;
            rays_per_pixel = 100;
            static_scene = true;

            sky_colour.x = 0.87;
            sky_colour.y = 0.98;
            sky_colour.z = 1;
        }

        RenderData create_gpu_struct(int num_spheres) {
            int start_frame_num = 0;

            return RenderData{rays_per_pixel, reflect_limit, num_spheres, start_frame_num, static_scene, sky_colour};
        }
};


Scene create_scene(int img_width, int img_height) {
    Camera cam;
    Meshes meshes;
    RenderSettings render_settings(meshes.spheres.size());

    int len_pixel_array = img_width * img_height * 3;

    std::vector<float> previous_render(len_pixel_array);

    return Scene{cam.gpu_struct, render_settings.gpu_struct, meshes.spheres, len_pixel_array, previous_render};
}


std::vector<float> get_pixel_colours(Scene *scene) {
    //get the pixel colours from the raytracer
    render(scene);  //will update scene.previous_render

    return scene->previous_render;
}


std::vector<sf::Uint8> parse_pixel_colours(std::vector<float> pixel_colours) {
    //turn array of rgb floats between 0 and 1 into something that can be drawn
    std::vector<sf::Uint8> parsed_pixel_colours(WIDTH * HEIGHT * 4);

    for (int x = 0; x < WIDTH; x++) {
        for (int y = 0; y < HEIGHT; y++) {
            //do stuff
            int pixel_colour_inx = (y * WIDTH + x) * 3;
            int result_inx = (y * WIDTH + x) * 4;

            //add the rgb colours
            for (int i = 0; i < 3; i++) {
                int colour = pixel_colours[pixel_colour_inx + i] * 255;

                if (colour > 255) {
                    colour = 255;
                } else if (colour < 0) {
                    colour = 0;
                }

                sf::Uint8 converted_colour = colour;
                parsed_pixel_colours[result_inx + i] = converted_colour;
            }

            parsed_pixel_colours[result_inx + 3] = 255;  //add alpha
        }
    }

    return parsed_pixel_colours;
}


void draw_screen(sf::RenderWindow *window, std::vector<float> pixel_colours) {
    std::vector<sf::Uint8> rgba_colours = parse_pixel_colours(pixel_colours);

    //create a texture continaing the pixel colours
    sf::Texture texture;
    texture.create(WIDTH, HEIGHT);
    texture.update(rgba_colours.data());

    sf::Sprite sprite(texture);

    window->draw(sprite);
    window->display();
}


int main() {
    sf::VideoMode dims(WIDTH, HEIGHT);
    sf::RenderWindow window(dims, CAPTION);

    Scene scene = create_scene(WIDTH, HEIGHT);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        std::vector<float> pixel_colours = get_pixel_colours(&scene);
        draw_screen(&window, pixel_colours);
    }

    return 0;
}