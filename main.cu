#include "object.cu"
#include "gpu/dispatch.cu"

#include <cmath>
#include <vector>
#include <chrono>

#include <SFML/Graphics.hpp>


const int SCENE_NUM = 3;

const int WIDTH = 1000;
const int HEIGHT = 800;
const float ASPECT = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);  //static cast is used to stop integer division

const std::string CAPTION = "ray tracer";


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
            //these values can be changed
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


class ImageTexture {
    public:
        std::string PARSED_TEXTURE_FILENAME = "textures/parsed_textures.txt";

        ImageTexture(std::string filename) {
            parse_file(filename);
        }

        Texture get_device_texture() {
            return Texture::create_image(width, height, rgb_values);
        }

    private:
        int width;
        int height;

        std::vector<Vec3> rgb_values;

        void parse_file(std::string filename) {
            std::vector<std::string> lines = read_file(PARSED_TEXTURE_FILENAME);

            for (int i = 0; i < lines.size(); i++) {
                if (lines[i] == filename) {
                    width = std::stoi(lines[i + 1]);
                    height = std::stoi(lines[i + 2]);
                    rgb_values = parse_rgb_values(lines[i + 3]);

                    return;
                }
            }

            //if we have not yet returned from the function, the file was not found
            throw std::runtime_error("Image file not found.\n");
        }

        std::vector<Vec3> parse_rgb_values(std::string rgb_string) {
            std::vector<std::string> splitted = split_string(rgb_string, ' ');

            std::vector<Vec3> parsed_rgb;

            //NOTE: the last character will just be "", so loop to the last but one character
            for (int i = 0; i < splitted.size() - 1; i += 3) {
                float r = std::stof(splitted[i]);
                float g = std::stof(splitted[i + 1]);
                float b = std::stof(splitted[i + 2]);

                parsed_rgb.push_back(Vec3(r, g, b));
            }

            return parsed_rgb;
        }
};


class Meshes {
    public:
        std::vector<Sphere> spheres;
        std::vector<Triangle> triangles;
        std::vector<Quad> quads;
        std::vector<OneWayQuad> one_way_quads;
        std::vector<Cuboid> cuboids;

        Meshes(int test_scene) {
            switch (test_scene) {
                case 0:
                    monkey_test_scene();
                    break;
                case 1:
                    reflection_test_scene();
                    break;
                case 2:
                    texture_test_scene();
                    break;
                case 3:
                    refract_test_scene();
                    break;
                default:
                    throw std::domain_error("Test scene must be number between 0 and 3 (inclusive).\n");
            }
        }

    private:
        void add_obj_faces(Object obj, Material mat) {
            //parse the object faces into triangles and add them to the list of triangles
            for (std::vector<float3> face : obj.faces) {
                if (face.size() == 3) {
                    Triangle tri(Vec3(face[0]), Vec3(face[1]), Vec3(face[2]), mat);
                    triangles.push_back(tri);
                } else if (face.size() == 4) {
                    Quad quad(Vec3(face[0]), Vec3(face[1]), Vec3(face[2]), Vec3(face[3]), mat);
                    quads.push_back(quad);
                } else {
                    throw std::logic_error("Only triangle or quad meshes are supported.\n");
                }
            }
        }

        void monkey_test_scene() {
            //setup simple test scene with a cornell box, suzanne mesh and sphere
            create_cornell_box(Vec3(-0.5, 0.5, 1.2), 1, 1, 1, 0.5);

            Texture monkey_tex = Texture::create_const_colour(Vec3(1, 1, 1));
            Material monkey_mat = Material::create_standard(monkey_tex, 0);

            Object m("low_poly_monkey.obj");
            m.enlarge(0.3);
            m.rotate(0, 2.3, 0);
            m.translate(0.1, -0.1, 1.6);

            add_obj_faces(m, monkey_mat);

            Texture sphere_tex = Texture::create_const_colour(Vec3(0.8, 0.8, 0.8));
            Material sphere_mat = Material::create_standard(sphere_tex, 1);
            Sphere sphere(Vec3(-0.25, -0.25, 1.95), 0.25, sphere_mat);

            spheres.push_back(sphere);
        }

        void reflection_test_scene() {
            //simple test scene with spheres of different smoothness values
            create_cornell_box(Vec3(-0.5, 0.5, 1.2), 1, 1, 1, 0.5);

            Texture sphere_tex = Texture::create_const_colour(Vec3(1, 1, 1));

            Material a = Material::create_standard(sphere_tex, 0);
            Material b = Material::create_standard(sphere_tex, 0.33);
            Material c = Material::create_standard(sphere_tex, 0.66);
            Material d = Material::create_standard(sphere_tex, 1);

            spheres.push_back(Sphere(Vec3(-0.2, 0.2, 1.7), 0.15, a));
            spheres.push_back(Sphere(Vec3(0.2, 0.2, 1.7), 0.15, b));
            spheres.push_back(Sphere(Vec3(-0.2, -0.2, 1.7), 0.15, c));
            spheres.push_back(Sphere(Vec3(0.2, -0.2, 1.7), 0.15, d));
        }

        void texture_test_scene() {
            //test scene with spheres with tetxtures
            create_cornell_box(Vec3(-0.5, 0.5, 1.2), 1, 1, 1, 0.5);

            ImageTexture earth("earth.png");
            Material earth_mat = Material::create_standard(earth.get_device_texture(), 0);

            spheres.push_back(Sphere(Vec3(0, 0, 1.7), 0.25, earth_mat));

            Texture tri_tex = Texture::create_checkerboard(Vec3(1, 1, 1), Vec3(0, 0, 0), 4);
            Material tri_mat = Material::create_standard(tri_tex, 0);

            Triangle t1(Vertex{Vec3(0.1, 0, 1.7), Vec2(0, 0)}, Vertex{Vec3(0.6, 0.5, 1.9), Vec2(0, 1)}, Vertex{Vec3(0.8, 0.4, 2), Vec2(1, 1)}, tri_mat);

            triangles.push_back(t1);
        }

        void refract_test_scene() {
            Texture t = Texture::create_const_colour(Vec3(1, 1, 1));
            Material no = Material::create_standard(t, 0);
            Material yes = Material::create_refractive(t, 1.8);

            spheres.push_back(Sphere(Vec3(-0.4, -0.1, 3), 0.3, yes));
            spheres.push_back(Sphere(Vec3(0.4, -0.1, 3), 0.3, no));

            Texture t2 = Texture::create_checkerboard(Vec3(0.1, 1, 0.1), Vec3(0.1, 0.5, 0.1), 8);
            Material fl = Material::create_standard(t2, 0);

            Texture t3 = Texture::create_const_colour(Vec3(1, 0.2, 0.2));
            Material b = Material::create_standard(t3, 0);

            quads.push_back(Quad(Vec3(-2, -0.5, 0), Vec3(2, -0.5, 0), Vec3(2, -0.5, 5), Vec3(-2, -0.5, 5), fl));
            quads.push_back(Quad(Vec3(-2, -0.5, 5), Vec3(2, -0.5, 5), Vec3(2, 2, 5), Vec3(-2, 2, 5), b));
        }

        void create_cornell_box(Vec3 tl_near_pos, float width, float height, float depth, float light_width) {
            Texture floor_tex = Texture::create_checkerboard(Vec3(0.1, 0.8, 0.1), Vec3(0.1, 0.5, 0.1), 8);
            Texture l_wall_tex = Texture::create_const_colour(Vec3(1, 0.2, 0.2));
            Texture r_wall_tex = Texture::create_const_colour(Vec3(0.3, 0.3, 1));
            Texture back_tex = Texture::create_const_colour(Vec3(0.2, 0.2, 0.2));
            Texture roof_tex = Texture::create_const_colour(Vec3(0.9, 0.9, 0.9));
            Texture front_tex = Texture::create_const_colour(Vec3(1, 1, 1));

            Material floor = Material::create_standard(floor_tex, 0);
            Material l_wall = Material::create_standard(l_wall_tex, 0);
            Material r_wall = Material::create_standard(r_wall_tex, 0);
            Material back = Material::create_standard(back_tex, 0);
            Material roof = Material::create_standard(roof_tex, 0);
            Material front = Material::create_standard(front_tex, 0);

            //offset vectors
            Vec3 w(width, 0, 0);
            Vec3 h(0, height, 0);
            Vec3 d(0, 0, depth);

            quads.push_back(Quad(tl_near_pos - h, tl_near_pos - h + w, tl_near_pos - h + w + d, tl_near_pos - h + d, floor));
            quads.push_back(Quad(tl_near_pos, tl_near_pos - h, tl_near_pos - h + d, tl_near_pos + d, l_wall));
            quads.push_back(Quad(tl_near_pos + w, tl_near_pos + w - h, tl_near_pos + w - h + d, tl_near_pos + w + d, r_wall));
            quads.push_back(Quad(tl_near_pos + d, tl_near_pos + w + d, tl_near_pos + w - h + d, tl_near_pos - h + d, back));
            quads.push_back(Quad(tl_near_pos, tl_near_pos + d, tl_near_pos + w + d, tl_near_pos + w, roof));
            one_way_quads.push_back(OneWayQuad(tl_near_pos, tl_near_pos + w, tl_near_pos + w - h, tl_near_pos - h, front, false));  //front wall is one way so we can see through it

            //add the light
            Material light_mat = Material::create_emissive(Vec3(1, 1, 1), 6);

            Vec3 light_tl_near_pos(tl_near_pos.x + width / 2 - light_width / 2, tl_near_pos.y, tl_near_pos.z + depth / 2 - light_width / 2);  //ensure light is in center of roof
            Cuboid light(light_tl_near_pos, light_width, 0.04, light_width, light_mat);

            cuboids.push_back(light);
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
        bool antialias;

        Vec3 sky_colour;

        void assign_default() {
            //these settings can be changed
            reflect_limit = 5;
            rays_per_pixel = 10;

            antialias = true;

            sky_colour.x = 0.7;
            sky_colour.y = 0.7;
            sky_colour.z = 0.98;
        }

        RenderData create_gpu_struct(int num_spheres) {
            return RenderData{rays_per_pixel, reflect_limit, antialias, sky_colour};
        }
};


Scene create_scene(int img_width, int img_height) {
    Camera cam;
    Meshes meshes(SCENE_NUM);
    RenderSettings render_settings(meshes.spheres.size());

    int len_pixel_array = img_width * img_height * 3;

    return Scene(cam.gpu_struct, render_settings.gpu_struct, meshes.spheres, meshes.triangles, meshes.quads, meshes.one_way_quads, meshes.cuboids, len_pixel_array);
}


int get_time() {
    //get ms since epoch
    auto clock = std::chrono::system_clock::now();
    auto duration = clock.time_since_epoch();
    int time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    return time;
}


std::vector<float> get_pixel_colours(Scene *scene) {
    //get the pixel colours from the raytracer
    int time = get_time();
    render(scene, time);  //will update scene.previous_render

    return scene->previous_render;
}


std::vector<sf::Uint8> parse_pixel_colours(std::vector<float> pixel_colours) {
    //turn array of rgb floats between 0 and 1 into something that can be drawn
    std::vector<sf::Uint8> parsed_pixel_colours(WIDTH * HEIGHT * 4);

    for (int x = 0; x < WIDTH; x++) {
        for (int y = 0; y < HEIGHT; y++) {
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

    int start_time = get_time();

    while (window.isOpen()) {
        //check if the window has been closed
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        std::vector<float> pixel_colours = get_pixel_colours(&scene);
        draw_screen(&window, pixel_colours);

        int elapsed = get_time() - start_time;
        float fps = 1000 / static_cast<float>(elapsed);
        start_time = get_time();

        printf("FPS: %f\r", fps);
        fflush(stdout);  //since the \n character is not used, stdout must be manually flushed
    }

    return 0;
}