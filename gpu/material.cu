#include "utils.cu"
#include <vector>


__host__ __device__ class Texture {
    //ideally, I would use inheritance and polymorphism. But virtual functions are weird with CUDA, so I'll just use one big class
    public:
        static const int COLOUR = 0;
        static const int GRADIENT = 1;
        static const int CHECKERBOARD = 2;
        static const int IMAGE = 3;

        int type;

        __host__ __device__ Texture() {}

        __host__ Texture(int texture_type) {
            type = texture_type;
        }

        //initialisers for each texture type
        __host__ static Texture create_const_colour(Vec3 texture_colour) {
            Texture tex(COLOUR);
            tex.colour = texture_colour;

            return tex;
        }

        __host__ static Texture create_gradient() {
            return Texture(GRADIENT);
        }

        __host__ static Texture create_checkerboard(Vec3 light_colour, Vec3 dark_colour, int num_sq) {
            Texture tex(CHECKERBOARD);

            tex.light = light_colour;
            tex.dark = dark_colour;
            tex.num_squares = num_sq;

            return tex;
        }

        __host__ static Texture create_image(int width, int height, std::vector<Vec3> rgb_values) {
            Texture tex(IMAGE);

            tex.img_tex_width = width;
            tex.img_tex_height = height;

            tex.allocate_memory(rgb_values);

            return tex;
        }

        __device__ Vec3 get_texture_colour(Vec2 uv_coord) {
            float u = uv_coord.x;
            float v = uv_coord.y;

            switch (type) {
                case COLOUR:
                    return constant_colour();
                case GRADIENT:
                    return gradient(u, v);
                case CHECKERBOARD:
                    return checkerboard(u, v);
                case IMAGE:
                    return image(u, v);
                default:
                    return Vec3(0, 0, 0);
            }
        }

    private:
        //constant colour
        Vec3 colour;

        __device__ Vec3 constant_colour() {
            return colour;
        }

        //graident
        __device__ Vec3 gradient(float u, float v) {
            return Vec3(u, v, 0);
        }

        //checkerboard
        Vec3 light;
        Vec3 dark;

        int num_squares;

        __device__ Vec3 checkerboard(float u, float v) {
            int u_coord = u * num_squares;
            int v_coord = v * num_squares;

            if ((u_coord + v_coord) % 2 == 0) {
                return light;
            } else {
                return dark;
            }
        }

        //image
        int img_tex_width;
        int img_tex_height;

        Vec3 *img_rgb;

        __host__ void allocate_memory(std::vector<Vec3> rgb_values) {
            int size = sizeof(rgb_values[0]) * rgb_values.size();

            cudaError_t error = cudaMalloc((void **)&img_rgb, size);  //allocate the memory
            check_cuda_error(error);

            Vec3 *rgb_array = &rgb_values[0];  //get the pointer to the underlying array

            error = cudaMemcpy(img_rgb, rgb_array, size, cudaMemcpyHostToDevice);  //copy the data over
            check_cuda_error(error);
        }

        __device__ Vec3 image(float u, float v) {
            int u_coord = (img_tex_width - 1) * u;
            int v_coord = (img_tex_height - 1) * v;

            return img_rgb[v_coord * img_tex_width + u_coord];
        }
};


__host__ __device__ class Material {
    public:
        Texture texture;

        float smoothness;  //[0, 1]. 0 = perfect diffuse, 1 = perfect reflect
        float refractive_index;

        Vec3 emitted_light;

        bool need_uv;  //can optimise by not calculating uv coords if not needed
        bool is_glass = false;

        __host__ __device__ Material() {}

        __host__ Material(Texture mat_tex, float smoothness_val) {
            texture = mat_tex;
            smoothness = smoothness_val;

            emitted_light = Vec3(0, 0, 0);

            need_uv = texture.type != Texture::COLOUR;
        }

        __host__ Material(Texture mat_tex, float smoothness_val, float emit_strength, Vec3 emit_colour) {
            //constructor for emissive texture
            texture = mat_tex;
            smoothness = smoothness_val;
            emitted_light = emit_colour * emit_strength;

            need_uv = texture.type != Texture::COLOUR;
        }
};