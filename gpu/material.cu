#include "utils.cu"
#include <vector>


__host__ __device__ class Texture {
    //ideally, I would use inheritance and polymorphism. But virtual functions are weird with CUDA, so I'll just use one big class
    public:
        static const int GRADIENT = 0;
        static const int CHECKERBOARD = 1;
        static const int IMAGE = 2;

        int type;

        __host__ __device__ Texture() {}

        __host__ Texture(int texture_type) {
            type = texture_type;
        }

        __host__ void assign_checkerboard(Vec3 light_colour, Vec3 dark_colour, int num_sq) {
            light = light_colour;
            dark = dark_colour;
            num_squares = num_sq;
        }

        __host__ void assign_image(int width, int height, std::vector<Vec3> rgb_values) {
            img_tex_width = width;
            img_tex_height = height;

            allocate_memory(rgb_values);
        }

        __device__ Vec3 get_texture_colour(Vec2 uv_coord) {
            float u = uv_coord.x;
            float v = uv_coord.y;

            if (type == GRADIENT) {
                return gradient(u, v);
            } else if (type == CHECKERBOARD) {
                return checkerboard(u, v);
            } else if (type == IMAGE) {
                return image(u, v);
            } else {
                return Vec3(0, 0, 0);
            }
        }

    private:
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


__host__ __device__ struct Material {
    Vec3 colour;

    float emission_strength;
    Vec3 emission_colour;

    float smoothness;  //[0, 1]. 0 = perfect diffuse, 1 = perfect reflect

    bool using_texture;

    Texture texture;
};