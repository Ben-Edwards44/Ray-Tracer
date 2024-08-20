#include "utils.cu"


__host__ __device__ class Texture {
    //ideally, I would use inheritance and polymorphism. But virtual functions are weird with CUDA, so I'll just use one big class
    public:
        static const int GRADIENT = 0;
        static const int CHECKERBOARD = 1;

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

        __device__ Vec3 get_texture_colour(float u, float v) {
            if (type == GRADIENT) {
                return gradient(u, v);
            } else if (type == CHECKERBOARD) {
                return checkerboard(u, v);
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
};


__host__ __device__ struct Material {
    Vec3 colour;

    float emission_strength;
    Vec3 emission_colour;

    float smoothness;  //[0, 1]. 0 = perfect diffuse, 1 = perfect reflect

    bool using_texture;

    Texture texture;
};