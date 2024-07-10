#include <cmath>
#include <curand_kernel.h>


//common overloads for working with float3 data types
__device__ float3 operator+(float3 &a, float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}


__device__ float3 operator-(float3 &a, float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}


__host__ __device__ class Vec3 {
    public:
        float x;
        float y;
        float z;

        __host__ __device__ Vec3(float val_x, float val_y, float val_z) {
            x = val_x;
            y = val_y;
            z = val_z;
        }

        __device__ Vec3(float3 vector) {
            x = vector.x;
            y = vector.y;
            z = vector.z;
        }

        __device__ Vec3() {}

        //common operations
        __device__ Vec3 operator+(Vec3 other_vec) {
            return Vec3(x + other_vec.x, y + other_vec.y, z + other_vec.z);
        }

        __device__ Vec3 operator+(float scalar) {
            return Vec3(x + scalar, y + scalar, z + scalar);
        }

        __device__ Vec3 operator-(Vec3 other_vec) {
            return Vec3(x - other_vec.x, y - other_vec.y, z - other_vec.z);
        }

        __device__ Vec3 operator-(float scalar) {
            return Vec3(x - scalar, y - scalar, z - scalar);
        }

        __device__ Vec3 operator*(float scalar) {
            return Vec3(x * scalar, y * scalar, z * scalar);
        }

        __device__ Vec3 operator*(Vec3 other_vec) {
            //element-wise multiplication
            return Vec3(x * other_vec.x, y * other_vec.y, z * other_vec.z);
        }

        __device__ Vec3 operator/(float scalar) {
            return Vec3(x / scalar, y / scalar, z / scalar);
        }

        __device__ float magnitude() {
            float mag_sq = x * x + y * y + z * z;
            return sqrt(mag_sq);
        }

        __device__ Vec3 normalised() {
            float mag = magnitude();
            Vec3 unit_vec(x / mag, y / mag, z / mag);

            return unit_vec;
        }

        __device__ float dot(Vec3 other_vec) {
            float new_x = x * other_vec.x;
            float new_y = y * other_vec.y;
            float new_z = z * other_vec.z;

            return new_x + new_y + new_z;
        }
};


__device__ struct RngData {
    int *seeds;
    int thread_index;
};


__device__ float get_random_num(RngData *data, int seed_inx) {
    //psuedorandom number generator
    curandState state;
    curand_init(data->seeds[seed_inx], data->thread_index, 0, &state);

    return curand_uniform(&state);
}