#include <cmath>


//for pseudorandom number generator (C++ MINSTD)
const int modulus = 1<<31 - 1;
const int multiplier = 48271;
const int increment = 0;


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

        __device__ Vec3 operator+(Vec3 *other_vec) {
            return Vec3(x + other_vec->x, y + other_vec->y, z + other_vec->z);
        }

        __device__ Vec3 operator-(Vec3 other_vec) {
            return Vec3(x - other_vec.x, y - other_vec.y, z - other_vec.z);
        }

        __device__ Vec3 operator-(Vec3 *other_vec) {
            return Vec3(x - other_vec->x, y - other_vec->y, z - other_vec->z);
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

        __device__ Vec3 operator*(Vec3 *other_vec) {
            //element-wise multiplication
            return Vec3(x * other_vec->x, y * other_vec->y, z * other_vec->z);
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

        __device__ float dot(Vec3 *other_vec) {
            float new_x = x * other_vec->x;
            float new_y = y * other_vec->y;
            float new_z = z * other_vec->z;

            return new_x + new_y + new_z;
        }
};


__device__ float pseudorandom_num(uint *state) {
    //PCG prng
    uint new_state = *state * 747796405 + 2891336453;
    *state = new_state;

	uint result = ((new_state >> ((new_state >> 28) + 4)) ^ new_state) * 277803737;
	result = (result >> 22) ^ result;

    float normalised = result / 4294967295.0;  //ensure between 0 and 1
	
    return normalised;
}


__device__ float normally_dist_num(uint *state) {
    //get a value in the standard normal distribution
    float theta = 2 * 3.14159 * pseudorandom_num(state);
    float rho = sqrt(-2 * log(pseudorandom_num(state)));
    return rho * cos(theta);
}