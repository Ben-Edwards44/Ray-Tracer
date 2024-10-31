#include <cmath>
#include <stdexcept>


__host__ void check_cuda_error(cudaError_t error, std::string msg) {
    if (error != cudaSuccess) {
        std::string err_msg = cudaGetErrorString(error);
        throw std::runtime_error("Error from CUDA (" + msg + "): " + err_msg);
    }
}


__host__ __device__ class Vec3 {
    public:
        float x;
        float y;
        float z;

        __host__ __device__ Vec3() {}

        __host__ __device__ Vec3(float val_x, float val_y, float val_z) {
            x = val_x;
            y = val_y;
            z = val_z;
        }

        __host__ __device__ Vec3(float3 vector) {
            x = vector.x;
            y = vector.y;
            z = vector.z;
        }

        //common operations
        __host__ __device__ Vec3 operator+(Vec3 other_vec) {
            return Vec3(x + other_vec.x, y + other_vec.y, z + other_vec.z);
        }

        __device__ Vec3 operator+(float scalar) {
            return Vec3(x + scalar, y + scalar, z + scalar);
        }

        __device__ Vec3 operator+(Vec3 *other_vec) {
            return Vec3(x + other_vec->x, y + other_vec->y, z + other_vec->z);
        }

        __device__ Vec3* operator+=(Vec3 other_vec) {
            x += other_vec.x;
            y += other_vec.y;
            z += other_vec.z;

            return this;
        }

        __host__ __device__ Vec3 operator-(Vec3 other_vec) {
            return Vec3(x - other_vec.x, y - other_vec.y, z - other_vec.z);
        }

        __device__ Vec3 operator-(Vec3 *other_vec) {
            return Vec3(x - other_vec->x, y - other_vec->y, z - other_vec->z);
        }

        __device__ Vec3 operator-(float scalar) {
            return Vec3(x - scalar, y - scalar, z - scalar);
        }

        __host__ __device__ Vec3 operator*(float scalar) {
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

        __device__ Vec3* operator*=(Vec3 other_vec) {
            //element-wise multiplication
            x *= other_vec.x;
            y *= other_vec.y;
            z *= other_vec.z;

            return this;
        }

        __device__ Vec3* operator*=(float scalar) {
            //element-wise multiplication
            x *= scalar;
            y *= scalar;
            z *= scalar;

            return this;
        }

        __host__ __device__ Vec3 operator/(float scalar) {
            return Vec3(x / scalar, y / scalar, z / scalar);
        }

        __device__ Vec3* operator/=(float scalar) {
            x /= scalar;
            y /= scalar;
            z /= scalar;

            return this;
        }

        __device__ bool operator==(Vec3 other_vec) {
            return x == other_vec.x && y == other_vec.y && z == other_vec.z;
        }

        __host__ __device__ float magnitude() {
            float mag_sq = x * x + y * y + z * z;
            return sqrt(mag_sq);
        }

        __host__ __device__ Vec3 normalised() {
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

        __host__ __device__ Vec3 cross(Vec3 other_vec) {
            //https://en.wikipedia.org/wiki/Cross_product
            float s1 = y * other_vec.z - z * other_vec.y;
            float s2 = z * other_vec.x - x * other_vec.z;
            float s3 = x * other_vec.y - y * other_vec.x;

            return Vec3(s1, s2, s3);
        }

        __host__ void set_mag(float desired_mag) {
            //set the vectors magnitude to the desired_mag
            float scale = desired_mag / magnitude();

            x *= scale;
            y *= scale;
            z *= scale;
        }
};


__host__ __device__ class Vec2 {
    public:
        float x;
        float y;

        __device__ Vec2() {}

        __host__ __device__ Vec2(float x_val, float y_val) {
            x = x_val;
            y = y_val;
        }

        __device__ Vec2 operator+(Vec2 other_vec) {
            return Vec2(x + other_vec.x, y + other_vec.y);
        }

        __device__ Vec2 operator*(float scalar) {
            return Vec2(x * scalar, y * scalar);
        }
};


template <typename T>
__host__ __device__ class DeviceStack {
    public:
        __host__ DeviceStack() {}

        __device__ void push(T item) {
            top++;
            items[top] = item;
        }

        __device__ T pop() {
            T item = items[top];
            top--;

            return item;
        }

        __device__ void empty() {
            top = -1;
        }

        __device__ bool is_empty() {
            return top == -1;
        } 

        __host__ void allocate_mem(int max_size) {
            int mem_size = max_size * sizeof(T);

            //allocate the memory
            cudaError_t error = cudaMalloc((void **)&items, mem_size);
            check_cuda_error(error, "allocating stack memory");
        }

    private:
        int top;
        T *items;
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