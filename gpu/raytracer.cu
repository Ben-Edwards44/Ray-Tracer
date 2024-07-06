#include <cmath>


__device__ float3 operator+(float3 &a, float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}


__device__ float3 operator-(float3 &a, float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}


__global__ struct CamData {
    //stored data needed by the device (calculated by the host)
    float3 pos;
    float3 tl_position;

    float focal_length;

    float delta_u;
    float delta_v;

    int image_width;
    int image_height;
};


__device__ class Vec3 {
    public:
        float x;
        float y;
        float z;

        __device__ Vec3(float val_x, float val_y, float val_z) {
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

        __device__ Vec3 operator+(Vec3 other_vec) {
            return Vec3(x + other_vec.x, y + other_vec.y, z + other_vec.z);
        }

        __device__ Vec3 operator-(Vec3 other_vec) {
            return Vec3(x - other_vec.x, y - other_vec.y, z - other_vec.z);
        }

        __device__ Vec3 operator*(float scalar) {
            return Vec3(x * scalar, y * scalar, z * scalar);
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


__device__ class Ray {
    public:
        int pixel_x;
        int pixel_y;

        Vec3 origin;
        Vec3 direction;

        __device__ Ray(int p_x, int p_y, CamData *camera_data) {
            pixel_x = p_x;
            pixel_y = p_y;

            set_direction_origin(camera_data);
        }

        __device__ Vec3 get_pos(float time) {
            return origin + direction * time;
        }

    private:
        __device__ Vec3 screen_to_world(int x, int y, CamData *camera_data) {
            //convert a point (x, y) on the viewport projection plane into a world space coordinate
            float3 local_pos;

            local_pos.x = x * camera_data->delta_u;
            local_pos.y = -y * camera_data->delta_v;
            local_pos.z = 0;

            return Vec3(camera_data->tl_position + local_pos);
        }

        __device__ void set_direction_origin(CamData *camera_data) {
            Vec3 view_pos = screen_to_world(pixel_x, pixel_y, camera_data);
            Vec3 o = camera_data->pos;
            Vec3 dir = view_pos - o;

            origin = o;
            direction = dir.normalised();
        }
};


__device__ Vec3 get_ray_colour(Ray *ray) {
    //check sphere intersection
    Vec3 center(0, 0, 2);
    float r = 1;

    Vec3 c_min_q = center - ray->origin;

    float a = ray->direction.dot(ray->direction);
    float b = (ray->direction * (-2)).dot(c_min_q);
    float c = c_min_q.dot(c_min_q) - r * r;

    bool intersects = b * b - 4 * a * c >= 0;

    if (intersects) {
        return Vec3(1, 1, 1);
    } else {
        return Vec3(0, 0, 0);
    }
}


__global__ void get_pixel_colour(float *pixel_array, CamData *camera_data) {
    int pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (pixel_coord_x >= camera_data->image_width || pixel_coord_y >= camera_data->image_height) {return;}  //account for grid size being too big
    
    int array_index = (pixel_coord_y * camera_data->image_width + pixel_coord_x) * 3;  //multiply by 3 to account for each pixel having r, b, g values
    
    Ray ray(pixel_coord_x, pixel_coord_y, camera_data);

    Vec3 colour = get_ray_colour(&ray);

    pixel_array[array_index] = colour.x;
    pixel_array[array_index + 1] = colour.y;
    pixel_array[array_index + 2] = colour.z;
}