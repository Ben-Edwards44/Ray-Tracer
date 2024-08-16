#include "utils.cu"


const float ANTIALIAS_OFFSET_RANGE = 0.0015;


__host__ __device__ struct CamData {
    //stored data needed by the device (calculated by the host)
    Vec3 pos;
    Vec3 tl_position;

    float focal_length;

    float delta_u;
    float delta_v;

    int image_width;
    int image_height;
};


__constant__ CamData const_cam_data;


__device__ struct RayHitData {
    bool ray_hits;
    float ray_travelled_dist;
    Vec3 hit_point;
    Vec3 normal_vec;
};


__host__ __device__ struct Material {
    Vec3 colour;
    float emission_strength;
    Vec3 emission_colour;
    int mat_type;

    //optional
    float fuzz_level;
};


__device__ class Ray {
    public:
        int pixel_x;
        int pixel_y;

        Vec3 origin;
        Vec3 direction;

        uint *rng_state;

        bool antialias;

        __device__ Ray(int p_x, int p_y, uint *state, bool should_antialias) {
            pixel_x = p_x;
            pixel_y = p_y;

            rng_state = state;

            antialias = should_antialias;

            set_direction_origin();
        }

        __device__ Vec3 get_pos(float dist) {
            return direction * dist + origin;
        }

        __device__ void reflect(RayHitData *hit_data, Material obj_material) {
            //reflect ray after hitting an object

            Vec3 new_direction;
            if (obj_material.mat_type == 0) {
                new_direction = true_lambertian_reflect(hit_data);
            } else if (obj_material.mat_type == 1) {
                new_direction = perfect_reflect(hit_data);
            }
            else {
                new_direction = fuzzy_perfect_reflect(hit_data, obj_material);
            }

            direction = new_direction;
            origin = hit_data->hit_point;
        }

        __device__ void apply_antialias() {
            if (!antialias) {return;}

            //shift the direction of the ray slightly to smooth edges
            float x_offset = (pseudorandom_num(rng_state) - 0.5) * 2 * ANTIALIAS_OFFSET_RANGE;
            float y_offset = (pseudorandom_num(rng_state) - 0.5) * 2 * ANTIALIAS_OFFSET_RANGE;
            float z_offset = (pseudorandom_num(rng_state) - 0.5) * 2 * ANTIALIAS_OFFSET_RANGE;

            direction.x += x_offset;
            direction.y += y_offset;
            direction.z += z_offset;
        }
    
    private:
        __device__ Vec3 screen_to_world(int x, int y) {
            //convert a point (x, y) on the viewport projection plane into a world space coordinate
            Vec3 local_pos;

            local_pos.x = x * const_cam_data.delta_u;
            local_pos.y = -y * const_cam_data.delta_v;
            local_pos.z = 0;

            return Vec3(const_cam_data.tl_position + local_pos);
        }

        __device__ void set_direction_origin() {
            Vec3 view_pos = screen_to_world(pixel_x, pixel_y);
            Vec3 o = const_cam_data.pos;
            Vec3 dir = view_pos - o;

            origin = o;
            direction = dir.normalised();
        }

        __device__ Vec3 diffuse_reflect(RayHitData *hit_data) {
            //diffuse reflect after hitting something (just choose a random direction)
            float dir_x = normally_dist_num(rng_state);
            float dir_y = normally_dist_num(rng_state);
            float dir_z = normally_dist_num(rng_state);

            Vec3 rand_vec(dir_x, dir_y, dir_z);

            if (rand_vec.dot(hit_data->normal_vec) < 0) {
                rand_vec *= -1;  //invert since we want a vector that points outwards
            }

            return rand_vec.normalised();
        }

        __device__ Vec3 true_lambertian_reflect(RayHitData *hit_data) {
            //reflected vector proportional to cos of the angle
            Vec3 rand_offset_vec = diffuse_reflect(hit_data);
            Vec3 new_dir = hit_data->normal_vec + rand_offset_vec;

            return new_dir.normalised();
        }

        __device__ Vec3 perfect_reflect(RayHitData *hit_data) {
            //angle incidence = angle reflection: r=d−2(d⋅n)n (where d is incoming vector, n is normal and r in reflected)
            float dot = direction.dot(hit_data->normal_vec);
            Vec3 reflected_vec = direction - hit_data->normal_vec * 2 * dot;

            return reflected_vec.normalised();
        }

        __device__ Vec3 fuzzy_perfect_reflect(RayHitData *hit_data, Material obj_material) {
            //angle reflection = angle incidence + some noise
            Vec3 reflected_vec = perfect_reflect(hit_data);
            Vec3 rand_offset_vec = diffuse_reflect(hit_data) * obj_material.fuzz_level;

            Vec3 new_dir = reflected_vec + rand_offset_vec;

            return new_dir.normalised();
        }
};