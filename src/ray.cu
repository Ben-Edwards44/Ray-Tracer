#include "camera.cu"


const float ANTIALIAS_OFFSET_RANGE = 0.001;


__host__ __device__ struct CamData {
    //stored data needed by the device (calculated by the host)
    Vec3 pos;
    Vec3 tl_position;

    float focal_length;

    Vec3 delta_u;
    Vec3 delta_v;

    int image_width;
    int image_height;
};


__device__ struct RayHitData {
    bool ray_hits;
    float ray_travelled_dist;
    Vec3 hit_point;
    Vec3 normal_vec;

    Vec2 texture_uv;
};


__device__ Vec3 lerp(Vec3 a, Vec3 b, float t) {
    return a + (b - a) * t;
}


__device__ class Ray {
    public:
        int pixel_x;
        int pixel_y;

        Vec3 origin;
        Vec3 direction;  //NOTE: this must be a normalised vector
        Vec3 direction_inv;

        uint *rng_state;

        bool antialias;

        __device__ Ray(int p_x, int p_y, uint *state, bool should_antialias) {
            pixel_x = p_x;
            pixel_y = p_y;

            rng_state = state;

            antialias = should_antialias;

            current_refractive_index = 1;  //air

            set_direction_origin();
        }

        __device__ Vec3 get_pos(float dist) {
            return direction * dist + origin;
        }

        __device__ void reflect(RayHitData *hit_data, Material obj_material) {
            //reflect ray after hitting an object
            Vec3 diffuse_dir = true_lambertian_reflect(hit_data);
            Vec3 specular_dir = perfect_reflect(hit_data);

            origin = hit_data->hit_point;

            change_direction(lerp(diffuse_dir, specular_dir, obj_material.smoothness).normalised());
        }

        __device__ void refract(RayHitData *hit_data, Material obj_material) {
            //refract the ray after hitting a glass material
            float n1;
            float n2;

            Vec3 reference_normal;  //normal in same direction as ray

            if (hit_data->normal_vec.dot(direction) > 0) {
                //moving outside the object
                n1 = obj_material.refractive_index;
                n2 = current_refractive_index;

                reference_normal = hit_data->normal_vec;
            } else {
                //moving inside the object
                n1 = current_refractive_index;
                n2 = obj_material.refractive_index;

                reference_normal = hit_data->normal_vec * -1;
            }

            current_refractive_index = n2;

            //snell's law (https://en.wikipedia.org/wiki/Snell%27s_law). NOTE: the min() is used to correct floating point errors
            float theta1 = acos(min(direction.dot(reference_normal), 1.0));
            float theta2 = asin(min(n1 * sin(theta1) / n2, 1.0));

            float ciritical_angle = asin(n2 / n1);

            float reflection_coeff = get_reflection_coeff(theta1, n1, n2);

            if (theta1 > ciritical_angle || reflection_coeff > pseudorandom_num(rng_state)) {
                //total internal reflection occurs or we are have a high reflectivity
                reflect(hit_data, obj_material);
                return;
            }

            //use the vector geometry explained here: https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel.html
            Vec3 perp_component;
            if (theta1 != 0) {
                perp_component = (direction - reference_normal * cos(theta1)) / sin(theta1);
            } else {
                //special case where the ray is incident normally
                perp_component = Vec3(0, 0, 0);
            }

            Vec3 resultant_refract = reference_normal * cos(theta2) + perp_component * sin(theta2);

            origin = hit_data->hit_point;

            change_direction(resultant_refract.normalised());
        }

        __device__ void apply_antialias() {
            if (!antialias) {return;}

            //shift the direction of the ray slightly to smooth edges
            Vec3 offset(0, 0, 0);
            offset.x = (pseudorandom_num(rng_state) - 0.5) * 2 * ANTIALIAS_OFFSET_RANGE;
            offset.y = (pseudorandom_num(rng_state) - 0.5) * 2 * ANTIALIAS_OFFSET_RANGE;
            offset.z = (pseudorandom_num(rng_state) - 0.5) * 2 * ANTIALIAS_OFFSET_RANGE;

            Vec3 new_dir = direction + offset;

            change_direction(new_dir.normalised());
        }
    
    private:
        float current_refractive_index;

        __device__ void set_direction_origin() {
            Vec3 view_pos = cam_pixel_to_world(pixel_x, pixel_y);
            Vec3 o = const_cam_data.cam_pos;
            Vec3 dir = view_pos - o;

            origin = o;

            change_direction(dir.normalised());
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

        __device__ float get_reflection_coeff(float theta1, float n1, float n2) {
            //at grazing angles, refractive substances reflect more. Use the Schlick approximation (https://en.wikipedia.org/wiki/Schlick%27s_approximation) to calculate how reflective we should be
            float sqrt_r0 = (n1 - n2) / (n1 + n2);
            float r0 = sqrt_r0 * sqrt_r0;

            float cos_theta = cos(theta1);  //TODO: we already did arccos(theta1) previously, so could save time

            return r0 + (1 - r0) * pow((1 - cos_theta), 5);  //in range [0, 1]
        }

        __device__ void change_direction(Vec3 new_dir) {
            //ensure we update the inverse direction as well
            direction = new_dir;
            direction_inv = Vec3(1, 1, 1) / new_dir;
        }
};