#include "utils.cu"


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


__host__ __device__ struct RenderData {
    //data sent from host
    int rays_per_pixel;
    int reflection_limit;

    int num_spheres;

    int frame_num;

    bool static_scene;
};


__device__ struct RayHitData {
    bool ray_hits = false;
    float ray_travelled_dist = INFINITY;
    Vec3 hit_point;
    Vec3 normal_vec;
};


__device__ class Ray {
    public:
        int pixel_x;
        int pixel_y;

        Vec3 origin;
        Vec3 direction;

        uint *rng_state;

        __device__ Ray(int p_x, int p_y, CamData *camera_data, uint *state) {
            pixel_x = p_x;
            pixel_y = p_y;

            rng_state = state;

            set_direction_origin(camera_data);
        }

        __device__ Vec3 get_pos(float dist) {
            return direction * dist + origin;
        }

        __device__ void diffuse_reflect(RayHitData *hit_data) {
            //diffuse reflect after hitting something
            float dir_x = normally_dist_num(rng_state);
            float dir_y = normally_dist_num(rng_state);
            float dir_z = normally_dist_num(rng_state);

            Vec3 new_dir(dir_x, dir_y, dir_z);

            if (new_dir.dot(hit_data->normal_vec) < 0) {
                new_dir = new_dir * -1;  //invert since we are reflecting inside the sphere
            }

            //assign the new values
            origin = hit_data->hit_point;
            direction = new_dir.normalised();
        }

        //__device__ void perfect_reflect(RayHitData *hit_data, int ray)

    private:
        __device__ Vec3 screen_to_world(int x, int y, CamData *camera_data) {
            //convert a point (x, y) on the viewport projection plane into a world space coordinate
            Vec3 local_pos;

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


__host__ __device__ struct Material {
    Vec3 colour;
    float emission_strength;
    Vec3 emission_colour;
};


__host__ __device__ class Sphere {
    public:
        Vec3 center;
        float radius;

        Material material;

        __host__ Sphere(Vec3 cent, float r, Material mat) {
            center = cent;
            radius = r;
            material = mat;
        };

        __device__ Sphere() {};

        __device__ RayHitData hit(Ray *ray) {
            //ray-sphere intersection results in quadratic equation t^2(d⋅d)−2td⋅(C−Q)+(C−Q)⋅(C−Q)−r^2=0
            //so we solve with quadratic formula
            Vec3 c_min_q = center - ray->origin;

            float a = ray->direction.dot(ray->direction);
            float b = ray->direction.dot(c_min_q) * (-2);
            float c = c_min_q.dot(c_min_q) - radius * radius;

            float discriminant = b * b - 4 * a * c;

            RayHitData hit_data;
            if (discriminant >= 0) {
                float ray_dist = (-b - sqrt(discriminant)) / (2 * a);  //negative solution to equation

                //only render spheres in front of camera
                if (ray_dist >= 0) {
                    Vec3 hit_point = ray->get_pos(ray_dist);

                    hit_data.ray_hits = true;
                    hit_data.ray_travelled_dist = ray_dist;
                    hit_data.hit_point = hit_point;
                    hit_data.normal_vec = (hit_point - center).normalised();  //vector pointing from center to point of intersection
                }
            }

            return hit_data;
        }
};


__device__ struct RayCollision {
    RayHitData *hit_data;
    Sphere *hit_sphere;
};


__device__ RayCollision get_ray_collision(Ray *ray, Sphere *mesh_data, int num_spheres) {
    RayHitData hit_data;
    Sphere hit_sphere;

    for (int i = 0; i < num_spheres; i++) {
        RayHitData current_hit = mesh_data[i].hit(ray);
        
        //check if this sphere is closest to camera
        if (current_hit.ray_travelled_dist <= hit_data.ray_travelled_dist) {
            hit_data = current_hit;
            hit_sphere = mesh_data[i];
        }
    }

    return RayCollision{&hit_data, &hit_sphere};
}


__device__ Vec3 trace_ray(Ray *ray, Sphere *mesh_data, RenderData *render_data) {
    Vec3 final_colour(0, 0, 0);
    Vec3 current_ray_colour(1, 1, 1);

    for (int _ = 0; _ < render_data->reflection_limit; _++) {
        RayCollision collision = get_ray_collision(ray, mesh_data, render_data->num_spheres);

        if (!collision.hit_data->ray_hits) {break;}  //ray has not hit anything

        ray->diffuse_reflect(collision.hit_data);

        Material material = collision.hit_sphere->material;
        Vec3 mat_emitted_light = material.emission_colour * material.emission_strength;  //TODO: precalculate

        final_colour = final_colour + mat_emitted_light * current_ray_colour;
        current_ray_colour = current_ray_colour * material.colour;
    }

    return final_colour;
}


__device__ Vec3 get_ray_colour(Vec3 previous_colour, Ray ray, Sphere *mesh_data, RenderData *render_data) {
    //check sphere intersection
    Vec3 colour(0, 0, 0);

    for (int _ = 0; _ < render_data->rays_per_pixel; _++) {
        Ray ray_copy = ray;
        Vec3 ray_colour = trace_ray(&ray_copy, mesh_data, render_data);
        colour = colour + ray_colour;
    }

    colour = colour / render_data->rays_per_pixel;

    if (render_data->static_scene && render_data->frame_num > 0) {
        //use progressive rendering (take average of previous renders)
        Vec3 previous_sum = previous_colour * render_data->frame_num;
        return (colour + previous_sum) / (render_data->frame_num + 1);
    } else {
        return colour;
    }
}


__global__ void get_pixel_colour(float *pixel_array, float *previous_render, CamData *camera_data, Sphere *mesh_data, RenderData *render_data, int *current_time) {
    //TODO: the number of params in this function is simply obscene: use a struct to clean things up
    
    int pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (pixel_coord_x >= camera_data->image_width || pixel_coord_y >= camera_data->image_height) {return;}  //account for grid size being too big
    
    int array_index = (pixel_coord_y * camera_data->image_width + pixel_coord_x) * 3;  //multiply by 3 to account for each pixel having r, b, g values

    Vec3 previous_colour(previous_render[array_index], previous_render[array_index + 1], previous_render[array_index + 2]);

    uint rng_state = array_index * 3145739 + *current_time * 6291469;

    Ray ray(pixel_coord_x, pixel_coord_y, camera_data, &rng_state);

    Vec3 colour = get_ray_colour(previous_colour, ray, mesh_data, render_data);

    pixel_array[array_index] = colour.x;
    pixel_array[array_index + 1] = colour.y;
    pixel_array[array_index + 2] = colour.z;
}