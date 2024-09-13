#include "meshes.cu"


__host__ __device__ struct RenderData {
    //data sent from host
    int rays_per_pixel;
    int reflection_limit;

    bool antialias;

    Vec3 sky_colour;
};


__constant__ RenderData const_render_data;


__device__ struct RayCollision {
    RayHitData hit_data;
    Material hit_mesh_material;
};


__device__ RayCollision get_ray_collision(Ray *ray) {
    RayCollision best_collision;

    //in the case where no collisions are found, the hit_data struct may have nonsense default values. So set sensible ones here.
    best_collision.hit_data.ray_hits = false;
    best_collision.hit_data.ray_travelled_dist = INF;

    for (int i = 0; i < const_meshes.num_meshes; i++) {
        RayHitData current_hit_data = const_meshes.meshes[i].hit(ray);

        if (!current_hit_data.ray_hits) {continue;}

        bool closest_to_cam = current_hit_data.ray_travelled_dist <= best_collision.hit_data.ray_travelled_dist;  //is this the closest to the camera so far?
        bool precision_error = -FLOAT_PRECISION_ERROR < current_hit_data.ray_travelled_dist < FLOAT_PRECISION_ERROR;  //floating point errors can cause a reflected ray to intersect with the same object twice (its origin is put just inside the object)
        
        if (closest_to_cam && !precision_error) {
            best_collision.hit_data = current_hit_data;
            best_collision.hit_mesh_material = const_meshes.meshes[i].material;
        }
    }

    return best_collision;
}


__device__ void update_ray(Ray *ray, RayHitData *hit_data, Material material) {
    //reflect or refract ray depending on the hit material type
    switch (material.type) {
        case Material::STANDARD:
            ray->reflect(hit_data, material);
            break;
        
        case Material::EMISSIVE:
            ray->reflect(hit_data, material);
            break;
        
        case Material::REFRACTIVE:
            ray->refract(hit_data, material);
            break;
    };
}


__device__ Vec3 trace_ray(Ray ray) {
    Vec3 final_colour(0, 0, 0);
    Vec3 current_ray_colour(1, 1, 1);

    for (int _ = 0; _ < const_render_data.reflection_limit; _++) {
        ray.apply_antialias();

        RayCollision collision = get_ray_collision(&ray);

        if (!collision.hit_data.ray_hits) {
            //ray has not hit anything - it has hit sky
            final_colour += const_render_data.sky_colour * current_ray_colour;
            break;
        }

        Material material = collision.hit_mesh_material;

        update_ray(&ray, &collision.hit_data, material);

        if (material.type == Material::EMISSIVE) {
            final_colour += material.emitted_light * current_ray_colour;
        } else {
            current_ray_colour *= material.texture.get_texture_colour(collision.hit_data.texture_uv);
        }
    }

    return final_colour;
}


__device__ Vec3 get_ray_colour(Vec3 previous_colour, Ray ray, int frame_num) {
    Vec3 colour(0, 0, 0);

    for (int _ = 0; _ < const_render_data.rays_per_pixel; _++) {
        Vec3 ray_colour = trace_ray(ray);  //passing by value copies the ray, so we can comfortably make changes to it
        colour += ray_colour;
    }

    colour /= const_render_data.rays_per_pixel;

    //use progressive rendering (take average of previous renders)
    Vec3 previous_sum = previous_colour * frame_num;
    
    return (colour + previous_sum) / (frame_num + 1);
}


__global__ void get_pixel_colour(float *pixel_array, float *previous_render, int *current_time, int *frame_num) {
    //TODO: the number of params in this function is simply obscene: use a struct to clean things up
    int pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (pixel_coord_x >= const_cam_data.image_width || pixel_coord_y >= const_cam_data.image_height) {return;}  //account for grid size being too big
    
    int array_index = (pixel_coord_y * const_cam_data.image_width + pixel_coord_x) * 3;  //multiply by 3 to account for each pixel having r, b, g values

    Vec3 previous_colour(previous_render[array_index], previous_render[array_index + 1], previous_render[array_index + 2]);

    uint rng_state = array_index * 3145739 + *current_time * 6291469;

    Ray ray(pixel_coord_x, pixel_coord_y, &rng_state, const_render_data.antialias);

    Vec3 colour = get_ray_colour(previous_colour, ray, *frame_num);

    pixel_array[array_index] = colour.x;
    pixel_array[array_index + 1] = colour.y;
    pixel_array[array_index + 2] = colour.z;
}