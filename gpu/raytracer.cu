#include "utils.cu"


const int INF = 1 << 31 - 1;
const float FLOAT_PRECISION_ERROR = 0.000001;

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


__host__ __device__ struct RenderData {
    //data sent from host
    int rays_per_pixel;
    int reflection_limit;

    int frame_num;

    bool antialias;

    Vec3 sky_colour;
};


__host__ __device__ struct Material {
    Vec3 colour;
    float emission_strength;
    Vec3 emission_colour;
    int mat_type;

    //optional
    float fuzz_level;
};


__device__ struct RayHitData {
    bool ray_hits;
    float ray_travelled_dist;
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

        bool antialias;

        __device__ Ray(int p_x, int p_y, CamData *camera_data, uint *state, bool should_antialias) {
            pixel_x = p_x;
            pixel_y = p_y;

            rng_state = state;

            antialias = should_antialias;

            set_direction_origin(camera_data);
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


struct Plane {
    //ax + by + cz + d = 0
    float a;
    float b;
    float c;
    float d;
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

        __device__ RayHitData hit(Ray *ray) {
            //ray-sphere intersection results in quadratic equation t^2(d⋅d)−2td⋅(C−Q)+(C−Q)⋅(C−Q)−r^2=0
            //so we solve with quadratic formula
            Vec3 c_min_q = center - ray->origin;

            float a = ray->direction.dot(ray->direction);
            float b = ray->direction.dot(c_min_q) * (-2);
            float c = c_min_q.dot(c_min_q) - radius * radius;

            float discriminant = b * b - 4 * a * c;

            RayHitData hit_data;
            bool valid_hit = false;

            if (discriminant >= 0) {
                float ray_dist = (-b - sqrt(discriminant)) / (2 * a);  //negative solution to equation

                //only render spheres in front of ray
                if (ray_dist > FLOAT_PRECISION_ERROR) {
                    valid_hit = true;

                    Vec3 hit_point = ray->get_pos(ray_dist);

                    hit_data.ray_hits = true;
                    hit_data.ray_travelled_dist = ray_dist;
                    hit_data.hit_point = hit_point;
                    hit_data.normal_vec = (hit_point - center).normalised();  //vector pointing from center to point of intersection
                }
            }

            if (!valid_hit) {
                //set the correct default values
                hit_data.ray_hits = false;
                hit_data.ray_travelled_dist = INF;
            }

            return hit_data;
        }
};


__host__ __device__ class Triangle {
    public:
        Material material;
        Vec3 normal_vec;

        Triangle() {}

        __host__ Triangle(Vec3 point1, Vec3 point2, Vec3 point3, Material mat) {
            points[0] = point1;
            points[1] = point2;
            points[2] = point3;

            material = mat;

            precompute();
        }

        __device__ RayHitData hit(Ray *ray) {
            Vec3 vecs_to_corner[3];

            for (int i = 0; i < 3; i++) {
                vecs_to_corner[i] = points[i] - ray->origin;
            }

            int num_outside = 0;
            for (int i = 0; i < 3; i++) {
                Vec3 normal = vecs_to_corner[i].cross(vecs_to_corner[(i + 1) % 3]);
                float dot_prod = normal.dot(ray->direction);

                if (dot_prod < 0) {num_outside++;}  //ray points outside of the triangle
            }

            //depending on the order of the points, the normals to the sides always point in or alyawys point out. Therefore, a ray interects <=> it is always within the vecs or always outside the vecs
            bool ray_hits = num_outside == 0 || num_outside == 3;

            RayHitData hit_data;
            bool valid_hit = false;

            if (ray_hits) {
                float ray_dist = get_ray_travelled_dist(ray);

                //only render triangles in front of ray
                if (ray_dist > FLOAT_PRECISION_ERROR) {
                    valid_hit = true;

                    hit_data.ray_hits = true;
                    hit_data.ray_travelled_dist = ray_dist;
                    hit_data.hit_point = ray->get_pos(hit_data.ray_travelled_dist);

                    Vec3 normal = normal_vec;
                    if (normal.dot(ray->direction) > 0) {normal *= -1;}  //normal should point in same direction as the ray

                    hit_data.normal_vec = normal;
                }
            }

            if (!valid_hit) {
                //set correct default values
                hit_data.ray_hits = false;
                hit_data.ray_travelled_dist = INF;
            }

            return hit_data;
        }

        private:
            Plane plane;

            Vec3 points[3];

            __host__ void precompute() {
                //precompute the plane the mesh lies on and (one of) its normal vectors. https://math.stackexchange.com/questions/2686606/equation-of-a-plane-passing-through-3-points
                Vec3 side1 = points[0] - points[1];
                Vec3 side2 = points[1] - points[2];

                normal_vec = side1.cross(side2).normalised();

                plane = {normal_vec.x, normal_vec.y, normal_vec.z};
                plane.d = -(plane.a * points[0].x + plane.b * points[0].y + plane.c * points[0].z);  //sub in a point to find constant
            }

            __device__ float get_ray_travelled_dist(Ray *ray) {
                //this algebra was worked out on paper (it is just the interesction between a line and plane really)
                float numerator = plane.d + plane.a * ray->origin.x + plane.b * ray->origin.y + plane.c * ray->origin.z;
                float denominator = plane.a * ray->direction.x + plane.b * ray->direction.y + plane.c * ray->direction.z;

                return -numerator / denominator;
            }
};


__host__ __device__ class Quad {
    public:
        Material material;

        Quad() {}

        __host__ Quad(Vec3 point1, Vec3 point2, Vec3 point3, Vec3 point4, Material mat) {
            material = mat;

            create_triangles(point1, point2, point3, point4);
        }

        __device__ RayHitData hit(Ray *ray) {
            RayHitData t1_hit = t1.hit(ray);
            RayHitData t2_hit = t2.hit(ray);

            //return whichever triangle hits (or just the 2nd one if none of them hit)
            if (t1_hit.ray_hits) {
                return t1_hit;
            } else {
                return t2_hit;
            }
        }

    protected:  //NOTE: protected is used instead of private so these can be inherited by OneWayQuad
        Triangle t1;
        Triangle t2;

        void create_triangles(Vec3 point1, Vec3 point2, Vec3 point3, Vec3 point4) {
            //a quad is just 2 triangles
            t1 = Triangle(point1, point2, point3, material);
            t2 = Triangle(point1, point4, point3, material);
        }
};


__host__ __device__ class OneWayQuad : public Quad {
    public:
        __host__ OneWayQuad(Vec3 point1, Vec3 point2, Vec3 point3, Vec3 point4, Material mat, bool invert_normal) {
            material = mat;

            create_triangles(point1, point2, point3, point4);
            get_normal_vec(invert_normal);
        }

        __device__ RayHitData hit(Ray *ray) {
            if (ray->direction.dot(normal_vec) < 0) {
                //as this is one way, rays travelling in opposite direction to the normal vector should not be counted as hits
                return RayHitData{false, INF};
            } else {
                return Quad::hit(ray);
            }
        }

    private:
        Vec3 normal_vec;

        __host__ void get_normal_vec(bool invert_normal) {
            //branchless (because its cooler) method of multiplying the normal by -1 <=> invert_normal is true
            int multiplier = 1 - 2 * invert_normal;
            normal_vec = t1.normal_vec * multiplier;
        }
};


__device__ struct AllMeshes {
    Sphere *spheres;
    Triangle *triangles;
    Quad *quads;
    OneWayQuad *one_way_quads;

    int num_spheres;
    int num_triangles;
    int num_quads;
    int num_one_way_quads;
};


__device__ struct RayCollision {
    RayHitData hit_data;
    Material hit_mesh_material;
};


template <typename T>
__device__ RayCollision get_specific_mesh_collision(Ray *ray, T *meshes, int num_meshes) {
    //get the closest collision with a specific mesh (e.g. sphere, triangle). NOTE: error occurs if there are no meshes
    RayHitData hit_data;
    Material hit_mesh_material;

    //in the case where no collisions are found, the hit_data struct may have nonsense default values. So we set sensible ones here
    hit_data.ray_hits = false;
    hit_data.ray_travelled_dist = INF;

    for (int i = 0; i < num_meshes; i++) {
        RayHitData current_hit = meshes[i].hit(ray);

        if (!current_hit.ray_hits) {continue;}

        bool closest_to_cam = current_hit.ray_travelled_dist <= hit_data.ray_travelled_dist;  //is this the closest to the camera so far?
        bool precision_error = -FLOAT_PRECISION_ERROR < current_hit.ray_travelled_dist < FLOAT_PRECISION_ERROR;  //floating point errors can cause a reflected ray to intersect with the same object twice (its origin is put just inside the object)
        
        if (closest_to_cam && !precision_error) {
            hit_data = current_hit;
            hit_mesh_material = meshes[i].material;
        }
    }

    return RayCollision{hit_data, hit_mesh_material};
}


__device__ RayCollision get_ray_collision(Ray *ray, AllMeshes *meshes) {
    RayCollision sphere_collision = get_specific_mesh_collision<Sphere>(ray, meshes->spheres, meshes->num_spheres);
    RayCollision triangle_collision = get_specific_mesh_collision<Triangle>(ray, meshes->triangles, meshes->num_triangles);
    RayCollision quad_collision = get_specific_mesh_collision<Quad>(ray, meshes->quads, meshes->num_quads);
    RayCollision one_way_quad_collision = get_specific_mesh_collision<OneWayQuad>(ray, meshes->one_way_quads, meshes->num_one_way_quads);

    RayCollision closest_collision = sphere_collision;

    if (triangle_collision.hit_data.ray_hits && triangle_collision.hit_data.ray_travelled_dist < closest_collision.hit_data.ray_travelled_dist) {
        //triangle is actually better collision
        closest_collision = triangle_collision;
    }
    if (quad_collision.hit_data.ray_hits && quad_collision.hit_data.ray_travelled_dist < closest_collision.hit_data.ray_travelled_dist) {
        //quad is actually better collision
        closest_collision = quad_collision;
    }
    if (one_way_quad_collision.hit_data.ray_hits && one_way_quad_collision.hit_data.ray_travelled_dist < closest_collision.hit_data.ray_travelled_dist) {
        //one way quad is actually better collision
        closest_collision = one_way_quad_collision;
    }

    return closest_collision;
}


__device__ Vec3 trace_ray(Ray ray, AllMeshes *meshes, RenderData *render_data) {
    Vec3 final_colour(0, 0, 0);
    Vec3 current_ray_colour(1, 1, 1);

    for (int _ = 0; _ < render_data->reflection_limit; _++) {
        ray.apply_antialias();

        RayCollision collision = get_ray_collision(&ray, meshes);

        if (!collision.hit_data.ray_hits) {
            //ray has not hit anything - it has hit sky
            final_colour += render_data->sky_colour * current_ray_colour;
            break;
        }

        ray.reflect(&collision.hit_data, collision.hit_mesh_material);

        Material material = collision.hit_mesh_material;
        Vec3 mat_emitted_light = material.emission_colour * material.emission_strength;  //TODO: precalculate

        final_colour += mat_emitted_light * current_ray_colour;
        current_ray_colour *= material.colour;
    }

    return final_colour;
}


__device__ Vec3 get_ray_colour(Vec3 previous_colour, Ray ray, AllMeshes *meshes, RenderData *render_data) {
    Vec3 colour(0, 0, 0);

    for (int _ = 0; _ < render_data->rays_per_pixel; _++) {
        Vec3 ray_colour = trace_ray(ray, meshes, render_data);  //passing by value copies the ray, so we can comfortably make changes to it
        colour += ray_colour;
    }

    colour /= render_data->rays_per_pixel;

    //use progressive rendering (take average of previous renders)
    Vec3 previous_sum = previous_colour * render_data->frame_num;
    
    return (colour + previous_sum) / (render_data->frame_num + 1);
}


__global__ void get_pixel_colour(float *pixel_array, float *previous_render, CamData *camera_data, AllMeshes *mesh_data, RenderData *render_data, int *current_time) {
    //TODO: the number of params in this function is simply obscene: use a struct to clean things up
    int pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (pixel_coord_x >= camera_data->image_width || pixel_coord_y >= camera_data->image_height) {return;}  //account for grid size being too big
    
    int array_index = (pixel_coord_y * camera_data->image_width + pixel_coord_x) * 3;  //multiply by 3 to account for each pixel having r, b, g values

    Vec3 previous_colour(previous_render[array_index], previous_render[array_index + 1], previous_render[array_index + 2]);

    uint rng_state = array_index * 3145739 + *current_time * 6291469;

    Ray ray(pixel_coord_x, pixel_coord_y, camera_data, &rng_state, render_data->antialias);

    Vec3 colour = get_ray_colour(previous_colour, ray, mesh_data, render_data);

    pixel_array[array_index] = colour.x;
    pixel_array[array_index + 1] = colour.y;
    pixel_array[array_index + 2] = colour.z;
}