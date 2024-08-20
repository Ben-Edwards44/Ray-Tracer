#include <cmath>
#include "ray.cu"


const int INF = 1 << 31 - 1;
const float FLOAT_PRECISION_ERROR = 0.000001;

const float PI = 3.141592653589793;


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

                    if (material.using_texture) {assign_texture_coords(&hit_data);}
                }
            }

            if (!valid_hit) {
                //set the correct default values
                hit_data.ray_hits = false;
                hit_data.ray_travelled_dist = INF;
            }

            return hit_data;
        }

    private:
        __device__ void assign_texture_coords(RayHitData *hit_data) {
            //assign (u, v) texture coords (u based on latitude, v based on longitude)
            float theta = asin((hit_data->hit_point.y - center.y) / radius);  //angle above center (on y axis)
            float phi = acos((hit_data->hit_point.x - center.x) / radius);  //angle from center on x axis

            float u = (theta + PI / 2) / PI;  //make latitude in range [0, 1]

            float v_ratio = (1 - phi / PI) / 2;  //0 at left end, 0.5 at right end

            bool behind = hit_data->hit_point.z > center.z;
            int mult = 1 - 2 * behind;

            float v = 1 * behind + mult * v_ratio;  //branchless method of making v loop 0 -> 0.5 when in front, then 0.5 -> 1 when behind

            hit_data->u = u;
            hit_data->v = v;
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
                return RayHitData{false, static_cast<float>(INF)};
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


__host__ __device__ class Cuboid {
    public:
        Material material;

        __host__ Cuboid(Vec3 tl_near_pos, float width, float height, float depth, Material mat) {
            material = mat;

            create_faces(tl_near_pos, width, height, depth);
        }

        __device__ RayHitData hit(Ray *ray) {
            RayHitData hit_data;

            //default values
            hit_data.ray_hits = false;
            hit_data.ray_travelled_dist = INF;

            for (int i = 0; i < 6; i++) {
                RayHitData face_hit = faces[i].hit(ray);

                if (face_hit.ray_hits && face_hit.ray_travelled_dist < hit_data.ray_travelled_dist) {
                    //this face is closest to camera
                    hit_data = face_hit;
                }
            }

            return hit_data;
        }

    private:
        Quad faces[6];

        __host__ void create_faces(Vec3 tl_near, float width, float height, float depth) {
            //create vectors so we can use vector arithmetic
            Vec3 w(width, 0, 0);
            Vec3 h(0, height, 0);
            Vec3 d(0, 0, depth);

            //get the corners of the cuboid
            Vec3 tr_near = tl_near + w;
            Vec3 br_near = tr_near - h;
            Vec3 bl_near = tl_near - h;
            Vec3 tl_far = tl_near + d;
            Vec3 tr_far = tl_far + w;
            Vec3 br_far = tr_far - h;
            Vec3 bl_far = tl_far - h;

            //assign the faces
            faces[0] = Quad(tl_near, tr_near, br_near, bl_near, material);  //front face
            faces[1] = Quad(tl_far, tr_far, br_far, bl_far, material);  //back face
            faces[2] = Quad(tl_near, bl_near, bl_far, tl_far, material);  //left face
            faces[3] = Quad(tr_near, br_near, br_far, tr_far, material);  //right face
            faces[4] = Quad(bl_near, br_near, br_far, bl_far, material);  //bottom face
            faces[5] = Quad(tl_near, tr_near, tr_far, tl_far, material);  //top face
        }
};


__device__ struct AllMeshes {
    Sphere *spheres;
    Triangle *triangles;
    Quad *quads;
    OneWayQuad *one_way_quads;
    Cuboid *cuboids;

    int num_spheres;
    int num_triangles;
    int num_quads;
    int num_one_way_quads;
    int num_cuboids;
};


__constant__ AllMeshes const_all_meshes;