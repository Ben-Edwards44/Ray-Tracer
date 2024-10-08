#include <cmath>
#include "ray.cu"


const int INF = 1 << 31 - 1;
const float FLOAT_PRECISION_ERROR = 0.000001;


struct Plane {
    //ax + by + cz + d = 0
    float a;
    float b;
    float c;
    float d;
};


struct Vertex {
    Vec3 world_point;
    Vec2 texture_point;
};


__host__ __device__ class Sphere {
    public:
        Vec3 center;
        float radius;

        Material material;

        __host__ Sphere() {}

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

                    if (material.need_uv) {assign_texture_coords(&hit_data);}
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

            hit_data->texture_uv = Vec2(u, v);
        }
};


__host__ __device__ class Triangle {
    public:
        Vec3 points[3];

        Material material;
        Vec3 normal_vec;

        __host__ Triangle() {}

        __host__ Triangle(Vec3 point1, Vec3 point2, Vec3 point3, Material mat) {
            points[0] = point1;
            points[1] = point2;
            points[2] = point3;

            material = mat;

            precompute();
        }

        __host__ Triangle(Vertex point1, Vertex point2, Vertex point3, Material mat) {
            texture_points[0] = point1.texture_point;
            texture_points[1] = point2.texture_point;
            texture_points[2] = point3.texture_point;

            //in c++, you cannot call the constructor directly (why??) so we just need to CTRL-C CTRL-V the code
            points[0] = point1.world_point;
            points[1] = point2.world_point;
            points[2] = point3.world_point;

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

                    if (material.need_uv) {assign_texture_coords(&hit_data);}
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

            Vec2 texture_points[3];

            float area;

            __host__ void precompute() {
                //precompute the plane the mesh lies on and (one of) its normal vectors. https://math.stackexchange.com/questions/2686606/equation-of-a-plane-passing-through-3-points
                Vec3 side1 = points[0] - points[1];
                Vec3 side2 = points[1] - points[2];

                normal_vec = side1.cross(side2).normalised();

                plane = {normal_vec.x, normal_vec.y, normal_vec.z};
                plane.d = -(plane.a * points[0].x + plane.b * points[0].y + plane.c * points[0].z);  //sub in a point to find constant

                area = 0.5 * side1.cross(side2).magnitude();
            }

            __device__ float get_ray_travelled_dist(Ray *ray) {
                //this algebra was worked out on paper (it is just the interesction between a line and plane really)
                float numerator = plane.d + plane.a * ray->origin.x + plane.b * ray->origin.y + plane.c * ray->origin.z;
                float denominator = plane.a * ray->direction.x + plane.b * ray->direction.y + plane.c * ray->direction.z;

                return -numerator / denominator;
            }

            __device__ Vec3 get_baycentric_coords(Vec3 hit_point) {
                //get the baycentric coords of the point in the triangle (https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates.html)
                Vec3 ab = points[1] - points[0];
                Vec3 ac = points[2] - points[0];
                Vec3 bc = points[2] - points[1];
                
                Vec3 ap = hit_point - points[0];
                Vec3 bp = hit_point - points[1];
                
                float area_u = 0.5 * ab.cross(ap).magnitude();
                float area_v = 0.5 * ac.cross(ap).magnitude();
                float area_w = 0.5 * bc.cross(bp).magnitude();

                return Vec3(area_u / area, area_v / area, area_w / area);
            }

            __device__ void assign_texture_coords(RayHitData *hit_data) {
                Vec3 b_coords = get_baycentric_coords(hit_data->hit_point);

                //linearlly interpolate the (u, v) texture points of each vertex
                hit_data->texture_uv = texture_points[0] * b_coords.x + texture_points[1] * b_coords.y + texture_points[2] * b_coords.z;
            }
};


__host__ __device__ class Quad {
    public:
        Triangle t1;
        Triangle t2;

        Material material;

        __host__ Quad() {}

        __host__ Quad(Vec3 p1, Vec3 p2, Vec3 p3, Vec3 p4, Material mat) {
            material = mat;

            point1 = p1;
            point2 = p2;
            point3 = p3;
            point4 = p4;

            create_triangles();
        }

        __device__ RayHitData hit(Ray *ray) {
            RayHitData t1_hit = t1.hit(ray);
            RayHitData t2_hit = t2.hit(ray);

            //get whichever triangle hits (or just the 2nd one if none of them hit)
            RayHitData hit_data;
            if (t1_hit.ray_hits) {
                hit_data = t1_hit;
            } else {
                hit_data = t2_hit;
            }

            return hit_data;
        }

    protected:  //NOTE: protected is used instead of private so these can be inherited by OneWayQuad
        Vec3 point1;
        Vec3 point2;
        Vec3 point3;
        Vec3 point4;

        __host__ void create_triangles() {
            //a quad is just 2 triangles
            Vertex v1{point1, Vec2(0, 0)};
            Vertex v2{point2, Vec2(1, 0)};
            Vertex v3{point3, Vec2(1, 1)};
            Vertex v4{point4, Vec2(0, 1)};

            t1 = Triangle(v1, v2, v3, material);
            t2 = Triangle(v1, v4, v3, material);
        }
};


__host__ __device__ class OneWayQuad : public Quad {
    public:
        __host__ OneWayQuad() {}

        __host__ OneWayQuad(Vec3 p1, Vec3 p2, Vec3 p3, Vec3 p4, bool invert_normal, Material mat) {
            material = mat;

            point1 = p1;
            point2 = p2;
            point3 = p3;
            point4 = p4;

            create_triangles();
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

        __host__ Cuboid() {}

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


__host__ __device__ class BoundingBox {
    public:
        Vec3 tl_near;
        
        float width;
        float height;
        float depth;

        Cuboid cuboid;

        __host__ BoundingBox() {}

        __host__ BoundingBox(Vec3 tl_near_pos, float w, float h, float d) {
            tl_near = tl_near_pos;
            width = w;
            height = h;
            depth = d;

            Texture t = Texture::create_const_colour(Vec3(0, 0, 0));
            Material m = Material::create_standard(t, 0);

            cuboid = Cuboid(tl_near, width, height, depth, m);
        }

        __device__ bool ray_hits(Ray *ray) {
            //TODO: make more efficient
            RayHitData hit_data = cuboid.hit(ray);

            return hit_data.ray_hits;
        }
};


__host__ __device__ class Mesh {
    public:
        __host__ Mesh() {}

        __host__ Mesh(std::vector<Triangle> host_triangle_array, Triangle *device_triangle_array) {
            host_triangles = host_triangle_array;
            device_triangles = device_triangle_array;

            num_triangles = host_triangle_array.size();

            get_bounding_box();
        }

        __device__ RayHitData hit(Ray *ray) {
            RayHitData closest_hit{false, INF};

            if (!bounding_box.ray_hits(ray)) {return closest_hit;}  //ray is never going to hit

            for (int i = 0; i < num_triangles; i++) {
                RayHitData hit_data = device_triangles[i].hit(ray);

                if (hit_data.ray_hits && hit_data.ray_travelled_dist < closest_hit.ray_travelled_dist) {closest_hit = hit_data;}
            }

            return closest_hit;
        }

    private:
        int num_triangles;

        Triangle *device_triangles;
        std::vector<Triangle> host_triangles;

        BoundingBox bounding_box;

        __host__ void get_bounding_box() {
            if (num_triangles == 0) {return;}  //0 triangles will cause errors

            float width = 0;
            float height = 0;
            float depth = 0;

            Vec3 tl = host_triangles[0].points[0];
            
            //loop through all triangles and update the bounds (it's ok that this is slow because it is done once before the first render)
            for (int i = 0; i < num_triangles; i++) {
                for (int x = 0; x < 3; x++) {
                    Vec3 point = host_triangles[i].points[x];
                    Vec3 diff = tl - point;

                    //update top left point (if point is too far left/up/near)
                    if (diff.x > 0) {
                        width += diff.x;
                        tl.x -= diff.x;
                    }
                    if (diff.y < 0) {
                        height -= diff.y;
                        tl.y -= diff.y;
                    }
                    if (diff.z > 0) {
                        depth += diff.z;
                        tl.z -= diff.z;
                    }

                    //update dimensions (if point is too far right/down/far)
                    if (-diff.x > width) {width = -diff.x;}
                    if (diff.y > height) {height = diff.y;}
                    if (-diff.z > depth) {depth = -diff.z;}
                }
            }

            bounding_box = BoundingBox(tl, width, height, depth);
        }
};


__host__ __device__ class Object {
    public:
        //again, like with the textures, inheritance and polymorphism is the correct way to go here. I just can't seem to get it to work on the GPU :(
        static const int SPHERE = 0;
        static const int TRIANGLE = 1;
        static const int QUAD = 2;
        static const int ONE_WAY_QUAD = 3;
        static const int CUBOID = 4;
        static const int MESH = 5;

        int type;

        Sphere sphere;
        Triangle triangle;
        Quad quad;
        OneWayQuad one_way_quad;
        Cuboid cuboid;
        Mesh mesh;

        Material material;

        __host__ Object(int t, Material mat) {
            type = t;
            material = mat;
        }

        __device__ RayHitData hit(Ray *ray) {
            switch (type) {
                case SPHERE:
                    return sphere.hit(ray);
                case TRIANGLE:
                    return triangle.hit(ray);
                case QUAD:
                    return quad.hit(ray);
                case ONE_WAY_QUAD:
                    return one_way_quad.hit(ray);
                case CUBOID:
                    return cuboid.hit(ray);
                case MESH:
                    return mesh.hit(ray);
            }
        }

        //initialisers for each type
        __host__ static Object create_sphere(Vec3 center, float radius, Material mat) {
            Sphere s(center, radius, mat);
            Object obj(SPHERE, mat);

            obj.sphere = s;

            return obj;
        }

        __host__ static Object create_triangle(Vec3 point1, Vec3 point2, Vec3 point3, Material mat) {
            Triangle t(point1, point2, point3, mat);
            Object obj(TRIANGLE, mat);

            obj.triangle = t;

            return obj;
        }

        __host__ static Object create_triangle(Vertex point1, Vertex point2, Vertex point3, Material mat) {
            Triangle t(point1, point2, point3, mat);
            Object obj(TRIANGLE, mat);

            obj.triangle = t;

            return obj;
        }

        __host__ static Object create_quad(Vec3 point1, Vec3 point2, Vec3 point3, Vec3 point4, Material mat) {
            Quad q(point1, point2, point3, point4, mat);
            Object obj(QUAD, mat);

            obj.quad = q;

            return obj;
        }

        __host__ static Object create_one_way_quad(Vec3 point1, Vec3 point2, Vec3 point3, Vec3 point4, bool invert_normal, Material mat) {
            OneWayQuad q(point1, point2, point3, point4, invert_normal, mat);
            Object obj(ONE_WAY_QUAD, mat);

            obj.one_way_quad = q;

            return obj;
        }

        __host__ static Object create_cuboid(Vec3 tl_near_pos, float width, float height, float depth, Material mat) {
            Cuboid c(tl_near_pos, width, height, depth, mat);
            Object obj(CUBOID, mat);

            obj.cuboid = c;

            return obj;
        }

        __host__ static Object create_mesh(std::vector<Triangle> host_triangle_array, Triangle *device_triangle_array, Material mat) {
            Mesh m(host_triangle_array, device_triangle_array);
            Object obj(MESH, mat);

            obj.mesh = m;

            return obj;
        }
};


__device__ struct AllObjects {
    Object *meshes;
    int num_meshes;
};


__constant__ AllObjects const_objects;