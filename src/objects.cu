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
        Vec3 br_far;

        float width;
        float height;
        float depth;

        Cuboid cuboid;

        __host__ BoundingBox() {
            tl_near = Vec3(0, 0, 0);
            br_far = Vec3(0, 0, 0);
        }

        __host__ void grow(Triangle triangle) {
            //increase the bounding box size to include the new triangle
            for (int i = 0; i < 3; i++) {
                Vec3 new_point = triangle.points[i];

                if (!assigned) {
                    tl_near = new_point;
                    br_far = new_point;

                    assigned = true;

                    continue;
                }

                tl_near.x = min(tl_near.x, new_point.x);
                tl_near.y = max(tl_near.y, new_point.y);
                tl_near.z = min(tl_near.z, new_point.z);

                br_far.x = max(br_far.x, new_point.x);
                br_far.y = min(br_far.y, new_point.y);
                br_far.z = max(br_far.z, new_point.z);
            }

            update_dimensions();
        }

        __host__ void update_dimensions() {
            //update the width, height, depth after the bounding box has grown
            Vec3 dims = br_far - tl_near;

            width = dims.x;
            height = -dims.y;
            depth = dims.z;
        }

        __device__ RayHitData ray_hits(Ray *ray) {
            //TODO: make more efficient
            if (!assigned) {return RayHitData{false, INF};}  //the bounding box has zero volume

            RayHitData hit_data = cuboid.hit(ray);

            return hit_data;
        }

        __host__ void assign_cuboid() {
            //the texture and mat are not actually used
            Texture t = Texture::create_const_colour(Vec3(0, 0, 0));
            Material m = Material::create_standard(t, 0);

            cuboid = Cuboid(tl_near, width, height, depth, m);
        }

    private:
        bool assigned = false;
};


__host__ __device__ struct BoundingBoxTris {
    BoundingBox box;

    int num_tris;
    int *device_tri_inxs;
};


__host__ __device__ class BVH {
    public:
        __host__ BVH() {}

        //temp!!!!!
        int num_tri;

        __host__ BVH(std::vector<Triangle> host_tris, Triangle *device_tris, int max_depth) {
            host_triangles = host_tris;
            device_triangles = device_tris;

            num_tri = host_tris.size();

            std::vector<int> inxs;
            for (int i = 0; i < host_tris.size(); i++) {
                inxs.push_back(i);
            }

            printf("num tris: %u\n", (int)host_tris.size());

            root_node_inx = build(inxs, max_depth);

            printf("built: %u\n", root_node_inx);

            printf("tl: %f,%f,%f\n", data_array[root_node_inx].box.tl_near.x, data_array[root_node_inx].box.tl_near.y, data_array[root_node_inx].box.tl_near.z);
            printf("d: %f,%f,%f\n", data_array[root_node_inx].box.width, data_array[root_node_inx].box.height, data_array[root_node_inx].box.depth);

            printf("tree:\n");
            for (int i = 0; i <= root_node_inx; i++) {
                printf("\t%u\n", i);
                printf("\t\tbb: %f,%f,%f\n", data_array[i].box.tl_near.x, data_array[i].box.tl_near.y, data_array[i].box.tl_near.z);
                if (left_pointer[i] == -1) {printf("\t\ttris:%u\n", data_array[i].num_tris);}
            }

            allocate_mem();
        }

        __device__ Vec3 test(Ray *ray) {
            RayHitData closest_hit{false, INF};

            for (int i = 0; i < num_tri; i++) {
                Triangle triangle = device_triangles[i];
                RayHitData hit_data = triangle.hit(ray);

                if (hit_data.ray_hits && hit_data.ray_travelled_dist < closest_hit.ray_travelled_dist) {closest_hit = hit_data;}
            }

            RayHitData act = closest_hit;
            //RayHitData got = traverse(ray, root_node_inx);

            //if (!act.ray_hits) {return Vec3(0, 0, 1);}

            int b = traverse(ray, root_node_inx);

            //if (b != 0) {
            //    printf("tests: %u\n", b);
            //}

            float c = (float)b / 20.0;

            return Vec3(c, c, c);

            //if (got.hit_point == act.hit_point) {
            //    return Vec3(1, 1, 1);
            //} else {
            //    //printf("wanted: %f, %f, %f. Got: %f, %f, %f\n", act.hit_point.x, act.hit_point.y, act.hit_point.z, got.hit_point.x, got.hit_point.y, got.hit_point.z);
            //    return Vec3(1, 0, 0);
            //}

            //if (got_inx == -1) {
            //    return Vec3(1, 0, 0);
            //} else {
            //    float c = (float)got_inx / ((float)root_node_inx + 1);
            //    return Vec3(c, c, c);
            //}
        }

        __device__ RayHitData hit(Ray *ray) {
            //traverse the tree and only check the triangles at the leaf nodes
            return RayHitData{false, INF};//traverse(ray, root_node_inx);
        }

    private:
        int root_node_inx;

        std::vector<Triangle> host_triangles;
        Triangle* device_triangles;

        //host
        std::vector<BoundingBoxTris> data_array;
        std::vector<int> left_pointer;
        std::vector<int> right_pointer;

        //device
        BoundingBoxTris *device_data_array;
        int *device_left_pointer;
        int *device_right_pointer;

        __device__ int traverse(Ray *ray, int node_inx) {
            //perform a dfs of the tree (recursion does not play too well on the gpu, so an iterative method is used instead)
            DeviceStack<int> stack;

            RayHitData best_hit{false, INF};

            //printf("hkjlasdf\n");

            bool root_hit = device_data_array[node_inx].box.ray_hits(ray).ray_hits;
            if (root_hit) {stack.push(node_inx);}

            int stats = 0;

            while (!stack.is_empty()) {
                int current_inx = stack.pop();

                int l = device_left_pointer[current_inx];
                int r = device_right_pointer[current_inx];

                //stats++;

                if (l == -1 && r == -1) {
                    //leaf node - check triangles
                    //stats += device_data_array[current_inx].num_tris;

                    stats = current_inx;

                    RayHitData leaf_triangle_hit = check_leaf_node(ray, current_inx);
                    if (leaf_triangle_hit.ray_hits && (leaf_triangle_hit.ray_travelled_dist < best_hit.ray_travelled_dist)) {best_hit = leaf_triangle_hit;}  //new best hit

                    continue;
                }

                RayHitData l_box = device_data_array[l].box.ray_hits(ray);
                RayHitData r_box = device_data_array[r].box.ray_hits(ray);

                if (l_box.ray_hits && l_box.ray_travelled_dist > 0) {
                    stack.push(l);
                }
                if (r_box.ray_hits && r_box.ray_travelled_dist > 0) {
                    stack.push(r);
                }
            }

            return stats;//best_hit;
        }

        __device__ RayHitData check_leaf_node(Ray *ray, int node_inx) {
            BoundingBoxTris node = device_data_array[node_inx];

            RayHitData closest_hit{false, INF};

            for (int i = 0; i < node.num_tris; i++) {
                int inx = node.device_tri_inxs[i];
                Triangle triangle = device_triangles[inx];
                RayHitData hit_data = triangle.hit(ray);

                if (hit_data.ray_hits && hit_data.ray_travelled_dist < closest_hit.ray_travelled_dist) {closest_hit = hit_data;}
            }

            return closest_hit;
        }

        __host__ int build(std::vector<int> triangle_inxs, int depth) {
            //build a binary search tree of bounding boxes to form a BVH
            BoundingBox current_box;

            for (int i : triangle_inxs) {
                current_box.grow(host_triangles[i]);
            }

            if (depth <= 0) {
                //this is a leaf node
                add_tree_node(current_box, -1, -1, triangle_inxs);
                return data_array.size() - 1;
            }

            std::pair<std::vector<int>, std::vector<int>> splitted = split_triangles(triangle_inxs, current_box);

            int left_pointer = build(splitted.first, depth - 1);
            int right_pointer = build(splitted.second, depth - 1);

            add_tree_node(current_box, left_pointer, right_pointer, triangle_inxs);

            return data_array.size() - 1;
        }

        __host__ std::pair<std::vector<int>, std::vector<int>> split_triangles(std::vector<int> triangle_inxs, BoundingBox box) {
            //split the triangles in the box along its longest axis
            int axis = 0;
            float len = box.width;

            if (box.height > len) {
                axis = 2;
                len = box.height;
            }
            //if (box.depth > len) {
            //    axis = 3;
            //    len = box.depth;
            //}

            std::pair<std::vector<int>, std::vector<int>> splitted;

            for (int i : triangle_inxs) {
                Vec3 point = host_triangles[i].points[0];

                bool go_in_a;
                if (axis == 0) {
                    go_in_a = point.x < box.tl_near.x + len / 2;
                } else if (axis == 1) {
                    go_in_a = point.y < box.tl_near.y - len / 2;
                } else {
                    go_in_a = point.z < box.tl_near.z + len / 2;
                }

                if (go_in_a) {
                    splitted.first.push_back(i);
                } else {
                    splitted.second.push_back(i);
                }
            }

            return splitted;
        }

        __host__ Vec3 get_ref_point(BoundingBox current_box, bool horizontal_split) {
            if (horizontal_split) {
                //we want the center of the top plane
                return Vec3(current_box.tl_near.x + current_box.width / 2, current_box.tl_near.y, current_box.tl_near.z + current_box.depth / 2);
            } else {
                //we want the center of the left plane
                return Vec3(current_box.tl_near.x, current_box.tl_near.y - current_box.height / 2, current_box.tl_near.z + current_box.depth / 2);
            }
        }

        __host__ void add_tree_node(BoundingBox node, int left, int right, std::vector<int> triangle_inxs) {
            left_pointer.push_back(left);
            right_pointer.push_back(right);

            node.assign_cuboid();

            //allocate the memory for the BoundingBoxTris struct
            BoundingBoxTris box{node, static_cast<int>(triangle_inxs.size())};

            int mem_size = triangle_inxs.size() * sizeof(int);

            cudaError_t error = cudaMalloc((void **)&box.device_tri_inxs, mem_size);
            check_cuda_error(error, "allocating bvh node triangle inxs");

            int *tri_inxs = &triangle_inxs[0];

            error = cudaMemcpy(box.device_tri_inxs, tri_inxs, mem_size, cudaMemcpyHostToDevice);
            check_cuda_error(error, "copying bvh node triangle inxs");

            data_array.push_back(box);
        }

        __host__ void allocate_mem() {
            //allocate the tree arrays to the device memory
            int data_mem_size = data_array.size() * sizeof(BoundingBoxTris);
            int pointer_mem_size = left_pointer.size() * sizeof(int);

            //allocate the memory
            cudaError_t error = cudaMalloc((void **)&device_data_array, data_mem_size);
            check_cuda_error(error, "allocating bvh data array memory");

            error = cudaMalloc((void **)&device_left_pointer, pointer_mem_size);
            check_cuda_error(error, "allocating bvh left pointer memory");

            error = cudaMalloc((void **)&device_right_pointer, pointer_mem_size);
            check_cuda_error(error, "allocating bvh right pointer memory");

            //get the pointers to the underlying arrays
            BoundingBoxTris *data = &data_array[0];
            int *left = &left_pointer[0];
            int *right = &right_pointer[0];
            
            //copy the values over
            error = cudaMemcpy(device_data_array, data, data_mem_size, cudaMemcpyHostToDevice);
            check_cuda_error(error, "copying bvh data array");

            error = cudaMemcpy(device_left_pointer, left, pointer_mem_size, cudaMemcpyHostToDevice);
            check_cuda_error(error, "copying bvh left pointers");

            error = cudaMemcpy(device_right_pointer, right, pointer_mem_size, cudaMemcpyHostToDevice);
            check_cuda_error(error, "copying bvh right pointers");
        }
};


__host__ __device__ class Mesh {
    public:
        BVH bvh;

        __host__ Mesh() {}

        __host__ Mesh(std::vector<Triangle> host_triangle_array, Triangle *device_triangle_array) {
            host_triangles = host_triangle_array;
            device_triangles = device_triangle_array;

            num_triangles = host_triangle_array.size();

            bvh = BVH(host_triangles, device_triangles, 4);
        }

        __device__ Vec3 test(Ray *ray) {
            return bvh.test(ray);
        }

        __device__ RayHitData hit(Ray *ray) {
            return bvh.hit(ray);

            /*
            RayHitData closest_hit{false, INF};

            //if (!bounding_box.ray_hits(ray)) {return closest_hit;}  //ray is never going to hit

            for (int i = 0; i < num_triangles; i++) {
                RayHitData hit_data = device_triangles[i].hit(ray);

                if (hit_data.ray_hits && hit_data.ray_travelled_dist < closest_hit.ray_travelled_dist) {closest_hit = hit_data;}
            }

            return closest_hit;
            */
        }

    private:
        int num_triangles;

        std::vector<Triangle> host_triangles;
        Triangle *device_triangles;

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

        __device__ Vec3 test(Ray *ray) {
            return mesh.test(ray);
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