#include "api.cpp"
#include "raytracer.cu"
#include <vector>
#include <chrono>
#include <random>


void check_error(cudaError_t error) {
    if (error != cudaSuccess) {
        std::string err_msg = cudaGetErrorString(error);
        throw std::runtime_error("Error from CUDA: " + err_msg);
    }
}


class MeshData {
    public:
        std::vector<Sphere> spheres;

        MeshData() {}

        void add_sphere(std::vector<float> center, float radius, Material *material) {
            Vec3 cent(center[0], center[1], center[2]);

            Sphere sphere(cent, radius, *material);

            spheres.push_back(sphere);
        }
};


template<typename T>
class ReadWriteDeviceArray {
    public:
        T *array;

        ReadWriteDeviceArray(int array_len) {
            len = array_len;
            mem_size = len * sizeof(T);

            allocate_unified_mem();
        }

        void free_memory() {
            //the memory cannot be accessed by cpu or gpu after calling this
            cudaFree(array);
        }

    private:
        int len;
        int mem_size;

        void allocate_unified_mem() {
            //allocate memory that can be accessed by both the gpu and cpu
            cudaError_t error = cudaMallocManaged(&array, mem_size);
            check_error(error);
        }
};


template <typename T>
class ReadOnlyDeviceArray {
    public:
        T *device_pointer;

        ReadOnlyDeviceArray(std::vector<T> values) {
            host_values = values;
            mem_size = sizeof(T) * values.size();

            allocate_mem();
        }

        void free_memory() {
            //should be called after we have finished with the data
            cudaFree(device_pointer);
        }
    
    private:
        int mem_size;
        std::vector<T> host_values;

        void allocate_mem() {
            cudaError_t error = cudaMalloc((void **)&device_pointer, mem_size);  //allocate the memory
            check_error(error);
            
            T *host_array = &host_values[0];  //get the pointer to the underlying array

            error = cudaMemcpy(device_pointer, host_array, mem_size, cudaMemcpyHostToDevice);  //copy the value over
            check_error(error);
        }
};


template <typename T>
class ReadOnlyDeviceValue {
    public:
        T *host_value;
        T *device_pointer;

        ReadOnlyDeviceValue(T value) {
            host_value = &value;
            mem_size = sizeof(T);

            allocate_mem();
        }

        void free_memory() {
            //should be called after we have finished with the data
            cudaFree(device_pointer);
        }
    
    private:
        int mem_size;

        void allocate_mem() {
            cudaError_t error = cudaMalloc((void **)&device_pointer, mem_size);  //allocate the memory
            check_error(error);
            
            error = cudaMemcpy(device_pointer, host_value, mem_size, cudaMemcpyHostToDevice);  //copy the value over
            check_error(error);
        }
};


dim3 get_block_size(int array_width, int array_height, dim3 thread_dim) {
    //we need to round up in cases where the array size is not divided exactly
    int blocks_x = array_width / thread_dim.x + 1;
    int blocks_y = array_height / thread_dim.y + 1;

    return dim3(blocks_x, blocks_y);
}


int get_time() {
    //get ms since epoch
    auto clock = std::chrono::system_clock::now();
    auto duration = clock.time_since_epoch();
    int time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    return time;
}


std::vector<float> run_ray_tracer(CamData *cam_data, MeshData *mesh_data, RenderData *render_data) {    
    ReadOnlyDeviceValue<CamData> device_cam_data(*cam_data);
    ReadOnlyDeviceArray<Sphere> spheres(mesh_data->spheres);
    ReadOnlyDeviceValue<RenderData> r_data(*render_data);

    int len_pixel_array = cam_data->image_width * cam_data->image_height * 3;
    ReadWriteDeviceArray<float> image_pixels(len_pixel_array);

    dim3 thread_dim(16, 16);  //max is 1024
    dim3 block_dim = get_block_size(cam_data->image_width, cam_data->image_height, thread_dim);

    get_pixel_colour<<<block_dim, thread_dim>>>(image_pixels.array, device_cam_data.device_pointer, spheres.device_pointer, r_data.device_pointer);  //launch kernel

    cudaDeviceSynchronize();  //wait until gpu has finished

    //copy pixel data before freeing memory
    std::vector<float> resulting_pixels;
    for (int i = 0; i < len_pixel_array; i++) {
        resulting_pixels.push_back(image_pixels.array[i]);
    }

    image_pixels.free_memory();
    device_cam_data.free_memory();

    return resulting_pixels;
}


CamData get_camera_data(JsonTree json) {
    float focal_len = json["camera"]["focal_length"].get_data()[0];
    std::vector<float> view_dims = json["camera"]["viewport"]["dimensions"].get_data();
    std::vector<float> img_dims = json["camera"]["image"]["dimensions"].get_data();
    std::vector<float> pos = json["camera"]["position"].get_data();

    Vec3 cam_pos(pos[0], pos[1], pos[2]);
    Vec3 tl_pos(cam_pos.x - view_dims[0] / 2, cam_pos.y + view_dims[1] / 2, cam_pos.z + focal_len);

    int img_width = img_dims[0];
    int img_height = img_dims[1];

    float delta_u = view_dims[0] / img_width;
    float delta_v = view_dims[1] / img_height;

    return CamData{cam_pos, tl_pos, focal_len, delta_u, delta_v, img_width, img_height};
}


MeshData get_mesh_data(JsonTree json) {
    MeshData mesh_data;

    int num_meshes = json["mesh_data"]["num_meshes"].get_data()[0];

    for (int i = 0; i < num_meshes; i++) {
        JsonTreeNode mesh = json["mesh_data"][std::to_string(i)];
        int type = mesh["type"].get_data()[0];

        std::vector<float> mat_colour = mesh["material"]["colour"].get_data();
        float mat_emit_strength = mesh["material"]["emission_strength"].get_data()[0];
        std::vector<float> mat_emit_colour = mesh["material"]["emission_colour"].get_data();

        Vec3 mat_c(mat_colour[0], mat_colour[1], mat_colour[2]);
        Vec3 mat_e_c(mat_emit_colour[0], mat_emit_colour[1], mat_emit_colour[2]);

        Material material{mat_c, mat_emit_strength, mat_e_c};

        if (type == 0) {
            mesh_data.add_sphere(mesh["center"].get_data(), mesh["radius"].get_data()[0], &material);
        }
    }

    return mesh_data;
}


RenderData get_render_data(JsonTree json, int num_spheres) {
    int reflect_limit = json["ray_data"]["reflect_limit"].get_data()[0];
    int rays_per_pixel = json["ray_data"]["rays_per_pixel"].get_data()[0];
    int time = get_time();
    
    return RenderData {rays_per_pixel, reflect_limit, num_spheres, get_time()};
}


int main() {
    JsonTree json(recieve_filename);
    json.build_tree_from_file();

    CamData cam_data = get_camera_data(json);
    MeshData mesh_data = get_mesh_data(json);
    RenderData render_data = get_render_data(json, mesh_data.spheres.size());

    int start = get_time();

    std::vector<float> pixel_colours = run_ray_tracer(&cam_data, &mesh_data, &render_data);

    int end = get_time();

    printf("Elapsed: %ums\n", end - start);

    send_pixel_data(pixel_colours);

    cudaError_t error = cudaPeekAtLastError();
    check_error(error);

    return 0;
}