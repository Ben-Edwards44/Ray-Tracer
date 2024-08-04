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


struct DataToSend {
    //data that must be sent to gpu
    CamData cam_data;
    MeshData mesh_data;
    RenderData render_data;

    int len_pixel_array;

    std::vector<float> previous_render;
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


void run_ray_tracer(DataToSend *data_obj) {
    //run the raytacing script on the gpu and store the result in the data_obj previous_render
    //assign memory on the gpu 
    ReadOnlyDeviceValue<CamData> device_cam_data(data_obj->cam_data);
    ReadOnlyDeviceValue<RenderData> r_data(data_obj->render_data);
    ReadOnlyDeviceValue<int> current_time(get_time());

    ReadOnlyDeviceArray<Sphere> spheres(data_obj->mesh_data.spheres);
    ReadOnlyDeviceArray<float> previous_render(data_obj->previous_render);

    ReadWriteDeviceArray<float> image_pixels(data_obj->len_pixel_array);

    dim3 thread_dim(16, 16);  //max is 1024
    dim3 block_dim = get_block_size(data_obj->cam_data.image_width, data_obj->cam_data.image_height, thread_dim);

    get_pixel_colour<<<block_dim, thread_dim>>>(image_pixels.array, previous_render.device_pointer, device_cam_data.device_pointer, spheres.device_pointer, r_data.device_pointer, current_time.device_pointer);  //launch kernel

    cudaDeviceSynchronize();  //wait until gpu has finished

    //copy pixel data before freeing memory
    for (int i = 0; i < data_obj->len_pixel_array; i++) {
        data_obj->previous_render[i] = image_pixels.array[i];
    }

    image_pixels.free_memory();
    device_cam_data.free_memory();
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
        int mat_type = mesh["material"]["type"].get_data()[0];

        Vec3 mat_c(mat_colour[0], mat_colour[1], mat_colour[2]);
        Vec3 mat_e_c(mat_emit_colour[0], mat_emit_colour[1], mat_emit_colour[2]);

        Material material{mat_c, mat_emit_strength, mat_e_c, mat_type};

        //set material options (different depending on material type)
        if (mat_type == 2) {
            material.fuzz_level = mesh["material"]["options"]["fuzz_level"].get_data()[0];
        }

        if (type == 0) {
            mesh_data.add_sphere(mesh["center"].get_data(), mesh["radius"].get_data()[0], &material);
        }
    }

    return mesh_data;
}


RenderData get_render_data(JsonTree json, int num_spheres) {
    int reflect_limit = json["ray_data"]["reflect_limit"].get_data()[0];
    int rays_per_pixel = json["ray_data"]["rays_per_pixel"].get_data()[0];
    bool static_scene = json["static_scene"].get_data()[0] == 1;

    std::vector<float> sky_colour = json["sky_colour"].get_data();
    Vec3 sky_col_vec(sky_colour[0], sky_colour[1], sky_colour[2]);

    int start_frame_num = 0;
    
    return RenderData {rays_per_pixel, reflect_limit, num_spheres, start_frame_num, static_scene, sky_col_vec};
}


void update_data_to_send(DataToSend *data_obj) {
    data_obj->render_data.frame_num++;

    if (data_obj->render_data.static_scene) {return;}  //the data will be the same as before, no need to update it

    JsonTree json(recieve_filename);
    json.build_tree_from_file();

    CamData cam_data = get_camera_data(json);
    MeshData mesh_data = get_mesh_data(json);
    RenderData render_data = get_render_data(json, mesh_data.spheres.size());

    int len_pixel_array = cam_data.image_width * cam_data.image_height * 3;
    std::vector<float> blank(len_pixel_array, 0);

    //assign the new values
    data_obj->cam_data = cam_data;
    data_obj->mesh_data = mesh_data;
    data_obj->render_data = render_data;
    data_obj->len_pixel_array = len_pixel_array;
    data_obj->previous_render = blank;
}


void render(DataToSend *data_obj) {
    update_data_to_send(data_obj);

    run_ray_tracer(data_obj);  //result stored in the previous render
    send_pixel_data(data_obj->previous_render);

    cudaError_t error = cudaPeekAtLastError();
    check_error(error);
}


int main() {
    DataToSend data_obj;

    //set default values
    data_obj.render_data.static_scene = false;

    std::string input;
    while (true) {
        std::cin >> input;

        if (input == "render") {
            render(&data_obj);
            std::cout << "render_complete\n";
        } else if (input == "quit") {
            break;
        }
    }

    return 0;
}