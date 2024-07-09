#include "api.cpp"
#include "raytracer.cu"
#include <vector>


void check_error(cudaError_t error) {
    if (error != cudaSuccess) {
        std::string err_msg = cudaGetErrorString(error);
        throw std::runtime_error("Error from CUDA: " + err_msg);
    }
}


class CameraData {
    public:
        float3 position;
        float3 tl_position;

        float focal_length;

        float view_width;
        float view_height;

        int image_width;
        int image_height;

        int pixel_array_len;

        float delta_u;
        float delta_v;

        std::vector<float> pixels;

        CameraData(float focal_len, float view_w, float view_h, int image_w, int image_h, std::vector<float> pos) {
            focal_length = focal_len;

            view_width = view_w;
            view_height = view_h;

            image_width = image_w;
            image_height = image_h;
            pixel_array_len = image_width * image_height * 3;

            set_pos(pos);
            set_tl_position();
            set_deltas();
            set_pixels();
        }

        void update_pixels(float new_array[]) {
            //assumes the array has the same size as the pixels vector
            for (int i = 0; i < pixels.size(); i++) {
                pixels[i] = new_array[i];
            }
        }

    private:
        void set_pos(std::vector<float> pos) {
            position.x = pos[0];
            position.y = pos[1];
            position.z = pos[2];
        }

        void set_deltas() {
            delta_u = view_width / image_width;
            delta_v = view_height / image_height;
        }

        void set_pixels() {
            std::vector<float> blank_screen(pixel_array_len, 1);
            pixels = blank_screen;
        }

        void set_tl_position() {
            //calc world position of top left pixel
            tl_position.x = position.x - view_width / 2;
            tl_position.y = position.y + view_height / 2;
            tl_position.z = focal_length;
        }
};


class MeshData {
    public:
        std::vector<Sphere> spheres;

        MeshData() {}

        void add_sphere(std::vector<float> center, float radius, std::vector<float> colour) {
            Vec3 cent(center[0], center[1], center[2]);
            Vec3 col(colour[0], colour[1], colour[2]);

            Sphere sphere = {cent, radius, col};

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


void run_ray_tracer(CameraData *camera, MeshData *mesh_data) {
    ReadWriteDeviceArray<float> image_pixels(camera->pixel_array_len);

    CamData cam_data = {camera->position, camera->tl_position, camera->focal_length, camera->delta_u, camera->delta_v, camera->image_width, camera->image_height};
    ReadOnlyDeviceValue<CamData> device_cam_data(cam_data);

    ReadOnlyDeviceArray<Sphere> spheres(mesh_data->spheres);
    ReadOnlyDeviceValue<int> num_spheres(mesh_data->spheres.size());

    dim3 thread_dim(4, 4);  //max is 1024
    dim3 block_dim = get_block_size(camera->image_width, camera->image_height, thread_dim);

    get_pixel_colour<<<block_dim, thread_dim>>>(image_pixels.array, device_cam_data.device_pointer, spheres.device_pointer, num_spheres.device_pointer);  //launch kernel

    cudaDeviceSynchronize();  //wait until gpu has finished

    camera->update_pixels(image_pixels.array);  //need to do this before freeing up the memory

    image_pixels.free_memory();
    device_cam_data.free_memory();
}


CameraData get_camera_data(JsonTree json) {
    float focal_len = json["camera"]["focal_length"].get_data()[0];
    std::vector<float> view_dims = json["camera"]["viewport"]["dimensions"].get_data();
    std::vector<float> img_dims = json["camera"]["image"]["dimensions"].get_data();
    std::vector<float> pos = json["camera"]["position"].get_data();

    CameraData camera(focal_len, view_dims[0], view_dims[1], img_dims[0], img_dims[1], pos);

    return camera;
}


MeshData get_mesh_data(JsonTree json) {
    MeshData mesh_data;

    int num_meshes = json["mesh_data"]["num_meshes"].get_data()[0];

    for (int i = 0; i < num_meshes; i++) {
        JsonTreeNode mesh = json["mesh_data"][std::to_string(i)];
        int type = mesh["type"].get_data()[0];

        if (type == 0) {
            mesh_data.add_sphere(mesh["center"].get_data(), mesh["radius"].get_data()[0], mesh["colour"].get_data());
        }
    }

    return mesh_data;
}


int main() {
    JsonTree json(recieve_filename);
    json.build_tree_from_file();

    CameraData camera = get_camera_data(json);
    MeshData mesh_data = get_mesh_data(json);

    run_ray_tracer(&camera, &mesh_data);
    send_pixel_data(camera.pixels);

    cudaError_t error = cudaPeekAtLastError();
    check_error(error);

    return 0;
}