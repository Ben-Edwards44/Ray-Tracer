#include "raytracer.cu"
#include <vector>
#include <random>


struct Scene {
    CamData cam_data;
    RenderData render_data;

    std::vector<Sphere> spheres;

    int len_pixel_array;

    std::vector<float> previous_render;
};


void check_error(cudaError_t error) {
    if (error != cudaSuccess) {
        std::string err_msg = cudaGetErrorString(error);
        throw std::runtime_error("Error from CUDA: " + err_msg);
    }
}


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


void run_ray_tracer(Scene *scene, int current_time_ms) {
    //run the raytacing script on the gpu and store the result in the data_obj previous_render
    //assign memory on the gpu 
    ReadOnlyDeviceValue<CamData> device_cam_data(scene->cam_data);
    ReadOnlyDeviceValue<RenderData> r_data(scene->render_data);
    ReadOnlyDeviceValue<int> current_time(current_time_ms);

    ReadOnlyDeviceArray<Sphere> spheres(scene->spheres);
    ReadOnlyDeviceArray<float> prev_render(scene->previous_render);

    ReadWriteDeviceArray<float> image_pixels(scene->len_pixel_array);

    dim3 thread_dim(16, 16);  //max is 1024
    dim3 block_dim = get_block_size(scene->cam_data.image_width, scene->cam_data.image_height, thread_dim);

    get_pixel_colour<<<block_dim, thread_dim>>>(image_pixels.array, prev_render.device_pointer, device_cam_data.device_pointer, spheres.device_pointer, r_data.device_pointer, current_time.device_pointer);  //launch kernel

    cudaDeviceSynchronize();  //wait until gpu has finished

    //copy pixel data before freeing memory
    for (int i = 0; i < scene->len_pixel_array; i++) {
        scene->previous_render[i] = image_pixels.array[i];
    }

    image_pixels.free_memory();
    device_cam_data.free_memory();
}


void render(Scene *scene, int current_time_ms) {
    //run the ray tracer to render a scene and store the resulting pixel values in the previous_render in the scene object
    run_ray_tracer(scene, current_time_ms);  //result stored in the previous render

    cudaError_t error = cudaPeekAtLastError();
    check_error(error);
}