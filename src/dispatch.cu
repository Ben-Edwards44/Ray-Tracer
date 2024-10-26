#include <vector>
#include <random>
#include <stdexcept>

#include "raytracer.cu"


const int PIXEL_ARRAY_LEN = SCREEN_WIDTH * SCREEN_HEIGHT * 3;


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
            check_cuda_error(error, "allocating unified memory");
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
            check_cuda_error(error, "allocating read only device array");
            
            T *host_array = &host_values[0];  //get the pointer to the underlying array

            error = cudaMemcpy(device_pointer, host_array, mem_size, cudaMemcpyHostToDevice);  //copy the value over
            check_cuda_error(error, "copying real only device array");
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
            check_cuda_error(error, "allocating read only device value");
            
            error = cudaMemcpy(device_pointer, host_value, mem_size, cudaMemcpyHostToDevice);  //copy the value over
            check_cuda_error(error, "copying read only device value");
        }
};


void allocate_constant_mem(RenderData render_data, AllObjects mesh_data) {
    //to be called before first scene (NOTE: no need to free constant memory, camera data is assigned by itself)
    cudaMemcpyToSymbol(const_render_data, &render_data, sizeof(render_data));
    cudaMemcpyToSymbol(const_objects, &mesh_data, sizeof(mesh_data));
}


struct VariableRenderData {
    int frame_num;
    
    std::vector<float> previous_render;
};


dim3 get_block_size(int array_width, int array_height, dim3 thread_dim) {
    //we need to round up in cases where the array size is not divided exactly
    int blocks_x = array_width / thread_dim.x + 1;
    int blocks_y = array_height / thread_dim.y + 1;

    return dim3(blocks_x, blocks_y);
}


void run_ray_tracer(VariableRenderData *data, int current_time_ms) {
    //run the raytacing script on the gpu and store the result in the previous_render attribute of the variable render data
    ReadOnlyDeviceValue<int> current_time(current_time_ms);
    ReadOnlyDeviceValue<int> device_frame_num(data->frame_num);

    ReadOnlyDeviceArray<float> prev_render(data->previous_render);

    ReadWriteDeviceArray<float> image_pixels(PIXEL_ARRAY_LEN);

    dim3 thread_dim(16, 16);  //max is 1024
    dim3 block_dim = get_block_size(SCREEN_WIDTH, SCREEN_HEIGHT, thread_dim);

    get_pixel_colour<<<block_dim, thread_dim>>>(image_pixels.array, prev_render.device_pointer, current_time.device_pointer, device_frame_num.device_pointer);  //launch kernel

    cudaDeviceSynchronize();  //wait until gpu has finished

    //copy pixel data before freeing memory
    for (int i = 0; i < PIXEL_ARRAY_LEN; i++) {
        data->previous_render[i] = image_pixels.array[i];
    }

    //free memory
    current_time.free_memory();
    device_frame_num.free_memory();
    prev_render.free_memory();
    image_pixels.free_memory();
}


void render(VariableRenderData *data, int current_time_ms) {
    //run the ray tracer to render a scene and store the resulting pixel values in the previous_render in the scene object
    run_ray_tracer(data, current_time_ms);  //result stored in the previous render
    data->frame_num++;

    cudaError_t error = cudaPeekAtLastError();
    check_cuda_error(error, "final check after render");
}