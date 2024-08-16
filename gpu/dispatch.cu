#include <vector>
#include <random>
#include <stdexcept>

#include "raytracer.cu"


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


class Scene {
    public:
        CamData cam_data;
        RenderData render_data;

        std::vector<Sphere> spheres;
        std::vector<Triangle> triangles;
        std::vector<Quad> quads;
        std::vector<OneWayQuad> one_way_quads;

        int len_pixel_array;

        std::vector<float> previous_render;

        AllMeshes all_mesh_struct;

        Scene(CamData cam, RenderData r_data, std::vector<Sphere> s, std::vector<Triangle> t, std::vector<Quad> q, std::vector<OneWayQuad> o_q, int len) {
            cam_data = cam;
            render_data = r_data;
            spheres = s;
            triangles = t;
            quads = q;
            one_way_quads = o_q;
            len_pixel_array = len;

            previous_render = std::vector<float>(len_pixel_array);

            all_mesh_struct = get_meshes();

            assign_constant_mem();
        }

    private:
        void assign_constant_mem() {
            //to be called before first scene (NOTE: no need to free constant memory)
            cudaMemcpyToSymbol(const_all_meshes, &all_mesh_struct, sizeof(all_mesh_struct));
            cudaMemcpyToSymbol(const_cam_data, &cam_data, sizeof(cam_data));
        }

        AllMeshes get_meshes() {
            //NOTE: I'm not sure I ever free the memory used here... (I'll just leave it for now)
            ReadOnlyDeviceArray<Sphere> d_spheres(spheres);
            ReadOnlyDeviceArray<Triangle> d_triangles(triangles);
            ReadOnlyDeviceArray<Quad> d_quads(quads);
            ReadOnlyDeviceArray<OneWayQuad> d_one_way_quads(one_way_quads);

            int num_spheres = spheres.size();
            int num_triangles = triangles.size();
            int num_quads = quads.size();
            int num_one_way_quads = one_way_quads.size();

            AllMeshes meshes{d_spheres.device_pointer, d_triangles.device_pointer, d_quads.device_pointer, d_one_way_quads.device_pointer, num_spheres, num_triangles, num_quads, num_one_way_quads};

            return meshes;
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
    ReadOnlyDeviceValue<RenderData> r_data(scene->render_data);
    ReadOnlyDeviceValue<int> current_time(current_time_ms);

    ReadOnlyDeviceArray<float> prev_render(scene->previous_render);

    ReadWriteDeviceArray<float> image_pixels(scene->len_pixel_array);

    dim3 thread_dim(16, 16);  //max is 1024
    dim3 block_dim = get_block_size(scene->cam_data.image_width, scene->cam_data.image_height, thread_dim);

    get_pixel_colour<<<block_dim, thread_dim>>>(image_pixels.array, prev_render.device_pointer, r_data.device_pointer, current_time.device_pointer);  //launch kernel

    cudaDeviceSynchronize();  //wait until gpu has finished

    //copy pixel data before freeing memory
    for (int i = 0; i < scene->len_pixel_array; i++) {
        scene->previous_render[i] = image_pixels.array[i];
    }

    //free memory
    r_data.free_memory();
    current_time.free_memory();
    prev_render.free_memory();
    image_pixels.free_memory();
}


void render(Scene *scene, int current_time_ms) {
    //run the ray tracer to render a scene and store the resulting pixel values in the previous_render in the scene object
    run_ray_tracer(scene, current_time_ms);  //result stored in the previous render
    scene->render_data.frame_num++;

    cudaError_t error = cudaPeekAtLastError();
    check_error(error);
}