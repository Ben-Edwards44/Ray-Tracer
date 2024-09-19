#include "material.cu"


const int SCREEN_WIDTH = 1000;
const int SCREEN_HEIGHT = 800;

const float ASPECT = static_cast<float>(SCREEN_WIDTH) / static_cast<float>(SCREEN_HEIGHT);  //static cast is used to stop integer division

const float PI = 3.141592653589793;


__device__ struct DeviceCamData {
    Vec3 cam_pos;
    Vec3 tl_pixel_pos;

    Vec3 delta_u;
    Vec3 delta_v;
};


__constant__ DeviceCamData const_cam_data;


__device__ Vec3 cam_pixel_to_world(int x, int y) {
    //convert a pixel point (x, y) on the screen plane into a world space coordinate
    Vec3 plane_point = const_cam_data.delta_u * x + const_cam_data.delta_v * y;

    return const_cam_data.tl_pixel_pos + plane_point;
}


__host__ __device__ class Camera {
    //these values can be changed
    const Vec3 CAM_POS = Vec3(0, 0, 0);

    const float FOV = 60 * (PI / 180);
    const float FOCAL_LEN = 0.1;

    const float X_ROT = 0 * (PI / 180);
    const float Y_ROT = 0 * (PI / 180);
    const float Z_ROT = 0 * (PI / 180);

    public:
        __host__ Camera() {}

        __host__ void assign_constant_mem() {
            float viewport_width = 2 * FOCAL_LEN * tan(FOV / 2);
            float viewport_height = viewport_width / ASPECT;

            Vec3 delta_u = get_u(viewport_width);
            Vec3 delta_v = get_v(viewport_height);
            Vec3 tl_pos = get_tl_pos(delta_u, delta_v);

            DeviceCamData cam_data{CAM_POS, tl_pos, delta_u, delta_v};
            
            cudaMemcpyToSymbol(const_cam_data, &cam_data, sizeof(cam_data));
        }

    private:
        __host__ Vec3 rotate_point(Vec3 point) {
            //use the matrix operations from object.cu to rotate a point
            std::vector<std::vector<float>> point_items = {{point.x}, {point.y}, {point.z}};
            Matrix rotated_point = RotationMatrix(RotationMatrix::X_AXIS, X_ROT) * RotationMatrix(RotationMatrix::Y_AXIS, Y_ROT) * RotationMatrix(RotationMatrix::Z_AXIS, Z_ROT) * Matrix(point_items);

            return Vec3(rotated_point.items[0][0], rotated_point.items[1][0], rotated_point.items[2][0]);
        }

        __host__ Vec3 get_u(float viewport_width) {
            //u is across the top of the screen, pointing left
            Vec3 default_point(1, 0, 0);
            Vec3 rotated_point = rotate_point(default_point);

            Vec3 u = rotated_point - Vec3(0, 0, 0);

            float mag_u = viewport_width / static_cast<float>(SCREEN_WIDTH);

            u.set_mag(mag_u);

            return u;
        }

        __host__ Vec3 get_v(float viewport_height) {
            //v is down the left of the screen, pointing down
            Vec3 default_point(0, -1, 0);
            Vec3 rotated_point = rotate_point(default_point);

            Vec3 v = rotated_point - Vec3(0, 0, 0);

            float mag_v = viewport_height / static_cast<float>(SCREEN_HEIGHT);

            v.set_mag(mag_v);

            return v;
        }

        __host__ Vec3 get_tl_pos(Vec3 delta_u, Vec3 delta_v) {
            //get the world position of the top left of the screen
            Vec3 z_offset(0, 0, CAM_POS.z + FOCAL_LEN);
            Vec3 u_step = delta_u * -SCREEN_WIDTH / 2;
            Vec3 v_step = delta_v * -SCREEN_HEIGHT / 2;

            return u_step + v_step + z_offset;
        }
};