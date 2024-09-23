#include <vector>
#include "utils.cu"


//at the moment, matrix operations are only supported on the host
class Matrix {
    public:
        int rows;
        int cols;

        std::vector<std::vector<float>> items;

        Matrix() {}

        Matrix(int num_row, int num_col) {
            rows = num_row;
            cols = num_col;

            reset_items();
        }

        Matrix(std::vector<std::vector<float>> mat_items) {
            items = mat_items;

            rows = mat_items.size();
            cols = (rows == 0 ? 0 : mat_items[0].size());
        }

        Matrix operator*(Matrix other_mat) {
            //perform matrix multiplication
            if (cols != other_mat.rows) {throw std::runtime_error("Matrix dimensions do not match for multiplication.");}

            int new_rows = rows;
            int new_cols = other_mat.cols;

            Matrix new_mat(new_rows, new_cols);

            for (int row_inx = 0; row_inx < new_rows; row_inx++) {
                for (int col_inx = 0; col_inx < new_cols; col_inx++) {
                    float sum = 0;

                    for (int i = 0; i < cols; i++) {
                        sum += items[row_inx][i] * other_mat.items[i][col_inx];
                    }

                    new_mat.items[row_inx][col_inx] = sum;
                }
            }

            return new_mat;
        }

        Matrix transpose() {
            //swap rows and cols
            Matrix transposed(cols, rows);

            for (int row_inx = 0; row_inx < rows; row_inx++) {
                for (int col_inx = 0; col_inx < cols; col_inx++) {
                    transposed.items[col_inx][row_inx] = items[row_inx][col_inx];
                }
            }

            return transposed;
        }

        void reset_items() {
            //populate the items with 0s
            std::vector<std::vector<float>> blank_mat(rows, std::vector<float>(cols, 0));  //2d array filled with 0s
            items = blank_mat;
        }
};


class EnlargementMatrix : public Matrix {
    public:
        float scale;

        EnlargementMatrix(float scale_fact, int dimensions) {
            rows = dimensions;
            cols = dimensions;

            scale = scale_fact;

            set_items();
        }

    private:
        void set_items() {
            reset_items();

            //set the leading diagonal to be the scale factor
            for (int i = 0; i < rows; i++) {
                items[i][i] = scale;
            }
        }
};


class RotationMatrix : public Matrix {
    public:
        static const int X_AXIS = 0;
        static const int Y_AXIS = 1;
        static const int Z_AXIS = 2;

        int axis;
        float angle;

        RotationMatrix(int rot_axis, float rot_angle) {
            axis = rot_axis;
            angle = rot_angle;

            rows = 3;
            cols = 3;

            set_items();
        }

    private:
        void x_rot(float s, float c) {
            items.push_back(std::vector<float>{1, 0, 0});
            items.push_back(std::vector<float>{0, c, s});
            items.push_back(std::vector<float>{0, -s, c});
        }

        void y_rot(float s, float c) {
            items.push_back(std::vector<float>{c, 0, -s});
            items.push_back(std::vector<float>{0, 1, 0});
            items.push_back(std::vector<float>{s, 0, c});
        }

        void z_rot(float s, float c) {
            items.push_back(std::vector<float>{c, -s, 0});
            items.push_back(std::vector<float>{s, c, 0});
            items.push_back(std::vector<float>{0, 0, 1});
        }

        void set_items() {
            //https://en.wikipedia.org/wiki/Rotation_matrix
            float s = sin(angle);
            float c = cos(angle);

            if (axis == X_AXIS) {
                x_rot(s, c);
            } else if (axis == Y_AXIS) {
                y_rot(s, c);
            } else {
                z_rot(s, c);
            }
        }
};