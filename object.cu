#include <cmath>
#include <vector>
#include <fstream>


std::vector<std::string> read_file(std::string filename) {
    std::ifstream file(filename);
    if (!file) {throw std::runtime_error("Could not find file to open.");}

    std::vector<std::string> lines;

    while (!file.eof()) {
        std::string line;
        getline(file, line);  //read next line of file
        
        lines.push_back(line);
    }

    file.close();
            
    return lines;
}


std::vector<std::string> split_string(std::string str, char split_char) {
    //implementation of python's .split() function
    std::string current_str = "";
    std::vector<std::string> splitted;

    for (char i : str) {
        if (i == split_char) {
            splitted.push_back(current_str);
            current_str = " ";
        } else {
            current_str += i;
        }
    }

    splitted.push_back(current_str);

    return splitted;
}


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


class Object {
    public:
        Matrix vertex_mat;
        std::vector<std::vector<float3>> faces;

        Object(std::string filename) {
            file_contents = read_file(filename);
            vertex_mat = get_vertex_mat(read_vertices());

            extract_faces();
        }

        void enlarge(float scale_fact) {
            EnlargementMatrix transform_mat(scale_fact, 3);
            vertex_mat = transform_mat * vertex_mat;  //NOTE: order matters in multiplication
            
            extract_faces();  //rebuild the faces with the transformed vertices
        }

        void rotate(float x_angle, float y_angle, float z_angle) {
            //NOTE: angles are measured in radians
            RotationMatrix x_rot(RotationMatrix::X_AXIS, x_angle);
            RotationMatrix y_rot(RotationMatrix::Y_AXIS, y_angle);
            RotationMatrix z_rot(RotationMatrix::Z_AXIS, z_angle);

            vertex_mat = x_rot * y_rot * z_rot * vertex_mat;  //NOTE: as long as the rotations are before the vertex_mat, the order does not matter
            
            extract_faces();  //rebuild the faces with the transformed vertices
        }

        void translate(float offset_x, float offset_y, float offset_z) {
            for (int i = 0; i < vertex_mat.cols; i++) {
                vertex_mat.items[0][i] += offset_x;
                vertex_mat.items[1][i] += offset_y;
                vertex_mat.items[2][i] += offset_z;
            }

            extract_faces();  //rebuild the faces with the transformed vertices
        }

    private:
        std::vector<std::string> file_contents;
        
        std::vector<std::vector<float>> read_vertices() {
            //read the coordinates of each vertex from the .obj file
            std::vector<std::vector<float>> vertices;

            for (std::string line : file_contents) {
                std::vector<std::string> split_line = split_string(line, ' ');

                if (split_line[0] == "v") {
                    //this is a vertex, so get its coordinates
                    std::vector<float> vertex;
                    
                    vertex.push_back(std::stof(split_line[1]));
                    vertex.push_back(std::stof(split_line[2]));
                    vertex.push_back(std::stof(split_line[3]));

                    vertices.push_back(vertex);
                }
            }

            return vertices;
        }

        Matrix get_vertex_mat(std::vector<std::vector<float>> vertices) {
            //convert a 2d vector of vertices to a matrix
            Matrix transposed_vertex_matrix(vertices);

            return transposed_vertex_matrix.transpose();
        }

        void extract_faces() {
            //populate the faces vector with the vertices from each face
            faces.clear();

            for (std::string line : file_contents) {
                std::vector<std::string> split_line = split_string(line, ' ');

                if (split_line[0] == "f") {
                    //this is a face
                    std::vector<float3> face;

                    for (int i = 1; i < split_line.size(); i++) {
                        //add the vertex to the face
                        std::vector<std::string> split_inxs = split_string(split_line[i], '/');
                        int vertex_inx = std::stoi(split_inxs[0]) - 1;  //must -1 because .obj is 1-indexed (for some odd reason)

                        float3 vertex;
                        vertex.x = vertex_mat.items[0][vertex_inx];
                        vertex.y = vertex_mat.items[1][vertex_inx];
                        vertex.z = vertex_mat.items[2][vertex_inx];

                        face.push_back(vertex);
                    }

                    faces.push_back(face);  //add the face to the total list of faces
                }
            }
        }
};