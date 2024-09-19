#include <cmath>
#include <vector>
#include <fstream>

#include "dispatch.cu"


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


class ObjFileMesh {
    public:
        Matrix vertex_mat;
        std::vector<std::vector<float3>> faces;

        ObjFileMesh(std::string filename) {
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