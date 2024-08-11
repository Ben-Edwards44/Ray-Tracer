#include <vector>
#include <fstream>


std::vector<std::string> read_file(std::string filename) {
    std::ifstream file(filename);
    if (!file) {throw std::logic_error("Could not find file to open.");}

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


class Object {
    public:
        std::vector<std::vector<float3>> faces;

        Object(std::string obj_filename, float offset_x, float offset_y, float offset_z) {
            file_contents = read_file(obj_filename);

            off_x = offset_x;
            off_y = offset_y;
            off_z = offset_z;

            extract_faces();
        }

    private:
        float off_x;
        float off_y;
        float off_z;

        std::vector<std::string> file_contents;

        std::vector<float3> extract_vertices() {
            //get the coordinates of each vertex from the .obj file
            std::vector<float3> vertices;

            for (std::string line : file_contents) {
                std::vector<std::string> split_line = split_string(line, ' ');

                if (split_line[0] == "v") {
                    //this is a vertex, so get its coordinates
                    float3 vertex;
                    
                    vertex.x = std::stof(split_line[1]) + off_x;
                    vertex.y = std::stof(split_line[2]) + off_y;
                    vertex.z = std::stof(split_line[3]) + off_z;

                    vertices.push_back(vertex);
                }
            }

            return vertices;
        }

        void extract_faces() {
            //populate the faces vector with the vertices from each face
            std::vector<float3> vertices = extract_vertices();

            for (std::string line : file_contents) {
                std::vector<std::string> split_line = split_string(line, ' ');

                if (split_line[0] == "f") {
                    //this is a face
                    std::vector<float3> face;

                    for (int i = 1; i < split_line.size(); i++) {
                        //add the vertex to the face
                        std::vector<std::string> split_inxs = split_string(split_line[i], '/');
                        int vertex_inx = std::stoi(split_inxs[0]) - 1;  //must -1 because .obj is 1-indexed (for some odd reason)

                        face.push_back(vertices[vertex_inx]);
                    }

                    faces.push_back(face);  //add the face to the total list of faces
                }
            }
        }
};