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
        std::vector<float3> vertices;

        Object(std::string obj_filename) {
            file_contents = read_file(obj_filename);
            vertices = extract_vertices();
        }

    private:
        std::vector<std::string> file_contents;

        std::vector<float3> extract_vertices() {
            std::vector<float3> vertex_data;

            for (std::string line : file_contents) {
                std::vector<std::string> split_line = split_string(line, ' ');

                if (split_line[0] == "v") {
                    //this is a vertex, so get its coordinates
                    float3 vertex;
                    
                    vertex.x = std::stof(split_line[1]);
                    vertex.y = std::stof(split_line[2]);
                    vertex.z = std::stof(split_line[3]);

                    vertex_data.push_back(vertex);
                }
            }

            return vertex_data;
        }
};