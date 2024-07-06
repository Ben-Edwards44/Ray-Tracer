#include <iostream>
#include <fstream>
#include <stdexcept>
#include <map>
#include <vector>


const std::string recieve_filename = "gpu/to_device.json";
const std::string send_filename = "gpu/to_host.json";


class JsonTreeNode {
    public:
        JsonTreeNode(std::string key) {
            node_key = key;
        }

        JsonTreeNode() {}  //no argument constructor for when we are copying an object.

        JsonTreeNode operator[] (std::string key) {
            return children[key];
        }

        std::vector<float> get_data() {
            //return the actual data from the node
            if (leaf_node) {
                return data;
            } else {
                throw std::range_error("Cannot get data from non leaf node");
            }
        }

        void add_data(std::vector<float> added_data) {
            //to be called on leaf nodes when building the tree from cuda
            leaf_node = true;
            data = added_data;
        }

        void add_child(JsonTreeNode child) {
            //to be called when building the tree from cuda
            children[child.node_key] = child;
        }

        void parse_value(std::string value_string) {
            //parse a JSON string and recursively build a tree from it
            int string_len = value_string.length();

            if (value_string[0] != '{' && value_string[1] != '}') {
                //base case - leaf node
                leaf_node = true;
                data = parse_data(value_string);
                return;
            }

            int bracket_count = 0;
            std::string current = "";
            for (int i = 1; i < string_len - 1; i++) {
                char current_char = value_string[i];

                //update bracket stack pointer
                if (current_char == '[' || current_char == '{') {
                    bracket_count++;
                } else if (current_char == ']' || current_char == '}') {
                    bracket_count--;
                }

                if (current_char == ' ') {
                    continue;
                } else if (current_char == ',' && bracket_count == 0) {
                    add_node(current);
                    current = "";
                } else if (current_char == '}' && bracket_count == 0){
                    current += current_char;
                    add_node(current);
                    current = "";
                } else {
                    current += current_char;
                }
            }

            if (current != "" && current != " ") {add_node(current);}  //add the final child
        }

        void get_value_string(std::string* result) {
            //update the result string with the current node's data

            *result += "\"" + node_key + "\"" + " : ";

            if (leaf_node) {
                //no need to worry about nested dicts, just add the data
                *result += "[";

                for (int i = 0; i < data.size(); i++) {
                    std::string num = std::to_string(data[i]);

                    *result += num;

                    if (i == data.size() - 1) {
                        *result += "]";
                    } else {
                        *result += ", ";
                    }
                }

                return;
            } 

            *result += "{";

            //add children to output string
            int elements_added = 0;
            for (auto kv_pair : children) {
                kv_pair.second.get_value_string(result);  //let child add its own data

                elements_added++;

                if (elements_added < children.size()) {
                    *result += ", ";
                }
                
            }

            *result += "}";
        }

        void print() {
            //print attrs - useful for debugging
            std::cout << "Current node: " << node_key << "\n";
            std::cout << "Children: ";

            for (auto key : children) {
                std::cout << key.second.node_key << ",";
            }

            std::cout << "\nData: ";

            for (int i : data) {
                std::cout << i << ",";
            }

            std::cout << "\n";
        }

    private:
        bool leaf_node = false;
        std::vector<float> data;
        std::map<std::string, JsonTreeNode> children;
        std::string node_key;

        void add_node(std::string node_string) {
            if (node_string.length() == 0) {return;}  //this is for cases where there is a } followed by a ,

            std::string key = "";
            std::string value = "";
            bool seen_colon = false;

            //extract the key and value
            for (char current_char : node_string) {
                if (current_char == '\"') {
                    continue;
                } else if (current_char == ':' && !seen_colon) {
                    seen_colon = true;
                } else if (seen_colon) {
                    value += current_char;
                } else {
                    key += current_char;
                }
            }

            JsonTreeNode child(key);
            child.parse_value(value);

            children[key] = child;
        }

        std::vector<float> parse_data(std::string value_string) {
            //parse "[1,2,3,4]" into an actual list
            std::vector<float> data_list;

            std::string current;
            for (char i : value_string) {
                if (i == '[' || i == ']') {
                    continue;
                } else if (i == ',') {
                    data_list.push_back(std::stof(current));
                    current = "";
                } else {
                    current += i;
                }
            }

            if (current != "") {data_list.push_back(std::stof(current));}

            return data_list;
        }
};


class JsonTree {
    public:
        JsonTreeNode root_node;
        std::string filename;

        JsonTree(std::string fn) {
            filename = fn;
        }

        JsonTreeNode operator[] (std::string key) {
            return root_node[key];
        }

        void set_root_node(JsonTreeNode root) {
            root_node = root;
        }

        void build_tree_from_file() {
            std::string file_contents = read_file(filename);

            JsonTreeNode root("root");
            root.parse_value(file_contents);

            root_node = root;
        }

        void write() {
            std::string json_string = "";
            root_node.get_value_string(&json_string);

            std::string final_string = json_string.substr(9);  //remove the starting '"root" : '

            write_to_file(filename, final_string);
        }

    private:
        std::string read_file(std::string filename) {
            std::string file_contents;

            std::ifstream file(filename);  //open the file

            if (!file) {throw std::logic_error("Could not find file to open.");}

            while (!file.eof()) {
                std::string line;
                getline(file, line);  //read line by line
                file_contents += line;
            }

            file.close();
            
            return file_contents;
        }

        void write_to_file(std::string filename, std::string data) {
            std::ofstream file(filename);  //open the file

            if (!file) {throw std::logic_error("Could not find file to open.");}

            file << data;

            file.close();
        }
};


void send_pixel_data(std::vector<float> screen_pixels) {
    //build the tree
    JsonTree json_writer(send_filename);
    JsonTreeNode root_node("root");
    JsonTreeNode pixel_data("pixel_data");

    pixel_data.add_data(screen_pixels);
    root_node.add_child(pixel_data);

    json_writer.set_root_node(root_node);

    //write the data
    json_writer.write();
}