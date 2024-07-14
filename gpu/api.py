import json


SEND_FILENAME = "gpu/to_device.json"
RECIEVE_FILENAME = "gpu/to_host.json"


class JsonData:
    def __init__(self, json_filename):
        self.json_filename = json_filename

        self.json_string = ""
        self.data_dictionary = {}

    def load_from_file(self):
        self.json_string = read_file(self.json_filename)
        self.data_dictionary = json.loads(self.json_string)

    def load_from_dict(self, data_dictionary):
        self.data_dictionary = data_dictionary
        self.json_string = json.dumps(self.data_dictionary)

    def write_to_file(self):
        write_to_file(self.json_filename, self.json_string)


def read_file(filename):
    with open(filename, "r") as file:
        data = file.read()

    return data


def write_to_file(filename, string):
    with open(filename, "w") as file:
        file.write(string)


def send_to_cuda(data_dictionary):
    data = JsonData(SEND_FILENAME)

    data.load_from_dict(data_dictionary)
    data.write_to_file()


def recieve_from_cuda():
    data = JsonData(RECIEVE_FILENAME)

    data.load_from_file()

    return data.data_dictionary


def clear_files():
    #discard the results of any past renders (to be called on startup)
    write_to_file(SEND_FILENAME, "")
    write_to_file(RECIEVE_FILENAME, "")