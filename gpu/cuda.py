import subprocess


class CudaScript:
    def __init__(self, source_code_path, needs_compile):
        self.source_path = source_code_path
        self.script_name = source_code_path.split(".")[0]

        self.process = self.spawn_process(needs_compile)

    def compile_code(self):
        compile_command = ["nvcc", self.source_path, "-o", self.script_name]
        compile_result = subprocess.run(compile_command)

        if compile_result.returncode != 0:
            raise Exception(f"CUDA script compilation error: {compile_result.stderr}")

    def spawn_process(self, needs_compile):
        if needs_compile:
            self.compile_code()

        process = subprocess.Popen([f"./{self.script_name}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

        return process
    
    def run(self, block):
        #assumes data is already in the json file
        self.process.stdin.write("render\n")
        self.process.stdin.flush()

        #if needed, block execution until rendering has finished
        output = ""
        safety = 0
        while block and output != "render_complete":
            safety += 1
            output = self.process.stdout.readline().strip()

            if safety > 100:
                self.kill_process()
                raise Exception("Safety threshold reached when communicating with process")

    def kill_process(self):
        self.process.kill()