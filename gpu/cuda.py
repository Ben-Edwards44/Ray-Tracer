import subprocess


class CudaScript:
    def __init__(self, source_code_path, needs_compile):
        self.source_path = source_code_path
        self.script_name = source_code_path.split(".")[0]

        if needs_compile:
            self.compile_code()

    def compile_code(self):
        compile_command = ["nvcc", self.source_path, "-o", self.script_name]
        compile_result = subprocess.run(compile_command)

        if compile_result.returncode != 0:
            raise Exception(f"CUDA script compilation error: {compile_result.stderr}")

    def run(self):
        run_command = [f"./{self.script_name}"]
        result = subprocess.run(run_command)

        if result.returncode != 0:
            raise Exception(f"CUDA script runtime error: {result.stderr}")