import subprocess
import schedule
import time

def get_gpu_memory_map() -> str:
    """Get the current gpu usage.

    Return:
        
    """
    output = subprocess.check_output(["nvidia-smi"])
    if isinstance(output, bytes):
        output = output.decode('utf-8')

    return output
    

if __name__ == "__main__":
    '''
    parse nvidia-smi
    '''
    output = get_gpu_memory_map()
    print(output)