import psutil
import os

process = psutil.Process(os.getpid())
mem_info = process.memory_info()

print(f"Current RAM usage: {mem_info.rss / (1024 ** 2):.2f} MB")
