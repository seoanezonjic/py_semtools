from concurrent.futures import ProcessPoolExecutor
from os import getpid
from loguru import logger
logger.remove(0)

class Parallelizer:
    def __init__(self, n_processes, chunk_size):
        self.n_processes = n_processes
        self.chunk_size = chunk_size

    def get_chunks(self, items):
        chunks = []
        if self.chunk_size == 0: # all workload is given to all cpus in a single working round
            chunk_size = len(items)// self.n_processes
            for index in range(0, self.n_processes):
                chunk = []
                for index in range(0, chunk_size):
                    chunk.append(items.pop())
                chunks.append(chunk)
            for i, item in enumerate(items):
                chunks[i].append(item)
            chunks = [ chunk for chunk in chunks if chunk] # remove possible empty chunks
        else: # The workload is partitioned in several fixed size chunks and each worker will execute more than one chunk
            while len(items) > 0:
                chunk = []
                for index in range(0, self.chunk_size):
                    if items: chunk.append(items.pop())
                chunks.append(chunk)
        return chunks
    
    def execute(self, items):
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            for result in executor.map(self.worker, items): return result

    def worker(self, arguments):
        task, all_args = arguments
        args, kwargs = all_args
        pID = getpid()
        logger.add(f"./logs/{pID}.log", format="{level} : {time} : {message}: {process}", filter=lambda record: record["extra"]["task"] == f"{pID}")
        child_logger = logger.bind(task=f"{pID}")
        child_logger.info("Starting chunk process")
        kwargs['logger'] = child_logger
       
        task(*args, **kwargs)         

        child_logger.success("Chunk finished succesfully")