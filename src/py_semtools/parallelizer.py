import sys
#from concurrent.futures import ProcessPoolExecutor
from loky import get_reusable_executor
from os import getpid
import inspect
from loguru import logger
logger.remove(0)

class Parallelizer:
    def __init__(self, n_processes, chunk_size):
        self.n_processes = n_processes
        self.chunk_size = chunk_size
        self.workers_logger = {}

    def get_chunks(self, items, workload_balance = None, workload_function = None):
        one_worker_round = False
        if self.chunk_size == 0: # all workload is given to all cpus in a single working round
            one_worker_round = True
            self.chunk_size = len(items)// self.n_processes

        if workload_balance != None: items = self.balance_workload(items, workload_balance, workload_function)

        chunks = []
        if one_worker_round: # all workload is given to all cpus in a single working round
            for index in range(0, self.n_processes):
                chunk = []
                for index in range(0, self.chunk_size):
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

    def balance_workload(self, items, workload_balance, workload_function):
        items.sort(key=workload_function)
        if workload_balance == 'min_max':
            items = self.balance_min_max(items)
        elif workload_balance == 'disperse_max':
            items = self.balance_disperse_max(items)
        items.reverse()
        return items

    def balance_min_max(self, items):
        item_workload = []
        while len(items) > 0:
            item_workload.append(items.pop(0)) # Take max workload
            if len(items) > 0: item_workload.append(items.pop()) # Take min workload
        return item_workload

    def balance_disperse_max(self, items):
        item_workload = []
        n_chunks = (len(items) // self.chunk_size)
        if len(items) % self.chunk_size > 0: n_chunks += 1
        chunks = [[] for i in range(n_chunks)]
        while len(items) > 0:
            for n in range(0, n_chunks):
                if len(items) > 0: chunks[n].append(items.pop()) 
                # The top i items with maximum workload is given to different workers 
                # to avoid that a single worker collapse with the top workload items
        for chunk in chunks: item_workload.extend(chunk)
        return item_workload

    def worker(self, arguments):
        task, all_args = arguments
        args, kwargs = all_args
        pID = getpid()
        if self.workers_logger.get(pID) == None: 
            logger.add(f"./logs/{pID}.log", format="{level} : {time} : {message}: {process}", filter=lambda record: record["extra"]["task"] == f"{pID}")
            child_logger = logger.bind(task=f"{pID}")
            child_logger.info("Starting chunk process")
            self.workers_logger[pID] = child_logger
        else:
            child_logger = self.workers_logger[pID]

        if 'logger' in inspect.getfullargspec(task).args:
            kwargs['logger'] = child_logger # check if method arguments include the logger argument to pass the object
        res = task(*args, **kwargs)         
        child_logger.success("Chunk finished succesfully")
        return res
    
    def execute(self, items):
        executor = get_reusable_executor(max_workers=self.n_processes)
        results = executor.map(self.worker, items)
        return results
