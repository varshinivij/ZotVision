import multiprocessing
import sys
import time
import torch
import PredictHazard
import uuid

class FireFighterWorker:  
    def __init__(self, model_path):
        self.image_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        while not self.image_queue.empty():
            self.process = multiprocessing.Process(
                target=self.worker,
                args=(self.image_queue, self.result_queue, self.model_path),
            )
            self.process.start()


    def get_image_queue(self):
        return self.image_queue

    def get_result_queue(self):
        return self.result_queue
    
    def image_id(self, image_path):
        return uuid.uuid5(uuid.NAMESPACE_URL, image_path)


    def worker(self, image_queue, result_queue, model_path):
        """Worker: loads its own model and processes images as they arrive."""
        model = PredictHazard()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        while True:
            image_path = image_queue.get()  # blocks until an image is available
            if image_path is None:  
                break
            with torch.no_grad():
                output = model(image_path)
            result_queue.put((image_path, output))
        
        return output

    
class FireFighterManager:
    def __init__(self, model_path, num_firefighters=4):
        self.workers = [FireFighterWorker(model_path) for _ in range(num_firefighters)]

    def send_image(self, image_path, worker_id=0):
        # Simple round-robin distribution
        worker = self.workers[worker_id]  # Use the specified worker
        id_number = worker.get_image_queue().put(image_path)

    def collect_results(self, worker_id=0):
        worker = self.workers[worker_id]
        return worker.get_result_queue().get()