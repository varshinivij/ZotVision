import multiprocessing
import queue
import torch
import cv2
from transformer import CNNViTHybrid, predict
from image_difference import ImageDifference

LABELS = {0: "none", 1: "hazard", 2: "person", 3: "both"}


class FireFighterWorker:
    def __init__(self, model_path):
        self.model_path = model_path
        self.image_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(
            target=self.worker,
            args=(self.image_queue, self.result_queue, self.model_path),
        )
        self.process.start()

    def worker(self, image_queue, result_queue, model_path):
        """Worker: loads its own model and processes images as they arrive.
        Drains stale frames so inference always runs on the latest one."""
        model = CNNViTHybrid()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        while True:
            image_path = image_queue.get()
            if image_path is None:
                break

            # Drain any frames that piled up while we were busy — keep only latest
            while True:
                try:
                    newer = image_queue.get_nowait()
                    if newer is None:
                        return
                    image_path = newer
                except queue.Empty:
                    break

            label = predict(model, image_path)
            result_queue.put((image_path, label))

    def enqueue(self, image_path):
        """Non-blocking: push a frame into the FIFO queue and return immediately."""
        self.image_queue.put(image_path)

    def get_result(self):
        """Non-blocking: return the next result or None if nothing is ready yet."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.image_queue.put(None)
        self.process.join()


class FireFighterManager:
    def __init__(self, model_path="", num_firefighters=4):
        self.workers = [FireFighterWorker(model_path) for _ in range(num_firefighters)]
        self.images = [ImageDifference() for _ in range(num_firefighters)]

    def send_image(self, image_path, worker_id=0):
        """Non-blocking: enqueue frame for ML processing. Returns immediately."""
        self.workers[worker_id].enqueue(image_path)

    def get_result(self, worker_id=0):
        """Non-blocking: check if a worker has a result ready. Returns (image_path, output) or None."""
        return self.workers[worker_id].get_result()

    def stop_all(self):
        for worker in self.workers:
            worker.stop()
