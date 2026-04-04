import multiprocessing
import time
import torch
import PredictHazard


def worker(image_queue, result_queue, model_path):
    """Worker: loads its own model and processes images as they arrive."""
    model = PredictHazard()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    while True:
        image_path = image_queue.get()  # blocks until an image is available
        if image_path is None:  # poison pill to shut down
            break
        with torch.no_grad():
            output = model(image_path)
        result_queue.put((image_path, output))


if __name__ == '__main__':
    image_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Start worker process (each loads its own model copy)
    p = multiprocessing.Process(
        target=worker,
        args=(image_queue, result_queue, 'model_weights.pth'),
    )
    p.start()

    # --- Example: feed images as they arrive ---
    # In practice, replace this with however images enter your system
    # (e.g., HTTP endpoint, file watcher, etc.)
    #
    # image_queue.put("path/to/image.jpg")

    # --- Example: read results ---
    # while True:
    #     image_path, output = result_queue.get()
    #     print(f"Result for {image_path}: {output}")

    # To shut down the worker cleanly:
    # image_queue.put(None)
    # p.join()
