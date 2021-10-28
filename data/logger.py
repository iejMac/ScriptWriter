import os
import pickle

LOG_DIR = "logs"

class Logger:
  def __init__(self, metrics):
    self.metrics = dict()
    for metric in metrics:
      self.metrics[metric] = []
    if not os.path.exists(LOG_DIR):
      os.mkdir(LOG_DIR)

  def log(self, val, metric):
    self.metrics[metric].append(val)
    pickle.dump(self.metrics[metric], open(os.path.join(LOG_DIR, f"{metric}.pkl"), "wb"))
