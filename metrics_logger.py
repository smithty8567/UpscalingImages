import os

class MetricLogger:
  def __init__(self, csv_path, metric_names):
    self.csv_path = csv_path
    self.make_header(metric_names)
    self.to_write = []

  def make_header(self, metric_names):
    self.metrics = {name: 0 for name in metric_names}
    if os.path.exists(self.csv_path):
      return
    with open(self.csv_path, 'w') as f:
      f.write(','.join(metric_names) + '\n')

  def log_metric(self, name, value):
    if not os.path.exists(self.csv_path):
      raise Exception("CSV file does not exist. Call make_header first.")
    if name not in self.metrics:
      raise Exception(f"Metric {name} not in header.")
    self.metrics[name] = value

  def append_to_write(self):
    try:
      with open(self.csv_path, 'a') as f:
        f.writelines(self.to_write)
      self.to_write = []
    except:
      return

  def next_iter(self):
    self.to_write += [','.join(str(self.metrics[name]) for name in self.metrics) + '\n']
    self.append_to_write()
