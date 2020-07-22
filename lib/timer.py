import time


class AverageMeter(object):
  """Computes and stores the average and current value"""

  # def __init__(self, name, fmt=':f'):
  #   self.name = name
  #   self.fmt = fmt
  #   self.reset()

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0.0
    self.sq_sum = 0.0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    self.sq_sum += val**2 * n
    self.var = self.sq_sum / self.count - self.avg ** 2

  # def __str__(self):
  #     fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
  #     return fmtstr.format(**self.__dict__)


class Timer(object):
  """A simple timer."""

  def __init__(self):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.avg = 0.

  def reset(self):
    self.total_time = 0
    self.calls = 0
    self.start_time = 0
    self.diff = 0
    self.avg = 0

  def tic(self):
    # using time.time instead of time.clock because time time.clock
    # does not normalize for multithreading
    self.start_time = time.time()

  def toc(self, average=True):
    self.diff = time.time() - self.start_time
    self.total_time += self.diff
    self.calls += 1
    self.avg = self.total_time / self.calls
    if average:
      return self.avg
    else:
      return self.diff


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
