import fcntl
import os
import socket
import torch
import torch.distributed as dist
import time 

MB = (1<<10) * 1e3
def printflock(*msgs):
    """ solves multi-process interleaved print problem """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

def test(nbytes,type):
  warmup = 5
  repeat = 25
  local_rank = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(local_rank)
  device = torch.device("cuda", local_rank)
  hostname = socket.gethostname()

  gpu = f"[{hostname}-{local_rank}]"

  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29500'

  dist.init_process_group(
      backend=dist.Backend.NCCL,
      init_method='env://',
      world_size=2,
      rank=local_rank
  )

  assert dist.is_initialized()
  assert dist.is_nccl_available()

  # global rank
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  # test all-reduce
  buf = torch.randn(nbytes // 4).to(device)

  torch.cuda.synchronize()
  # warmup
  for _ in range(warmup):
    if type == "a":
      dist.all_reduce(buf,op=dist.ReduceOp.SUM)
    elif type == "b":
      dist.broadcast(buf,src=0)
  torch.cuda.synchronize()

  dist.barrier()
  begin = time.perf_counter()
  for _ in range(repeat):
    if type == "a":
      dist.all_reduce(buf,op=dist.ReduceOp.SUM)
    elif type == "b":
      dist.broadcast(buf,src=0)
  torch.cuda.synchronize()
  end = time.perf_counter()
  dist.barrier()

  if rank == 0:
      printflock(f"pt={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}")
      printflock(f"device compute capabilities={torch.cuda.get_device_capability()}")
      printflock(f"pytorch compute capabilities={torch.cuda.get_arch_list()}")
      avg_time = (end - begin) * 1e6 / repeat
      alg_band = nbytes / MB / (end - begin)
      if type == "b": 
        bus_band = alg_band
      elif type == "a":
        bus_band = 2 * (world_size - 1) / world_size * alg_band
      printflock(f"{gpu}, time {round(avg_time,2)} us, Bus bandwidth {round(bus_band,2)} MB/s")

if __name__ == "__main__":
  test(int((1<<10) * MB),"a")