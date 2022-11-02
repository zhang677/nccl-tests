import fcntl
import socket
import torch
import torch.distributed as dist
import time
import math 

MB = int((1<<10) * 1e3)
GB = int((1<<20) * 1e3)
Byte = 4
def printflock(*msgs):
    """ solves multi-process interleaved print problem """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

def store(time,bandwidth):
  f = open("tmp.txt","w")
  f.write(str(time)+','+str(bandwidth))
  f.close()

def load():
  f = open("tmp.txt","r")
  ln = f.readline().split(",")
  f.close()
  return (float(ln[0]),float(ln[1]))

def test(wsize,nbytes,type):
  warmup = 5
  repeat = 25
  rank = dist.get_rank()
  torch.cuda.set_device(rank)
  device = torch.device("cuda", rank)
  hostname = socket.gethostname()

  gpu = f"[{hostname}-{rank}]"
  
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
    avg_time_s = (end - begin) / repeat
    alg_band = nbytes / avg_time_s
    if type == "b": 
      bus_band = alg_band
    elif type == "a":
      bus_band = 2 * (wsize - 1) / wsize * alg_band
    store(avg_time_s,alg_band)
    printflock(f"{gpu}, Bytes: {nbytes} B,Time: {round(avg_time_s * 1e6,2)} us, Bus bandwidth: {round(bus_band / GB,2)} GB/s")
    return (avg_time_s,alg_band)

def test_latency(wsize,it=3,type="a"):
  latency = 0
  for i in range(it):
    nbytes = int(Byte << i)
    test(wsize,nbytes,type)
    dist.barrier()
    (t,_) = load()
    latency += t
  return latency / it

def test_bandwidth(wsize,maxbytes,type="a"):
  test(wsize,maxbytes,type)
  dist.barrier()
  (_, bandwidth) = load()
  return bandwidth

def test_ab(wsize,type="a"):
  assert torch.cuda.is_available()
  assert torch.cuda.device_count() == wsize

  dist.init_process_group(
      backend=dist.Backend.NCCL,
      init_method='env://',
      world_size=wsize,
  )

  device = torch.device("cuda",dist.get_rank())
  max_nbytes = torch.tensor(torch.cuda.mem_get_info(device)[0]).to(device)
  dist.all_reduce(max_nbytes,op=dist.ReduceOp.MIN)
  max_nbytes = min(int (4 * GB), int (GB << int(math.log2(max_nbytes.item()/GB))))
  if dist.get_rank() == 0:
    printflock(f"max_nbytes: {max_nbytes} B")
  
  alpha = test_latency(wsize)
  beta = 1 / test_bandwidth(wsize, max_nbytes)
  dist.barrier()

  return (alpha,beta)
  


if __name__ == "__main__":
  (alpha,beta) = test_ab(4,"b")
  if dist.get_rank() == 0:
    store(alpha,beta)
    printflock(f"alpha(us): {round(alpha * 1e6,2)}, beta(us/GB): {round(beta * 1e6 * GB,2)}")