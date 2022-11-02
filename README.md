# nccl-tests
This is the test for multiGPUs communication

# Dependency
```
pytorch
```

# Run
`OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3,4,5,7 python -m torch.distributed.run --nproc_per_node=4 --master_port 11000 latency_test.py`
