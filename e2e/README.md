# FasterTransformer(FlexQ Version)

[FasterTransformer](https://github.com/NVIDIA/FasterTransformer) provides a script and recipe to run the highly optimized transformer-based encoder and decoder component, and it is tested and maintained by NVIDIA. To evaluate end-to-end latency, this codebase is modified from [bytedance/ABQ-LLM/tree/main/fastertransformer](https://github.com/bytedance/ABQ-LLM/tree/main/fastertransformer).

Note that current codebase is for efficiency evaluation. We use random weights therefore no meaningful output.

## FasterTransformer E2E Evaluation

Please complete the FasterTransformer compilation (Make sure you install MPI):
```
cd ./FlexQ/e2e
bash build.sh
``` 

Modify the evaluation configuration:
```
# For LLaMA model, modify: e2e/examples/cpp/llama/llama_config.ini
# For OPT model, modify: e2e/examples/cpp/multi_gpu_gpt/gpt_config.ini

The following are the precision parameter settings for different baselines:
FP16:               int8_mode=0
W8A16 (CUTLASS):    int8_mode=1
W8A8 (SmoothQuant): int8_mode=2
W6Ax (FlexQ):       int8_mode=5
Additionally, for multi-GPU testing, you need to modify the tensor_para_size parameter (set it to the number of GPUs).
```

Run e2e efficiency evaluation:
```
cd build_release

# For single-GPU LLaMA model evaluation
./bin/llama_example

# For single-GPU OPT model evaluation
./bin/multi_gpu_gpt_example

# For multi-GPU evaluation
mpirun -n 2 ./bin/llama_example
```
