# if [ -d "cublas_results" ]; then
#     rm -rf cublas_results
# fi

mkdir -p cublas_results

BS="1 2 4 8"
for M in $BS; do
    name="llama_7b"
    ./bin/test_cublas_kernel ${M} 12288 4096 > ./cublas_results/${name}_${M}x12288x4096_w6a6.txt
    ./bin/test_cublas_kernel ${M} 4096 4096 > ./cublas_results/${name}_${M}x4096x4096_w6a6.txt
    ./bin/test_cublas_kernel ${M} 11008 4096 > ./cublas_results/${name}_${M}x11008x4096_w6a6.txt
    ./bin/test_cublas_kernel ${M} 4096 11008 > ./cublas_results/${name}_${M}x4096x11008_w6a8.txt
    name="llama_30b"
    ./bin/test_cublas_kernel ${M} 19968 6656 > ./cublas_results/${name}_${M}x19968x6656_w6a6.txt
    ./bin/test_cublas_kernel ${M} 6656 6656 > ./cublas_results/${name}_${M}x6656x6656_w6a6.txt
    ./bin/test_cublas_kernel ${M} 17920 6656 > ./cublas_results/${name}_${M}x17920x6656_w6a6.txt
    ./bin/test_cublas_kernel ${M} 6656 17920 > ./cublas_results/${name}_${M}x6656x17920_w6a8.txt
    name="llama_2_13b"
    ./bin/test_cublas_kernel ${M} 15360 5120 > ./cublas_results/${name}_${M}x15360x5120_w6a6.txt
    ./bin/test_cublas_kernel ${M} 5120 5120 > ./cublas_results/${name}_${M}x5120x5120_w6a6.txt
    ./bin/test_cublas_kernel ${M} 13824 5120 > ./cublas_results/${name}_${M}x13824x5120_w6a6.txt
    ./bin/test_cublas_kernel ${M} 5120 13824 > ./cublas_results/${name}_${M}x5120x13824_w6a8.txt
    name="llama_2_70b"
    ./bin/test_cublas_kernel ${M} 24576 8192 > ./cublas_results/${name}_${M}x24576x8192_w6a6.txt
    ./bin/test_cublas_kernel ${M} 8192 8192 > ./cublas_results/${name}_${M}x8192x8192_w6a6.txt
    ./bin/test_cublas_kernel ${M} 28672 8192 > ./cublas_results/${name}_${M}x28672x8192_w6a6.txt
    ./bin/test_cublas_kernel ${M} 8192 28672 > ./cublas_results/${name}_${M}x8192x28672_w6a8.txt
    name="opt_30b"
    ./bin/test_cublas_kernel ${M} 21504 7168 > ./cublas_results/${name}_${M}x21504x7168_w6a6.txt
    ./bin/test_cublas_kernel ${M} 7168 7168 > ./cublas_results/${name}_${M}x7168x7168_w6a6.txt
    ./bin/test_cublas_kernel ${M} 28672 7168 > ./cublas_results/${name}_${M}x28672x7168_w6a6.txt
    ./bin/test_cublas_kernel ${M} 7168 28672 > ./cublas_results/${name}_${M}x7168x28672_w6a8.txt
    # name="opt_175b"
    # ./bin/test_cublas_kernel ${M} 36864 12288 > ./cublas_results/${name}_${M}x36864x12288_w6a6.txt
    # ./bin/test_cublas_kernel ${M} 12288 12288 > ./cublas_results/${name}_${M}x12288x12288_w6a6.txt
    # ./bin/test_cublas_kernel ${M} 49152 12288 > ./cublas_results/${name}_${M}x49152x12288_w6a6.txt
    # ./bin/test_cublas_kernel ${M} 12288 49152 > ./cublas_results/${name}_${M}x12288x49152_w6a8.txt
done
