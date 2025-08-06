if [ -d "build" ]; then
    rm -rf build
fi

mkdir build
cd build

cmake .. \
        -DSM=86 \
        -DCMAKE_BUILD_TYPE=Release 

make -j128