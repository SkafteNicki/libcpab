CPPFLAG = $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
LDFLAGS = $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CUDALIB = /usr/local/cuda-10.0/lib64/

# Build

transformer.so: transformer.cpp transformer.cu.o cpab_ops.c.o cpab_ops.cu.o
	g++ -std=c++11 transformer.cpp transformer.cu.o cpab_ops.c.o cpab_ops.cu.o \
	-shared -o transformer.so -I $(CPPFLAG) -fPIC -O2 -lcudart -L $(CUDALIB) \
	-L $(LDFLAGS) -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0 -Wl,--no-undefined

transformer.cu.o: transformer_cuda.cu
	nvcc transformer_cuda.cu -std=c++11 -c -o transformer.cu.o -I $(CPPFLAG) \
	-D GOOGLE_CUDA=1 -x cu -arch compute_35 -Xcompiler -fPIC \
	--expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED

cpab_ops.c.o: $(realpath ../core/cpab_ops.cpp)
	g++ -std=c++11 $(realpath ../core/cpab_ops.cpp) -shared -o cpab_ops.c.o  \
	-I $(CPPFLAG) -fPIC -O2 -lcudart -L $(CUDALIB) -L $(LDFLAGS) \
	-ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0 

cpab_ops.cu.o: $(realpath ../core/cpab_ops.cu)
	nvcc $(realpath ../core/cpab_ops.cu) -std=c++11 -c -o cpab_ops.cu.o -I $(CPPFLAG) \
	-D GOOGLE_CUDA=1 -x cu -arch compute_35 -Xcompiler -fPIC \
	--expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED

clean:
	rm cpab_ops.cu.o
	rm cpab_ops.c.o
	rm transformer.cu.o 
	rm transformer.so