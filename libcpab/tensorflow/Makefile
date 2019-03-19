CPPFLAG = $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
LDFLAGS = $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CUDALIB = /usr/local/cuda-10.0/lib64/

# Build

transformer.so: transformer.cpp transformer.cu.o
	g++ -std=c++11 transformer.cpp transformer.cu.o -shared -o transformer.so  \
	-I $(CPPFLAG) -fPIC -O2 -lcudart -L $(CUDALIB) -L $(LDFLAGS) \
	-ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0 

transformer.cu.o: transformer_cuda.cu
	nvcc transformer_cuda.cu -std=c++11 -c -o transformer.cu.o -I $(CPPFLAG) \
	-D GOOGLE_CUDA=1 -x cu -arch compute_35 -Xcompiler -fPIC \
	--expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED

clean:
	rm transformer.cu.o 
	rm transformer.so