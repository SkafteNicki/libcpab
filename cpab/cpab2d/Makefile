CPPFLAG = $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
LDFLAGS = $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CUDALIB = /usr/local/cuda-9.0/lib64/

# Build

CPAB_ops.so: CPAB_ops.cc CPAB_ops.cu.o
	g++ -std=c++11 CPAB_ops.cc CPAB_ops.cu.o -shared -o CPAB_ops.so  \
	-I $(CPPFLAG) -fPIC -O2 -lcudart -L $(CUDALIB) -L $(LDFLAGS) \
	-ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0 

CPAB_ops.cu.o: CPAB_ops.cu.cc
	nvcc CPAB_ops.cu.cc -std=c++11 -c -o CPAB_ops.cu.o -I $(CPPFLAG) \
	-D GOOGLE_CUDA=1 -x cu -arch compute_35 -Xcompiler -fPIC \
	--expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED

clean:
	rm CPAB_ops.cu.o 
	rm CPAB_ops.so

