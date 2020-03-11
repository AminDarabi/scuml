scuml: scuml.cu Library/Matrix.cuh Library/Network.cuh Library/heads.cuh Library/scuml.cuh Library/utility.cuh
	nvcc -lcublas -lcurand scuml.cu -o scuml
install: scuml
	cp scuml /usr/local/bin/scuml