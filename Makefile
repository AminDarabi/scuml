scuml: scuml.cu
	nvcc -lcublas -lcurand scuml.cu -o scuml
install: scuml
	cp scuml /usr/local/bin/scuml