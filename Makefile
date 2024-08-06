main : main.o
	nvcc main.o -o main -lsfml-graphics -lsfml-window -lsfml-system

main.o : main.cu
	nvcc -c ./main.cu