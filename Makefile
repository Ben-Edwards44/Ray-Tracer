main : main.o
	nvcc main.o -o main -lsfml-graphics -lsfml-window -lsfml-system

main.o : src/main.cu
	nvcc -c ./src/main.cu -o main.o