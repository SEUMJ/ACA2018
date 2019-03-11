CC       := g++
LIB_NAME :=
CFLAGS   := -fPIC -O3 -g -Wall -fpermissive -std=c++0x
LDFLAGS  := -L../polaris/lib -L./mkl/lib \
            -lpolaris \
            -lpthread -lrt -lgomp
INC      := -I./ \
            -I../polaris/include
OBJ_DIR  := ./obj
OUT_DIR  := ./lib
BIN_DIR  := ./bin

all: $(patsubst main.cpp, $(BIN_DIR)/main, $(wildcard *.cpp))

$(BIN_DIR)/main: $(OBJ_DIR)/main.o $(OBJ_DIR)/preprocess.o
	$(CC) $(INC) $(CFLAGS)  -o $(BIN_DIR)/main $(OBJ_DIR)/main.o $(OBJ_DIR)/preprocess.o $(LDFLAGS)

$(OBJ_DIR)/%.o: %.cpp dirmake
	$(CC) -c $(INC) $(CFLAGS) -o $@ $<

dirmake:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(OBJ_DIR)

clean:
	rm -f $(OBJ_DIR)/*.o $(BIN_DIR)/* Makefile.bak

rebuild: clean build
