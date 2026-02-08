# Conway's game of life in CUDA

mickael@mistral.ai

on peut utiliser les varaible comme des flag ou on lit chaque octect comme 8 cellules

## What is this all about?

The [Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) is a cellular automaton devised by the British mathematician John Horton Conway in 1970. It is a zero-player game, meaning that its evolution is determined by its initial state, requiring no further input. One interacts with the Game of Life by creating an initial configuration and observing how it evolves.

The Game of Life is played on a grid of square cells, each of which can be in one of two possible states, alive or dead, represented by a 1 or 0, respectively. The state of each cell evolves in discrete time steps as determined by the following rules:

    We consider that a cell has two states: dead or alive and eight neighbors: top, top-right, right, bottom-right, bottom, bottom-left, left, top-left.
    Any live cell with fewer than two live neighbors dies (underpopulation).
    Any live cell with two or three live neighbors lives on to the next generation.
    Any live cell with more than three live neighbors dies (overpopulation).
    Any dead cell with exactly three live neighbors becomes a live cell (reproduction).

These rules lead to a wide variety of patterns and structures, some of which are stable, others that oscillate, and still others that move across the grid.

In our implementation, we will use a 2D grid of size `NxN` and we will use zero-border conditions: neighbors of cells on the border of the grid are considered dead.

## Project structure

The overall project is organized as having a python layer interacting with C++/CUDA, thanks to torch bindings.
The project is structured as follows:

    .
    ├── CMakeLists.txt
    ├── conway.cpp
    ├── conway.cu
    ├── conway.h
    ├── conway.py
    ├── grids # contain some starting grids
    ├── README.md
    └── requirements.txt

## How to build

Create a virtual environment and install the requirements:

``` bash
    python -m venv conway_env
    source conway_env/bin/activate
    pip install -r requirements.txt
```

The core of the project is built using CMake. The following commands will build the project:

``` bash
    cmake -S . -B build -G Ninja # configures the project for building
    cmake --build build # actually builds the project
    # /!\ don't forget to re-build whenever you modify the cpp/cuda code /!\
```

## How to run & test

The project can be tested using the following command:

``` bash
    # test if your program is correct
    python conway.py test

    # show the game of life animation (you can specify a starting grid with --file grids/<path_to_file>)
    # with your implementation (make sure your program pass the test first)
    python conway.py show [--file grids/<path_to_file>]

    # check the speed of your implementation
    python conway.py profile
```

## What to do?

The end-goal is to get the fastest implementation of the game of life, as judged by the number of FPS reported by `python conway.py profile --grid-size 4000`. You can modify the following files `conway.cu`.

First, complete the initial TODOs in conway.cu and make sure your code matches the reference implementation. You can use `python conway.py test` to check if your implementation is correct.

Then, you have to find ways to make the program faster! Don't hesitate to use the profiling tools:
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html): `nsys profile -t cuda -s none python conway.py profile --grid-size 4000 --iterations 1` will produce a `.nsys-rep` file that you can open with Nsight Systems.
- [Nsight Compute](https://docs.nvidia.com/nsight-compute): `ncu -o kernel_profile --set full -s 3 -c 1 -k game_of_life_kernel python conway.py profile --grid-size 4000 --iterations 1` will produce a `.ncu-rep` file that you can open with Nsight Compute.

You and your team have to write a report of the different optimizations you tried, **why** you tried them, and the results you obtained.
