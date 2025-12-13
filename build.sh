#!/bin/sh

set -xe

# gcc nn_sorter.c nn.c -Wall -Wextra -Ofast -lpthread -o nn_sorter -lm
gcc gui_nn_sorter.c nn.c -o gui_nn_sorter -Wall -Wextra -O3 -lraylib -lm -lpthread -D_DEFAULT_SOURCE
