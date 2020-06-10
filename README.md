# Earthquake inversion

This is code for inverting earthquake data or something like that.

## Install

To install, you need pytorch and fastprogress. 

For example, with anaconda:
```
conda activate [my_env_name]
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install fastprogress -c fastai
```

or with pip:
```
pip install --user torch torchvision
pip install --user fastprogress 
```

## Usage

You need a matrix in a space separated file.

```
usage: sismos.py [-h] [-t TIME] [-n NUM_RESTARTS] [-s SCALE_FACTOR] [-e ERROR]
                 [-p POP_SIZE] [-c TRY_WITH_CUDA]
                 matrix

positional arguments:
  matrix                file with (symmetric) distance matrix

optional arguments:
  -h, --help            show this help message and exit
  -t TIME, --time TIME  Number of seconds to spend finding a good solution
                        (per restart) (default: 6)
  -n NUM_RESTARTS, --num_restarts NUM_RESTARTS
                        Number of times that we try to restart (default: 6)
  -s SCALE_FACTOR, --scale_factor SCALE_FACTOR
                        to avoid numerical instability (default: 50)
  -e ERROR, --error ERROR
                        either use l1, smoothl1 or l2 (default: l1)
  -p POP_SIZE, --pop_size POP_SIZE
                        Population size for diff evo (default: 50)
  -c TRY_WITH_CUDA, --try_with_cuda TRY_WITH_CUDA
                        Use cuda (if available) (default: false)
```
