# Distance Matrix to points

Given an approximate distance (symmetric) matrix, produce points in 2d-space for which the pairwise distance matrix matches the given one. This is useful in earthquake analysis, apparently. For a more in-depth explanation, please see [this](HowItWorks.ipynb) notebook.

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

You need a matrix in a space separated file. For example, say your matrix is in a file called `matrix.txt`. Then you could do:

```bash
python sismos.py matrix.txt
```

to use the default settings. The full set of options are described below. See [the bottom of the README](#example-usage) for an example.

```txt
usage: sismos.py [-h] [-t TIME] [-e ERROR] [-n NUM_RESTARTS] [-p POP_SIZE]
                 [-s SHUFFLES] [-d]
                 matrix

positional arguments:
  matrix                file with (symmetric) distance matrix

optional arguments:
  -h, --help            show this help message and exit
  -t TIME, --time TIME  Number of seconds to spend finding a good solution
                        (per restart) (default: 10)
  -e ERROR, --error ERROR
                        either use L1, L2, or smoothL1 (default: L2)
  -n NUM_RESTARTS, --num_restarts NUM_RESTARTS
                        Number of times that we try to restart (default: 5)
  -p POP_SIZE, --pop_size POP_SIZE
                        Population size for diff evo (default: 60)
  -s SHUFFLES, --shuffles SHUFFLES
                        Number of times populations mix. (default: 1)
  -d, --disable_cuda    If true, disable CUDA (even when available). If false,
                        use CUDA only if available. Ignored if you don't have
                        CUDA. (default: False)
```

### Example usage

A run could look like this:

```bash
python sismos.py matrix.dist.dat --time 60 --error L1 --num_restarts 8 --pop_size 100 --shuffles 2 --disable_cuda
```
The above command runs for 1 minute, uses L1 error instead of L2, and considers 8 populations, each of size 100. Furthermore, after 20 and 40 seconds, it mixes the 8 populations. CUDA is disabled.
