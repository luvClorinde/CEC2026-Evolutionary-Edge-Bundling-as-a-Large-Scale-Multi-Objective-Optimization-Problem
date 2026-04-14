# Edge Bundling via Evolutionary Computation: Algorithm Comparison Framework

This repository provides an experimental framework for comparing evolutionary algorithms in the context of edge bundling.

## Overview

The framework enables systematic comparison of multiple algorithms under a unified setting for edge bundling tasks. Users can control experimental conditions such as the number of runs, graph datasets, iterations, and population size.

## Available Graphs

The following graphs are pre-configured:

* `1`
* `jpair`
* `usair`

## Adding New Graphs

To add a new graph:

1. Modify the files in `util/testgraphs`
2. Load the new graph data in the same manner as existing graphs

## Running Experiments

### Run All Algorithms

python runner.py

You can configure:

* Number of runs
* Graph dataset
* Number of iterations
* Population size

All algorithms will be executed under the same conditions.

### Run Individual Algorithms

python <algorithm_file>.py

You can directly modify each algorithm file to tune detailed parameters.

## Requirements and Notes

* This implementation assumes NVIDIA GPU-based parallel computation
* A compatible CUDA Toolkit must be installed in advance

The `requirements.txt` includes:

cupy-cuda12x

However, you must install the version that matches your environment:

* CUDA 11.x → cupy-cuda11x
* CUDA 12.x → cupy-cuda12x

If the versions do not match, the code will not run correctly.

## Summary

* Unified framework for fair algorithm comparison
* Flexible experimental configuration
* GPU acceleration required
* Easily extensible with new graph datasets
