## About The Project

This project is an implementation of a fuzzy C-Means clustering algorithm by Python. <br>
Because of educational reasons, the algorithm is implemented without the use of Numpy.

## Usage

The project can run by using the following command:
```sh
python main.py [data set] [flags]
```
The `[data set]` argument is the data set's path, there are four data sets (`data1.csv`, ..., `data4.csv`) available in the main directory. <br>
The flags options:
* `-plotc`: plots the cost function for a range of ![formula](https://render.githubusercontent.com/render/math?math=C) values, it is assumed that ![formula](https://render.githubusercontent.com/render/math?math=m\=3).
* `-plotm`: plots the cost function for a range of ![formula](https://render.githubusercontent.com/render/math?math=m) values, it is assumed that ![formula](https://render.githubusercontent.com/render/math?math=C\=3).
* `-crisp`: by using this flag, coloring will be crisp instead of fuzzy.
* `-center`: cluster's centers will be shown by red points.
