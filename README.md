# Git Mock Assignment (OOP for AI 2024-25)

In this assignment, you are required to create a very simple Python script that you'll need to submit, having a sequence of automatic tests (concerning mainly style and libraries) pass.

## Your code

The code is a very simple implementation of a numpy array.
Write it in a file `main.py`. In this script...

* You should create a list containing all the numbers from 1 to 10.
* A numpy array should be created from this list
* The numpy array should be printed, without adding any other text

## The specs

* You should use a virtualenv/conda environment for this project
* You should install specifically numpy **version 1.26.1**

## The files

In addition to the `main.py` file, you should also provide a `requirements.txt` file in which you provide the list of libraries needed to run your code.
The file has to be filled in using the [pip requirements standard](https://pip.pypa.io/en/stable/reference/requirements-file-format/).

## The tests

The automatic tests are set up inside the folder `.github/workflows`, which you should not touch or tamper with.
The tests will be checking:

1. That the script `main.py` abides by the Python `flake8` style requirements
2. That the script `main.py` runs successfully and print exactly only the content of the array
3. That you use the relevant version of Numpy, as indicated above

## Git

1. Create your own version of the repository **using the appropriate "Use this template" button** in the upper-right part of the GitHub interface
2. Clone the repo locally
3. Implement your code
4. Populate the `requirements.txt` file appropriately, as indicated above
5. Remember to keep track of the various operations by committing frequently
   * Use meaningful commit messages
6. Push everything
