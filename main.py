from examples.FNN_example_run import FNN_example_run
from examples.CNN_example_run import CNN_example_run

def main():
    FNN_example_run(0.001, 32, 5, False)
    CNN_example_run(0.001, 32, 5, False)
if __name__ == "__main__":
    main()