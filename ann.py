from chess import *


def test():
  for i in range(10, 1000, 10):
    chess("train", "chess1.png", i)


if __name__ == '__main__':
  test()