from chess import *


def test():
  for i in range(10, 150, 10):
    print i
    # chess("train", "chess1.png", i)
    # chess("train", "chess6.png", i)
    chess("train", "chess11.png", i)
    print("\n")


if __name__ == '__main__':
  test()