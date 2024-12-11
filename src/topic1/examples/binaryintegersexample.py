from src.topic1.binaryintegers import BE_binary_int32
from src.topic1.binaryintegers import LE_binary_int32


def main():
    x = float(input("Please enter a number (0 to quit): "))
    while x != 0:
        print("Using the Big Endian approach, the binary of", x,"is:",BE_binary_int32(x))
        print("Using the Little Endian approach, the binary of", x,"is:",LE_binary_int32(x))
        x = float(input("Please enter a number (0 to quit): "))

if __name__ == "__main__":
    main()
