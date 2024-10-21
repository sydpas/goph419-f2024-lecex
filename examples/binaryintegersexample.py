from src.topic1classexercises.binaryintegers import BE_binary_int32
from src.topic1classexercises.binaryintegers import LE_binary_int32


def main():
    x = float(input("Please enter a number: "))
    print("Using the Big Endian approach, the binary of", x,"is:",BE_binary_int32(x))
    print("Using the Little Endian approach, the binary of", x,"is:",LE_binary_int32(x))

if __name__ == "__main__":
    main()
