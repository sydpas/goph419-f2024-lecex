from src.topic1.functions import exp

def main():
    x=0
    while x < 100:
        x = float(input("Please enter a number: "))
        print(f"exp({x}) = {exp(x)}")

if __name__ == "__main__":
    main()