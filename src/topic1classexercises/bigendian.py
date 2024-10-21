b = 2
x = 173
p = 31
binary = []

while p >= 0 and x >= 0:
    if x >= (b**p):
        d = 1
    else:
        d = 0
    binary.append(d)
    x = x - ((b**p) * d)
    p = p - 1
print(binary)