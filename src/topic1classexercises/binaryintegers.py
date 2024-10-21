#   big endian

def BE_binary_int32(x):
    p = 31
    d = []
    while p >= 0:
        d.append(1 if x>= 2**p else 0)
        x -= d[-1] * 2**p
        p -= 1
    return "".join(str(b) for b in d)

def LE_binary_int32(x):
    p=0
    d=[]
    while p < 32:
        d.append(x % 2)
        x //= 2
        p += 1
    return "".join(str(b) for b in d)

