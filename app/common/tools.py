

def divisors(num: int):
    res = []
    for i in range(1, int(num/2)+1):
        if num % i == 0:
            res.append(i)
    return res

def divisors_tuple(num: int):
    res = []
    for i in range(1, int(num/2)+1):
        if num % i == 0:
            res.append(i)
    return tuple(res)
