t = int(input())
for _ in range(t):
    n = int(input())
    s = input()

    if n % 2 != 0:
        print(-1)
    else:
        # Just print n//2 '(' and n//2 ')'
        print('(' * (n // 2) + ')' * (n // 2))
