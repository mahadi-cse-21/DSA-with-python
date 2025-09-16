tc = int(input())
for _ in range(tc):
    n = int(input())
    arr = list(map(int, input().split()))
    arr2 = list(map(int, input().split()))

    count = 0
    # Check no swap case
    if arr == sorted(arr) and arr2 == sorted(arr2):
        count += 1

    # Check all single swaps only (i.e., swap one index only)
    for i in range(n):
        a = arr[:]
        b = arr2[:]
        a[i], b[i] = b[i], a[i]
        if a == sorted(a) and b == sorted(b):
            count += 1

    print(count)
