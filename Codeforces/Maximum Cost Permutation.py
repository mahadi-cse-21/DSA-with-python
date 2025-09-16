tc = int(input())
for _ in range(tc):
    n = int(input())
    arr = list(map(int, input().split()))
    arr_set = set(arr)
    missing = [j for j in range(1, n + 1) if j not in arr_set]
    missing = list(reversed(missing))
    k = 0
    for j in range(1, n + 1):
        if arr[j - 1] == 0 and k < len(missing):
            arr[j - 1] = missing[k]
            k += 1
    min_index = n
    max_index = -1
    found = False
    for j in range(1, n + 1):
        if arr[j - 1] != j:
            if not found:
                min_index = j
                found = True
            max_index = j
    if not found:
        print(0)
    else:
        print(max_index - min_index + 1)
