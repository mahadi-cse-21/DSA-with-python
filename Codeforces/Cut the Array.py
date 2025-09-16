from collections import defaultdict

tc = int(input())
for _ in range(tc):
    n = int(input())
    arr = list(map(int, input().split()))

    if n < 3:
        print(-1)
        continue

    # Compute prefix sums
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]

    total = prefix[n]
    ans = []

    # Try every pair (l, r) with l < r
    for l in range(1, n - 1):
        for r in range(l + 1, n):
            s1 = prefix[l] % 3
            s2 = (prefix[r] - prefix[l]) % 3
            s3 = (total - prefix[r]) % 3

            if (s1 == s2 == s3) or (len({s1, s2, s3}) == 3):
                ans = [l, r]
                break
        if ans:
            break

    if ans:
        print(ans[0], ans[1])
    else:
        print("0 0")
