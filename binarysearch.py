A  = [1,1,2,2,3,4,5,10,20,100,400,400,400]


def binary_search(A, x):
    L = 0
    N = len(A)
    R = N-1

    while L<R:
        M = L+((R-L)//2)
        if A[M]==x:
            R = M
        elif x<A[M]:
            R = M-1
        elif x>A[M]:
            L=M+1

    return L
print(binary_search(A, 7))
