tc = int(input())
while tc > 0:
    n = int(input())
    siz = n
    p = list(map(int,input().split(' ')))
    if n == 1 or n == 2:
        print("YES")
        tc -= 1
        continue

    for i in range(n):
        if(p[i]==n):
            j = i
            break
    i = j

    siz-=1
    check = False
    while(siz>0):

        if(j>0 and siz == p[j-1] and siz>0):
            siz-=1
            j-=1
        elif (i+1<n and siz == p[i+1] and siz>0):
            siz-=1
            i+=1
        else:
            check = True
            break
    if check==False and siz==0 and j==0 and i==n-1:
        print("YES")
    else:
        print("NO")
    tc-=1

