class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if len(s)<len(t): return ""
        ans = ""
        for i  in range(len(s)):

            for j in range(len(s)):
                test = s[i:j+1]

                tmp = 0
                for k in range(len(t)):
                    if test.find(t[k])==-1:
                        tmp=1
                        break
                if(tmp==0):
                    if ans=="" and len(test)>=len(t):
                        ans = test
                    elif len(test)<len(ans) :
                        ans = test
        return ans
sol = Solution()
print(sol.minWindow("ADOBECODEBANC", "ABC"))
