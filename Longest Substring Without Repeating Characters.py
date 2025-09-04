class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if s==" ": return 1
        ans = 0
        for i in range(len(s)):
            newstring = ""
            for j in range(i, len(s)):
                if s[j] not in newstring:
                    newstring = newstring + s[j]
                else:
                    ans = max(ans, j-i)
                    print(newstring)
                    newstring = ""
                    break
            if newstring:
                ans = max(ans, len(s)-i)
        return ans