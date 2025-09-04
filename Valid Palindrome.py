from dataclasses import replace
from os import remove


class Solution:
    def isPalindrome(self, s: str) -> bool:
        s1 = s

        s1 = s1.replace(" ", "")
        s1 = s1[::-1]
        s = s.replace(" ", "")
        for i in s1:
            if not i.isalnum():
                s1 = s1.replace(i, "")

        for i in s:
            if not i.isalnum():
                s = s.replace(i, "")
        s = s.lower()
        s1 = s1.lower()
        return s1==s
solution = Solution()
print(solution.isPalindrome("aba"))