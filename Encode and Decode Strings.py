from typing import List


class Solution:

    def encode(self, strs: List[str]) -> str:
        list1 = ""
        for word in strs:
            newword = "ji+"+word

            list1+=newword
        return  list1

    def decode(self, s: str) -> List[str]:
        if s=="ji+": return [""]

        list2 = s.split("ji+")
        list1 =[]
        for word in list2:
              list1.append(word)
        list1.remove("")
        return list1
solution = Solution()
s  =  solution.encode(["","eat"])
print(solution.encode(["","eat"]))
print(solution.decode(s))
