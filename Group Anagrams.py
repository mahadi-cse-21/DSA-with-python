from typing import List
from collections import defaultdict

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        list1 = defaultdict(list)
        list2 = [[]]
        for single in strs:
            sorted_word = ''.join(sorted(single))
            list1[sorted_word].append(single)
        list2 = list(list1.values())
        return list2
solve = Solution()
ans = solve.groupAnagrams(["eat", "tea", "tan", "ate", "nat"])
print(ans)
