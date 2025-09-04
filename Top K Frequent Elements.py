from collections import defaultdict
from typing import List


class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        list1 = defaultdict(int)
        for element in nums:
            list1[element]+=1
        list2 = list(list1.items())
        sortedList = sorted(list2, key=lambda x: x[1], reverse=True)
        ans = sortedList[:k]
        return [item[0] for item in ans]



