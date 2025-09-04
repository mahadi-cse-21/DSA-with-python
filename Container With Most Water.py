from typing import List


class Solution:
    def maxArea(self, heights: List[int]) -> int:
        max = 0
        for i in range(len(heights)-1):
            for j in range(i+1, len(heights)):
                min_height = min(heights[i],heights[j])
                if(min_height *(j-i) > max):
                    max = min_height*(j-i)
        return max
solu = Solution()
print(solu.maxArea([1,7,2,5,4,7,3,6]))