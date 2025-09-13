from typing import List


class Solution:
    def findMin(self, nums: List[int]) -> int:
        if(len(nums) == 1):
            return nums[0]
        mid = len(nums) // 2
        left = nums[:mid]
        right = nums[mid:]
        leftmin = self.findMin(left)
        rightmin = self.findMin(right)
        minimum = min(leftmin, rightmin)
        return minimum
sol = Solution()
print(sol.findMin([1,2,3,4,5]))