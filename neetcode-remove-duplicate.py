from typing import List

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        nums1 = set(nums)
        nums[:len(nums1)]=nums1
        return len(nums)

sol = Solution()
result = sol.removeDuplicates([1,1,2,3,4])
print(result)