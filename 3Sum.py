from typing import List

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        result_set = set()

        for i in range(len(nums) - 2):
            for j in range(i + 1, len(nums) - 1):
                for k in range(j + 1, len(nums)):
                    if nums[i] + nums[j] + nums[k] == 0:
                        triplet = tuple(sorted([nums[i], nums[j], nums[k]]))
                        result_set.add(triplet)

        return [list(t) for t in result_set]

# Test
solution = Solution()
print(solution.threeSum([-1, 0, 1, 2, -1, -4]))
