from typing import List


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        nums = sorted(nums)
        nums1 = sorted(set(nums))

        nums2 = list(nums1)

        diff = []
        for i in range(0,len(nums2)-1):
                diff.append(nums2[i + 1] - nums2[i])


        max_len = 0
        cnt = 1
        for i in range(len(diff)):
            if diff[i] == 1:
                cnt += 1
            else:
                max_len = max(max_len, cnt)
                cnt=1

        max_len = max(max_len, cnt)
        if (len(nums)) == 0: return 0
        print(diff)
        print(nums2)
        return max_len
sol = Solution()
print(sol.longestConsecutive([9,1,4,7,3,-1,0,5,8,-1,6]))
