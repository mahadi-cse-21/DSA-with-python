from typing import List
import numpy
from numpy import unsignedinteger


class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        answer = []
        product =  []
        for i in nums:
            test = nums.copy()
            test.remove(i)
            product.append(numpy.prod(test))

        return product
solu = Solution()
ans = solu.productExceptSelf([-1,0,1,2,3])
print(ans)