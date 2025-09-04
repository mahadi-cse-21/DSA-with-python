from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit =0
        for i in range(len(prices) -1):
            for j in range ( i+1, len(prices)):
               if prices[j]>prices[i]: max_profit = max(max_profit,prices[j]-prices[i])
        return max_profit
solu = Solution()
print(solu.maxProfit([10,1,5,6,7,1]))

