# Definition for singly-linked list.
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        rev = []
        first  = head
        while first:
            rev.append(first)
            first = first.next
        i ,j = 0, len(rev)-1
        for i in range(len(rev)//2):
            current_next = rev[i].next
            rev[i].next = rev[j-i]
            rev[i].next.next = current_next
            i +=1
        rev[i].next = None
        return


solu = Solution()
head = ListNode(2)
head.next = ListNode(4)
head.next.next = ListNode(6)
head.next.next.next = ListNode(8)
head.next.next.next.next = ListNode(10)
solu.reorderList(head)




