from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        new_head = ListNode()
        while head:
            val = head.val
            temp = ListNode(val)
            head = head.next
            if new_head==None:
                new_head = temp
            else:
                temp.next = new_head
                new_head = temp
        return new_head


