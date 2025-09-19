# Definition for singly-linked list.
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        first = head
        arr = []
        while first:
            arr.append(first)
            first = first.next
        length = len(arr)
        if length == 1:
            return None
        length = length - n - 1
        first = arr[length]
        if arr[length + 1] == head:
            head = head.next
            return head
        if first.next:
            first.next = first.next.next
        else:
            first.next = None

        return head



