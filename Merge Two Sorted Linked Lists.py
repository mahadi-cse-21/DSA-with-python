# Definition for singly-linked list.
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        answer = None
        head = answer

        while list1 and list2:
            if list1.val < list2.val:
                if head is None:
                    head = list1
                    answer = head

                else:
                    head.next = list1
                    head = head.next

                list1 = list1.next
            else:
                if head is None:
                    head = list2
                    answer = head

                else:
                    head.next = list2
                    head = head.next

                list2 = list2.next

        if list1:
            if head == None:
                head = list1
                answer = head
            else:
                head.next = list1
        else:
            if head == None:
                head = list2
                answer = head

            else:
                head.next = list2

        return answer