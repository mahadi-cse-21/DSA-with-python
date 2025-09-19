from typing import Optional, List

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return None

        # Step 1: Find the first non-empty list to use as head
        head = None
        start_idx = 0
        for idx, node in enumerate(lists):
            if node is not None:
                head = node
                start_idx = idx
                break

        if head is None:
            return None  # all lists are None

        # Step 2: Merge remaining lists one by one into head
        for i in lists[start_idx + 1:]:
            if i is None:
                continue

            first1 = head      # current head list
            first2 = i         # new list to merge

            # Special case: insert nodes from first2 before head
            while first2 and first2.val < head.val:
                temp = first2.next
                first2.next = head
                head = first2
                first2 = temp

            # Main merging logic
            prev = head
            curr = head.next

            while first2:
                if curr is None:
                    # Attach the rest of first2
                    prev.next = first2
                    break

                if first2.val < curr.val:
                    temp = first2.next
                    prev.next = first2
                    first2.next = curr
                    prev = first2
                    first2 = temp
                else:
                    prev = curr
                    curr = curr.next

        return head
