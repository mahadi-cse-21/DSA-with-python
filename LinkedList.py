
class Node:
    def __init__(self, val, next=None):

          self.val = val
          self.next = next
    def __str__(self):
        return str(self.val)
def display(head):
    current = head
    elements = []
    while(current):
        elements.append(str(current.val))
        current = current.next
    print( ' -> '.join(elements))
def search(head,val):
    current = head
    while(current):
        if(current.val == val):
            return True
        current = current.next
    return False

head = Node(1)
A = Node(4)
B = Node(8)
C = Node(10)
head.next=A
A.next = B
B.next=C
display(head)
print(search(head,5))
print(search(head,10))
del(head)

class doubleNode:
    def __init__(self, val, left=None, right = None):
        self.val = val
        self.left = left
        self.right = right
    def __str__(self):
        return str(self.val)

def display2(head):
    current = head
    elements = []
    while(current):
        elements.append(str(current.val))
        current = current.right
    print(' <-> '.join(elements))

head = tail = doubleNode(1)
A = doubleNode(4)
B = doubleNode(8)
C = doubleNode(10)
head.right = A
A.right=B
A.left = head
B.right = C
B.left=A
C.left=B
display2(head)


