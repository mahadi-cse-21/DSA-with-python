class node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def inorder(root):
    if root is None:
        return
    inorder(root.left)
    print(root.val, end=' ')
    inorder(root.right)


def insert(root, val):
    if root is None:
        return node(val)

    if val < root.val:
        root.left = insert(root.left, val)
    else:
        root.right = insert(root.right, val)

    return root


# Test the implementation
root = None
root = insert(root, 1)
root = insert(root, 4)
root = insert(root, 20)
root = insert(root, 12)  # Fixed this line - you had root = (root,12) which is incorrect

print("Inorder traversal of the BST:")
inorder(root)