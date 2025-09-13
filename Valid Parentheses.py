class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) % 2 != 0:
            return False
        test = ""
        for i in s:
            if i == '(' or i == '{' or i == '[':
                test=test+i

            else:
                if i ==')':
                    if test.endswith('('):
                        test=test[:-1]
                    else :
                        return False
                elif i == '}':
                    if test.endswith('{'):
                       test=test[:-1]
                    else :
                        return False
                elif i == ']':
                    if test.endswith('['):
                        test=test[:-1]

                    else:
                        return False
                print(test)

        if len(test) == 0:
            return True
        else:
            return False

sol = Solution()
print(sol.isValid("[]"))