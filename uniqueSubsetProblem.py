def get_all_subsets(nums):
    if not nums:
        return [set()]
    element = next(iter(nums))
    rest = nums - {element}
    subsets_without_element = get_all_subsets(rest)
    subsets_with_element = [subsets | {element} for subsets in subsets_without_element]

    return subsets_with_element + subsets_without_element
nums = {1,5,5}
print("All subsets of nums: ",get_all_subsets(nums))