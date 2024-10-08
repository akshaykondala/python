def quicksort(arr):
    
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]    
    return quicksort(left) + middle + quicksort(right)

def print_list(arr):
    """Prints the elements of a list."""
    print("[", end="")
    for i, elem in enumerate(arr):
        if i < len(arr) - 1:
            print(f"{elem}, ", end="")
        else:
            print(f"{elem}", end="")
    print("]")


#Test Case
unsorted_list = [3, 6, 8, 10, 1, 2, 1]

print("Unsorted List:")
print_list(unsorted_list)

sorted_list = quicksort(unsorted_list)

print("\nSorted List:")
print_list(sorted_list)
