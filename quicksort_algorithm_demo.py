def quicksort(arr):
    """
    Sorts the input list using the quicksort algorithm.

    Parameters:
    - arr (list): The input list to be sorted.

    Returns:
    - list: The sorted list.
    """
    if len(arr) <= 1:
        return arr

    # Choose the pivot as the middle element
    pivot = arr[len(arr) // 2]

    # Divide the list into three parts: elements less than, equal to, and greater than the pivot
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    # Recursively apply quicksort to the left and right sublists
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

# Example usage and demonstration
unsorted_list = [3, 6, 8, 10, 1, 2, 1]

print("Unsorted List:")
print_list(unsorted_list)

sorted_list = quicksort(unsorted_list)

print("\nSorted List:")
print_list(sorted_list)
