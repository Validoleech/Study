import pytest



def test_add_custom_list():
    cl1 = CustomList([5, 1, 3, 7])
    cl2 = CustomList([1, 2, 7])
    result = cl1 + cl2
    assert result == CustomList([6, 3, 10, 7])

def test_add_custom_list_with_list():
    cl1 = CustomList([1])
    lst = [2, 5]
    result = cl1 + lst
    assert result == CustomList([3, 5])

def test_add_list_with_custom_list():
    lst = [2, 5]
    cl1 = CustomList([1])
    result = lst + cl1
    assert result == CustomList([3, 5])

def test_sub_custom_list():
    cl1 = CustomList([5, 1, 3, 7])
    cl2 = CustomList([1, 2, 7])
    result = cl1 - cl2
    assert result == CustomList([4, -1, -4, 7])

def test_sub_custom_list_with_list():
    cl1 = CustomList([1])
    lst = [2, 5]
    result = cl1 - lst
    assert result == CustomList([-1, -5])

def test_sub_list_with_custom_list():
    lst = [2, 5]
    cl1 = CustomList([1])
    result = lst - cl1
    assert result == CustomList([1, 5])

def test_equality_custom_list():
    cl1 = CustomList([5, 1, 3, 7])
    cl2 = CustomList([1, 2, 7])
    assert cl1 == cl2  # Сумма одинаковая

def test_inequality_custom_list():
    cl1 = CustomList([5, 1, 3, 7])
    cl2 = CustomList([5, 1, 3])
    assert cl1 != cl2  # Суммы разные

def test_str_custom_list():
    cl = CustomList([1, 2, 3])
    assert str(cl) == "[1, 2, 3] (sum: 6)"