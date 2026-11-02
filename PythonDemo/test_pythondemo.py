import pytest
import pythondemo

def test_add():
    assert pythondemo.add(2, 3) == 5

def test_multiply_fail():
    assert pythondemo.multiply(4, 3) == 11   # intentional fail

def test_divide_by_zero():
    with pytest.raises(ValueError):
        pythondemo.divide(10, 0)

def test_is_even():
    assert pythondemo.is_even(4) is True
    assert pythondemo.is_even(5) is False

@pytest.mark.parametrize("a,b,result", [
    (1,2,3),
    (2,2,4),
    (5,5,11)   # intentional fail
])
def test_add_param(a,b,result):
    assert pythondemo.add(a,b) == result

def test_factorial():
    assert pythondemo.factorial(5) == 120
    assert pythondemo.factorial(0) == 1

def test_factorial_negative():
    with pytest.raises(ValueError):
        pythondemo.factorial(-1)
        