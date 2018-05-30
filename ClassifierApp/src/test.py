def test_function(a, b):
    return a + b


def invoker(function_name, args):
    return function_name(*args)


print(invoker(test_function, (5, 6)))

