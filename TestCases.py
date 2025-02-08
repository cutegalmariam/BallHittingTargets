def test1():
    m = 0.5
    b = 0.07
    return m, b


def test2():
    m = 0.5
    b = 0.5
    return m, b


def test3():
    m = 3.0
    b = 0.07
    return m, b


def test4():
    m = 3.0
    b = 0.5
    return m, b


def test5():
    m = 1.0
    b = 0.2
    return m, b


def test6():
    m = 0.5
    b = 0.0
    return m, b


def get_mass_drag(test_case_num):
    m = 0
    b = 0
    match test_case_num:
        case 1:
            m = test1()[0]
            b = test1()[1]
        case 2:
            m = test2()[0]
            b = test2()[1]
        case 3:
            m = test3()[0]
            b = test3()[1]
        case 4:
            m = test4()[0]
            b = test4()[1]
        case 5:
            m = test5()[0]
            b = test5()[1]
        case 6:
            m = test6()[0]
            b = test6()[1]
    return m, b
