from mypackage import HelloWorld


def test_init_sets_name():
    hw = HelloWorld("Alice")
    assert hw.name == "Alice"


def test_hello_prints_correct_message(capsys):
    hw = HelloWorld("Bob")
    hw.hello()
    captured = capsys.readouterr()
    assert captured.out.strip() == f"Hello {hw.name}!"


def test_name_length():
    hw = HelloWorld("Charlie")
    assert hw.name_length() == 7
