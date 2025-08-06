class HelloWorld:
    """
    Example class; greets!
    """

    def __init__(self, name: str):
        """
        Args:
            name: string to greet.
        """
        self.name = name

    def hello(self):
        """Prints greeting to name."""
        print(f"Hello {self.name}!")

    def name_length(self) -> int:
        """Returns length of name."""
        return len(self.name)


def main():
    """Prints `Hello World`."""
    hw = HelloWorld(name="World")
    hw.hello()


if __name__ == "__main__":
    main()
