import sys


class Progress(object):
    """A simple progress bar implementation"""

    total = 0
    title = "Progress"
    n_chars = 60
    output = sys.stdout
    char = "█"
    void_char = "░"

    def __init__(self, total=0, title="Progress", n_chars=60, output=sys.stdout):
        """Constructor

        parameters
        ----------
        total : int
            The total number of steps to be completed.
        title : str
            The title of the progress bar
        n_chars : int
            The number of characters to use in the progress bar (the lenght of the progress bar).
        output : file-like object
            It could be a stream like sys.stdout or sys.stderr or a file.
        """
        self.total = total
        self.title = title
        self.n_chars = n_chars
        self.output = output

    def reset(self, total):
        """Reset the progress bar

        parameters
        ----------
        total : int
            The total number of steps to be completed.
        """
        self.total = total

    def start(self):
        """Start the progress bar."""
        self.output.write(f"\n{self.title} [{(self.void_char * self.n_chars)}]")
        self.output.flush()
        self.update(0)

    def update(self, count=1):
        """Update the progress bar with a given count.

        parameters
        ----------
        count : int
            The number of steps already completed
        """
        progress = int(count / self.total * self.n_chars)
        self.output.write(
            f"\r{self.title} [{(self.char * progress)}{(self.void_char * (self.n_chars - progress))}]"
        )
        self.output.flush()

    def finish(self):
        """Finish the progress bar."""
        self.output.write(f"\r{self.title} [{(self.char * self.n_chars)}]")
        self.output.flush()
        self.output.write("\n")
        self.output.flush()


default_progress = Progress()
