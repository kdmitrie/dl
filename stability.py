import sys

class Stability:
    """
    Class that ensures a stable script execution.
    In the case of hardware failure before execution ends, the script is restarted
    """
    def __init__(self, done_file: str, exit_if_done: bool=True):
        self.done_file = done_file
        if exit_if_done:
            self.exit_if_done()

    def get_signature(self):
        return ' '.join(sys.argv)

    def is_done(self):
        with open(self.done_file, 'r') as f:
            data = f.read()

        return self.get_signature() in data.split('\n')

    def exit_if_done(self):
        if self.is_done():
            print('The script is already run. Exiting.')
            quit()

    def mark_done(self):
        with open(self.done_file, 'a') as f:
            f.write(self.get_signature() + '\n')
