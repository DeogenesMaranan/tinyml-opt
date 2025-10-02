"""Loading indicator utility for providing visual feedback during long operations."""

import sys
import time
import threading


class LoadingIndicator:
    """A threaded loading indicator with animated spinner."""
    
    def __init__(self, message="Loading"):
        self.message = message
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the loading animation in a separate thread."""
        self.running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the loading animation and clear the line."""
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()
    
    def _animate(self):
        """Internal method to animate the spinner."""
        chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        i = 0
        while self.running:
            sys.stdout.write(f'\r{chars[i % len(chars)]} {self.message}...')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1