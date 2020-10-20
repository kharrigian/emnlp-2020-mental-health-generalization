
"""
Workaround for spawning multiple processes within a spawn
of multiple processes.

Source: https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
"""

#####################
### Imports
#####################

## Multiprocessing
import multiprocessing
import multiprocessing.pool

#####################
### Classes
#####################

class NoDaemonProcess(multiprocessing.Process):

    """
    Make 'daemon' attribute always return False
    """

    def _get_daemon(self):
        """
        Helper to identify existing _daemon

        Args:
            self
        
        Returns:
            daemon (bool): Always returns False.
        """
        return False

    def _set_daemon(self,
                    value):
        """
        Helper to assign a daemon process a value.
        Method just needs to exist.

        Args:
            value (Any): daemon process
        
        Returns:
            None
        """
        pass

    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    """
    We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
    because the latter is only a wrapper function, not a proper class.
    """
    Process = NoDaemonProcess