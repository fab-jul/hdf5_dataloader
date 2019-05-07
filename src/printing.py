"""
Copied from fjcommon, https://github.com/fab-jul/fjcommon/blob/master/fjcommon/printing.py.
"""

import collections


def print_join(it, joiner='\n'):
    """
    Like print(joiner.join(it)) but for iterators that take a long time to produce elements, this version prints
    elements as they are available
    """
    it = iter(it)
    first = next(it)
    print(first, flush=True, end='')
    for el in it:
        print(joiner + el, flush=True, end='')


# Progress Printing ------------------------------------------------------------


class ProgressPrinter(object):
    """
    Supports usage as context manager or iterator:

    with ProgressPrinter('info') as p:
        ...
        p.update(some_progress)

    for el in ProgressPrinter('info', iter_list=some_list):
        ... do sth with el
    """
    def __init__(self, info=None, iter_list=None):
        """
        :param info: if not None, printed on call
        :param iter_list: if not None, instance can be used as iterable
        """
        assert iter_list is None or isinstance(iter_list, collections.Sequence), 'iter_list must be sequence'
        if info:
            print(info)
        self.did_update_line = False
        self.iter_list = iter_list
        self.iter_list_idx = 0

    def update(self, p):
        self.did_update_line = True
        progress_print(p)

    def finish_line(self):
        if self.did_update_line:
            self.did_update_line = False
            progress_print(-1, _reset_cache=True)
            print()  # Finish line

    def __iter__(self):
        return self

    def __next__(self):
        if not self.iter_list:
            raise StopIteration
        if self.iter_list_idx >= len(self.iter_list):
            self.finish_line()
            raise StopIteration
        el = self.iter_list[self.iter_list_idx]
        progress_print(self.iter_list_idx / len(self.iter_list))
        self.iter_list_idx += 1
        return el

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.finish_line()
        return True


def progress_print(p, _last_pstr_cache=[''], _reset_cache=False):
    if _reset_cache:
        _last_pstr_cache[0] = ''
        return
    assert 0 <= p <= 1
    pstr = '{:.2f}'.format(p)
    if _last_pstr_cache[0] != pstr:  # prevents flickering on some consoles: only update if something changed!
        _last_pstr_cache[0] = pstr
        print('\r{}'.format(pstr), end='')

