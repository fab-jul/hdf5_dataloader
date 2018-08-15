import pytest
from hdf5 import dataloader
from fjcommon import functools_ext as ft
from fjcommon import iterable_ext as it
from collections import Counter, namedtuple

class _FakeFile(object):
    """ Behaves like a HDF5 file """
    def __init__(self, p, content: dict):
        self.p = p
        self.content = content
        self._open = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def keys(self):
        return self.content.keys()

    def __getitem__(self, item):
        assert self._open
        try:
            return self.content[item]
        except KeyError:
            raise KeyError('{} not in {}'.format(item, self.content))

    def __str__(self):
        return str(self.content)

    def close(self):
        self._open = False


ElementsGeneratorConfig = namedtuple('ElementsGeneratorConfig', [
    'num_readers', 'num_shards', 'num_per_shard', 'last_one_smaller_by', 'batch_size',
    'filter_smaller_shards', 'assert_exactly_once', 'shuffle_shards'])


@pytest.fixture()
def make_shard_reader():
    def _make_shard_reader(c: ElementsGeneratorConfig):
        file_ps = ft.lmap(str, range(c.num_shards))
        num_els_total = c.num_shards * c.num_per_shard - c.last_one_smaller_by
        print('# before filter: {}'.format(num_els_total))
        shards = make_shards(num_els_total, c.num_per_shard)
        opener = get_opener(shards, c.num_per_shard)
        sr = dataloader.ShardReader(
                file_ps, c.num_readers, c.batch_size,
                filter_smaller_shards=c.filter_smaller_shards,
                assert_exactly_once=c.assert_exactly_once,
                shuffle_shards=c.shuffle_shards,
                opener=opener)
        return sr
    return _make_shard_reader


@pytest.fixture(
        params=[
            ElementsGeneratorConfig(
                    num_readers=4,
                    num_per_shard=50,
                    num_shards=10,
                    last_one_smaller_by=10,
                    batch_size=3,
                    filter_smaller_shards=True,
                    assert_exactly_once=False,
                    shuffle_shards=True),
            ElementsGeneratorConfig(
                    num_readers=4,
                    num_per_shard=50,
                    num_shards=10,
                    last_one_smaller_by=10,
                    batch_size=3,
                    filter_smaller_shards=True,
                    assert_exactly_once=False,
                    shuffle_shards=False),
            ElementsGeneratorConfig(
                    num_readers=1,
                    num_per_shard=50,
                    num_shards=10,
                    last_one_smaller_by=0,
                    batch_size=1,
                    filter_smaller_shards=False,
                    assert_exactly_once=True,
                    shuffle_shards=True)
        ])
def elements_generator_config(request):
    return request.param


@pytest.fixture()
def elements(elements_generator_config, make_shard_reader):
    c = elements_generator_config  # :: ElementsGeneratorConfig
    sr = make_shard_reader(c)
    return iterate_and_check(c, sr), c


def iterate_and_check(c: ElementsGeneratorConfig, sr):
    all_els = all_elements(sr)
    print('# after filter:  {}'.format(len(all_els)))
    seen = []
    num_batches = -1
    for idx in range(len(all_els)):
        if idx % c.batch_size == 0:
            num_batches += 1
        if num_batches == sr.max_full_batches:
            break
        el = sr[idx]
        seen.append(el)
    return _assert_equal_els(ft.lmap(int, all_els),
                             ft.lmap(int, seen))


def get_opener(shards, num_per_shard):
    def opener(p):
        shard = shards[int(p)]
        return _FakeFile(p, {str(k % num_per_shard): str(k) for k in shard})
    return opener


def make_shards(num_els_total, num_per_shard):
    return list(it.sliced_iter(list(range(num_els_total)), num_per_shard))


def all_elements(shard_reader: dataloader.ShardReader):
    return ft.lconcat(f.content.values() for f in ft.concat(shard_reader.files.values()))


def _assert_equal_els(all_els, seen):
    all_els = Counter(all_els)
    seen = Counter(seen)
    for el, num_occ in all_els.items():
        assert seen[el] == num_occ  #, 'Differing on {}'.format(el))
    return all_els, seen


# Tests ========================================================================


def test_ShardReader(elements):
    (all_els, seen), c = elements
    assert max(seen.values()) <= 2

    if c.assert_exactly_once:
        assert all(count == 1 for count in all_els.values())
        assert all(count == 1 for count in seen.values())


def test_reshuffle(elements_generator_config, make_shard_reader):
    c = elements_generator_config
    sr = make_shard_reader(c)
    all_els_a, _ = iterate_and_check(c, sr)
    assert len(all_els_a) > 0
    sr.initialize(shuffle_shards=True)
    all_els_b, _ = iterate_and_check(c, sr)
    assert set(all_els_a) == set(all_els_b)



