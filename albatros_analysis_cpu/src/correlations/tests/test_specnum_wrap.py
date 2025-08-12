from src.correlations import baseband_data_classes as bdc
import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
        tracemalloc.Filter(False, "*numba*")
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

def test_specnum_wrap():
    files=['/gpfs/fs1/home/s/sievers/mohanagr/albatros_analysis/src/correlations/tests/data/specnum_wrap1.raw',
     '/gpfs/fs1/home/s/sievers/mohanagr/albatros_analysis/src/correlations/tests/data/specnum_wrap2.raw',
     '/gpfs/fs1/home/s/sievers/mohanagr/albatros_analysis/src/correlations/tests/data/specnum_wrap3.raw']
    start_file_num=0
    acclen=10
    idxstart=0
    tracemalloc.start()
    ant1=bdc.BasebandFileIterator(files,start_file_num,idxstart,acclen)
    spec_start=2**32-10
    data=ant1.__next__() #first 10
    snapshot1=tracemalloc.take_snapshot()
    display_top(snapshot1)
    assert ant1.obj._overflowed==True
    assert len(data['specnums'])==10
    assert data['specnums'][-1]==spec_start+acclen-1
    data=ant1.__next__() #first 20
    # print(data)
    assert len(data['specnums'])==5 #5 missing
    assert data['specnums'][-1]==spec_start+2*acclen-1
    data=ant1.__next__() #first 30
    print(type(ant1.obj.spec_idx), type(data['specnums']))
    # print(data)
    data=ant1.__next__() #first 40, 5 from file 1, 5 from file 2
    data['specnums'][-1]+5-1==ant1.obj.spec_idx[5-1] #we must have hit the end of file 1 midway in this block
    # print(data)
    assert ant1.obj._overflowed==False #second file has no overflows
    assert data['specnums'][-1]==spec_start+4*acclen-1
    data=ant1.__next__() #first 50, next 10 from file 2. end of file 2
    # print(data)
    data['specnums'][-1]+5-1==ant1.obj.spec_idx[-1]
    data=ant1.__next__() #file 3, has another wrap
    assert ant1._OVERFLOW_CTR==2
    assert len(data['specnums'])==5 #because we introduced a huge gap by wrapping in middle of file
