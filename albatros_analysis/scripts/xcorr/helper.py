import numpy as np
import time
import argparse
from os import path
import sys

sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
# from albatros_analysis.src.correlations import correlations as cr
from albatros_analysis.src.utils import baseband_utils as butils
import json


# def get_init_info_2ant(init_t, end_t, spec_offset, dir_parent0, dir_parent1):
#     # spec offset definition:
#     # offset in actual spdcrum numbers from two antennas that line up the two timestreams

#     f_start0, idx0 = butils.get_file_from_timestamp(init_t, dir_parent0, "f")
#     f_end0, _ = butils.get_file_from_timestamp(end_t, dir_parent0, "f")
#     files0 = butils.time2fnames(
#         butils.get_tstamp_from_filename(f_start0),
#         butils.get_tstamp_from_filename(f_end0),
#         dir_parent0,
#         "f",
#         mind_gap=True,
#     )

#     f0_obj = bdc.Baseband(f_start0)
#     specnum0 = f0_obj.spec_num[0] + idx0

#     f_start1, idx1 = butils.get_file_from_timestamp(init_t, dir_parent1, "f")
#     f_end1, _ = butils.get_file_from_timestamp(end_t, dir_parent1, "f")
#     files1 = butils.time2fnames(
#         butils.get_tstamp_from_filename(f_start1),
#         butils.get_tstamp_from_filename(f_end1),
#         dir_parent1,
#         "f",
#         mind_gap=True,
#     )

#     f1_obj = bdc.Baseband(f_start1)
#     specnum1 = f1_obj.spec_num[0] + idx1

#     init_offset = specnum0 - specnum1
#     print("before correction", idx0, idx1)
#     # idx0 += (spec_offset - init_offset) #needed offset - current offset, adjust one antenna's starting
#     idx1 -= spec_offset - init_offset  # the other way around.
#     if idx1 < 0:
#         raise NotImplementedError(
#             "Edge case, idx < 0. Don't start right at the beginning of a file."
#         )
#     # not handling the edge case for now
#     print("after correction", idx0, idx1)
#     return files0, idx0, files1, idx1


def get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents):
    """_summary_

    Parameters
    ----------
    init_t : float
        Start time in unix timestamp format.
    end_t : float
        End time in unix timestamp format
    spec_offsets : list
        spectrum number offsets for each antenna with reference to the first antenna.
        first antenna's offset with itself is always set to 0.
    dir_parents : list
        List of paths where each antenna's data (5-digit dirs) resides.

    Returns
    -------
    List
        list of In-file specidx offsets for each antenna, List of files for each antenna

    Raises
    ------
    NotImplementedError
        Note that due to an edge case, the method _might_ fail
        if the start time corresponds to beginning of a file,
        and in-file index pointer needs to seek to past times due to clock offsets.
    """
    # spec offset definition:
    # offset in actual spectrum numbers from two antennas that line up the two timestreams
    idxs = len(dir_parents) * [0]
    specnums = len(dir_parents) * [0]
    files = []
    for anum, dir_parent in enumerate(dir_parents):
        f_start, idx = butils.get_file_from_timestamp(init_t, dir_parent, "f")
        idxs[anum] = idx
        f_end, _ = butils.get_file_from_timestamp(end_t, dir_parent, "f")

        files.append(
            butils.time2fnames(
                butils.get_tstamp_from_filename(f_start),
                butils.get_tstamp_from_filename(f_end),
                dir_parent,
                "f",
                mind_gap=True,
            )
        )
        f_obj = bdc.Baseband(f_start)
        specnums[anum] = f_obj.spec_num[0] + idx
    
    if len(spec_offsets) == 1: #only one antenna
        return idxs, files

    for jj in range(1, len(idxs)):  # all except first antenna
        init_offset = specnums[0] - specnums[jj] # ref_ant - ant_jj
        print("before correction", idxs[0], idxs[jj])
        # idx0 += (spec_offset - init_offset) #needed offset - current offset, adjust one antenna's starting
        print(spec_offsets[jj] - init_offset) #this is the offset within the respective files
        idxs[jj] -= spec_offsets[jj] - init_offset  # the other way around.
        if idxs[jj] < 0:
            raise NotImplementedError(
                "Edge case, idx < 0. Don't start right at the beginning of a file."
            )
        # not handling the edge case for now
        print("after correction", idxs[0], idxs[jj])
    return idxs, files
