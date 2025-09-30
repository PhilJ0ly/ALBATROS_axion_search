#!/usr/bin/env python3

# Philippe Joly 2025-09-24

"""
Script to find continuous time intervals from timestamped raw ALBATROS files.
"""

import os
import glob
from typing import List, Tuple
import numpy as np


def find_continuous_intervals(base_path: str, t_0: int, t_f: int, T: int) -> List[Tuple[int, int]]:
    """
    Find continuous time intervals where files exist without gaps > T.
    
    Args:
        base_path: Path containing the subdirectories (17213, 17214, etc.)
        t_0: Start timestamp to search from
        t_f: End timestamp to search to  
        T: Duration of data in each file (gap threshold)
        
    Returns:
        List of tuples (start_time, end_time) representing continuous intervals
    """
    
    # Collect all timestamps from all subdirectories
    all_timestamps = []
    
    # Look for subdirectories matching the pattern 1721X
    subdirs = glob.glob(os.path.join(base_path, "1721*"))
    
    for subdir in subdirs:
        if os.path.isdir(subdir):
            # Get all .raw files in this subdirectory
            raw_files = glob.glob(os.path.join(subdir, "*.raw"))
            
            for file_path in raw_files:
                filename = os.path.basename(file_path)
                # Extract timestamp (remove .raw extension)
                if filename.endswith('.raw'):
                    try:
                        timestamp = int(filename[:-4])  # Remove .raw
                        all_timestamps.append(timestamp)
                    except ValueError:
                        print(f"Warning: Could not parse timestamp from {filename}")
    
    # Filter timestamps to the requested range and sort
    filtered_timestamps = [ts for ts in all_timestamps if t_0 <= ts <= t_f]
    filtered_timestamps.sort()
    
    if not filtered_timestamps:
        print(f"No files found in time range {t_0} to {t_f}")
        return []
    
    print(f"Found {len(filtered_timestamps)} files in time range")
    print(f"First file: {filtered_timestamps[0]}")
    print(f"Last file: {filtered_timestamps[-1]}")
    
    # Find continuous intervals
    continuous_intervals = []
    current_start = filtered_timestamps[0]
    current_end = filtered_timestamps[0]
    
    for i in range(1, len(filtered_timestamps)):
        current_ts = filtered_timestamps[i]
        prev_ts = filtered_timestamps[i-1]
        
        # Check if there's a gap larger than T
        if current_ts - prev_ts > T:
            # End the current interval
            continuous_intervals.append((current_start, current_end))
            # Start a new interval
            current_start = current_ts
            current_end = current_ts
        else:
            # Continue the current interval
            current_end = current_ts
    
    # Don't forget the last interval
    continuous_intervals.append((current_start, current_end))
    
    return continuous_intervals

def print_intervals(intervals: List[Tuple[int, int]]):
    """Print the continuous intervals in a readable format."""
    if not intervals:
        print("No continuous intervals found.")
        return
    
    print(f"\nFound {len(intervals)} continuous interval(s):")
    print("-" * 50)
    
    for i, (start, end) in enumerate(intervals, 1):
        duration = end - start
        print(f"Interval {i}: {start} to {end} (duration: {duration} seconds)")

def analyze_directory(base_path: str = "."):
    """
    Analyze all files in the directory structure and show statistics.
    Useful for understanding your data before setting parameters.
    """
    all_timestamps = []
    
    # Collect all timestamps
    subdirs = glob.glob(os.path.join(base_path, "*"))
    
    for subdir in subdirs:
        if os.path.isdir(subdir):
            raw_files = glob.glob(os.path.join(subdir, "*.raw"))
            subdir_name = os.path.basename(subdir)
            print(f"Directory {subdir_name}: {len(raw_files)} files")
            
            for file_path in raw_files:
                filename = os.path.basename(file_path)
                if filename.endswith('.raw'):
                    try:
                        timestamp = int(filename[:-4])
                        all_timestamps.append(timestamp)
                    except ValueError:
                        pass
    
    if all_timestamps:
        all_timestamps.sort()
        print(f"\nTotal files: {len(all_timestamps)}")
        print(f"Time range: {all_timestamps[0]} to {all_timestamps[-1]}")
        print(f"Total duration: {all_timestamps[-1] - all_timestamps[0]} seconds")
        
        # Calculate gaps between consecutive files
        gaps = [all_timestamps[i] - all_timestamps[i-1] for i in range(1, len(all_timestamps))]
        if gaps:
            print(f"Average gap: {sum(gaps)/len(gaps):.1f} seconds")
            print(f"Min gap: {min(gaps)} seconds")
            print(f"Max gap: {max(gaps)} seconds")
            
            # Show distribution of gap sizes
            gap_counts = {}
            for gap in gaps:
                gap_counts[gap] = gap_counts.get(gap, 0) + 1
            
            print("\nGap distribution (top 10):")
            for gap, count in sorted(gap_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {gap}s: {count} occurrences")

def main():
    # Configuration - modify these values as needed
    BASE_PATH = "/scratch/mohanagr/mars_axion/baseband/" 
    T_START = 1721409707  # Start timestamp
    T_END = 1721410504    # End timestamp  
    T_DURATION = 60       # Duration of data in each file (seconds)
    
    print(f"Searching for continuous intervals...")
    print(f"Time range: {T_START} to {T_END}")
    print(f"File duration (T): {T_DURATION} seconds")
    print(f"Base path: {BASE_PATH}")
    print()
    
    # Find continuous intervals
    intervals = find_continuous_intervals(BASE_PATH, T_START, T_END, T_DURATION)
    
    # Print results
    print_intervals(intervals)
    
    return intervals


def split_by_gaps(times: List[np.int64], plasma: List[int], gaps: List[Tuple[int, int]]):
    """
    Split time and plasma lists into continuous chunks based on data gaps.
    
    Parameters:
    times : list
        List of timestamps 
    plasma : list
        List of plasma values where plasma[i] corresponds to interval times[i] -> times[i+1]
        Length should be len(times) - 1
    gaps : list of tuples
        List of (gap_start, gap_duration) tuples defining data gaps
        
    Returns:
    time_chunks : list of lists
        List of continuous time chunks
    plasma_chunks : list of lists
        List of corresponding plasma chunks
     
    """
    
    if len(plasma) != len(times) - 1:
        raise ValueError(f"Plasma length ({len(plasma)}) must be len(times) - 1 ({len(times) - 1})")
    
    
    time_chunks = []
    current_chunk = []
    kept_indices = []  
    
    for i, t in enumerate(times):
        exclude = False
        for gap_start, gap_duration in gaps:
            gap_end = gap_start + gap_duration
            if gap_start <= t <= gap_end:
                exclude = True
                break
        
        if exclude:
            if current_chunk:
                time_chunks.append(current_chunk)
                current_chunk = []
        else:
            should_break = False
            if current_chunk:
                for gap_start, gap_duration in gaps:
                    gap_end = gap_start + gap_duration
                    last_t = times[kept_indices[-1]]
                    if last_t < gap_start and t > gap_end:
                        should_break = True
                        break
            
            if should_break and current_chunk:
                time_chunks.append(current_chunk)
                current_chunk = [t]
                kept_indices.append(i)
            else:
                current_chunk.append(t)
                kept_indices.append(i)
    
    if current_chunk:
        time_chunks.append(current_chunk)
    
    plasma_chunks = []
    chunk_start_idx = 0
    
    for chunk in time_chunks:
        chunk_len = len(chunk)
        # For each time chunk, we have chunk_len - 1 plasma intervals
        if chunk_len > 1:
            plasma_chunk = []
            for i in range(chunk_len - 1):
                # Find the original index for this time point
                original_idx = kept_indices[chunk_start_idx + i]
                plasma_chunk.append(plasma[original_idx])
            
            plasma_chunks.append(plasma_chunk)
        elif chunk_len == 1:
            # Single timestamp, no plasma interval
            plasma_chunks.append([])
        
        chunk_start_idx += chunk_len
    
    return time_chunks, plasma_chunks


# Example usage:
if __name__ == "__main__":
    times = [np.int64(1721631600), np.int64(1721631900), np.int64(1721632200), 
             np.int64(1721632500), np.int64(1721632800), np.int64(1721633100), 
             np.int64(1721633400), np.int64(1721633700), np.int64(1721634000), 
             np.int64(1721634300), np.int64(1721634600), np.int64(1721634900), 
             np.int64(1721635200), np.int64(1721635500), np.int64(1721635800), 
             np.int64(1721636100), np.int64(1721636400), np.int64(1721636700), 
             np.int64(1721637000), np.int64(1721637300), np.int64(1721637600), 
             np.int64(1721637900), np.int64(1721638200), np.int64(1721638500), 
             np.int64(1721638800), np.int64(1721639100), np.int64(1721639400), 
             np.int64(1721639700), np.int64(1721640000), np.int64(1721640300), 
             np.int64(1721640600), np.int64(1721640900), np.int64(1721641200), 
             np.int64(1721641500), np.int64(1721641800), np.int64(1721642100), 
             np.int64(1721642400), np.int64(1721642700), np.int64(1721643000), 
             np.int64(1721643300), np.int64(1721643600), np.int64(1721643900), 
             np.int64(1721644200), np.int64(1721644500), np.int64(1721644800), 
             np.int64(1721645100), np.int64(1721645400), np.int64(1721645700), 
             np.int64(1721646000)]
    
    # Create dummy plasma data
    plasma = list(range(len(times) - 1))
    
    # Define gaps
    gaps = [
        (1721639204, 49),  # gap from 1721639204 to 1721639253
        (1721640533, 51),  # gap from 1721640533 to 1721640584
    ]
    
    # Split the data
    time_chunks, plasma_chunks = split_by_gaps(times, plasma, gaps)
    
    # Print results
    print(f"Split into {len(time_chunks)} chunks:\n")
    for i, (t_chunk, p_chunk) in enumerate(zip(time_chunks, plasma_chunks)):
        print(f"Chunk {i+1}:")
        print(f"  Time: {len(t_chunk)} timestamps")
        print(f"  Plasma: {len(p_chunk)} intervals")
        print(f"  First time: {t_chunk[0]}, Last time: {t_chunk[-1]}")
        if p_chunk:
            print(f"  First plasma: {p_chunk[0]}, Last plasma: {p_chunk[-1]}")
        
        print()


