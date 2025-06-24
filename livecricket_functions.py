import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_mouse_centroid(dataframe, start_frame=None, end_frame=None):
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = len(dataframe)
    # Get x coordinates and check for nans
    x_coords = [dataframe.nosex.iloc[start_frame:end_frame],
                dataframe.leftearx.iloc[start_frame:end_frame],
                dataframe.rightearx.iloc[start_frame:end_frame],
                dataframe.tailbasex.iloc[start_frame:end_frame]]
    centroid_x_sum = np.where(np.any(pd.isnull(x_coords), axis=0), 
                             np.nan,
                             sum(x_coords))

    # Get y coordinates and check for nans 
    y_coords = [dataframe.nosey.iloc[start_frame:end_frame],
                dataframe.lefteary.iloc[start_frame:end_frame],
                dataframe.righteary.iloc[start_frame:end_frame],
                dataframe.tailbasey.iloc[start_frame:end_frame]]
    centroid_y_sum = np.where(np.any(pd.isnull(y_coords), axis=0),
                             np.nan, 
                             sum(y_coords))
    centroid_x = centroid_x_sum/4.0
    centroid_y = centroid_y_sum/4.0
    return centroid_x, centroid_y

def get_mouse_speed(dataframe, start_frame=None, end_frame=None, fps=30):
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = len(dataframe)
    centroid_x, centroid_y = get_mouse_centroid(dataframe, start_frame, end_frame)
    # Calculate speed in cm per frame
    dx = np.diff(centroid_x)
    dy = np.diff(centroid_y)
    # If either dx or dy is nan, speed should be nan for that step
    speed_cm = np.sqrt(dx**2 + dy**2)
    speed_cm[np.isnan(dx) | np.isnan(dy)] = np.nan
    # Convert to cm per second
    speed_seconds = speed_cm * fps
    # Pad the result to match the input shape (same length as centroid_x)
    # We'll prepend a nan so that output length matches input
    speed_seconds_full = np.empty_like(centroid_x)
    speed_seconds_full[:] = np.nan
    speed_seconds_full[1:] = speed_seconds
    return speed_seconds_full

def get_cricket_speed(dataframe, start_frame=None, end_frame=None, fps=30):
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = len(dataframe)
    # Calculate speed in cm per frame
    dx = np.diff(dataframe.stimx.iloc[start_frame:end_frame])
    dy = np.diff(dataframe.stimy.iloc[start_frame:end_frame])
    # If either dx or dy is nan, speed should be nan for that step
    speed_cm = np.sqrt(dx**2 + dy**2)
    speed_cm[np.isnan(dx) | np.isnan(dy)] = np.nan
    # Convert to cm per second
    speed_seconds = speed_cm * fps
    # Pad the result to match the input shape (same length as centroid_x)
    # We'll prepend a nan so that output length matches input
    speed_seconds_full = np.empty_like(dataframe.stimx.iloc[start_frame:end_frame])
    speed_seconds_full[:] = np.nan
    speed_seconds_full[1:] = speed_seconds
    return speed_seconds_full

def get_distance_travelled(dataframe, start_frame=None, end_frame=None):
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = len(dataframe)
    centroid_x, centroid_y = get_mouse_centroid(dataframe, start_frame, end_frame)
    # Calculate distance in cm per frame
    dx = np.diff(centroid_x)
    dy = np.diff(centroid_y)
    # If either dx or dy is nan, distance should be nan for that step
    distance_cm = np.sqrt(dx**2 + dy**2)
    distance_cm[np.isnan(dx) | np.isnan(dy)] = np.nan
    # Sum up all the distances to get total distance travelled
    # Use nansum to ignore nan values in the summation
    total_distance = np.nansum(distance_cm)
    return total_distance

def get_mouse_heading_to_cricket(dataframe, start_frame=None, end_frame=None):
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = len(dataframe)
    headings = get_mouse_heading(dataframe, (start_frame, end_frame), return_cricket_azimuth = True)   
    mouse_heading, cricket_azimuth = zip(*headings)                    
    heading_cricket_mouse = ((np.array(cricket_azimuth) - np.array(mouse_heading) + np.pi) % (2*np.pi)) - np.pi #recheck this formula
    return np.rad2deg(heading_cricket_mouse)


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def time_to_frame_converter(time, fps=30):
    return int(time*fps)

def get_mouse_heading(dataframe,
                    frame_num,
                    return_cricket_azimuth = False,
                    object_azimuth = None):
    """
    Get the heading of the mouse for a given frame or range of frames.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        DataFrame containing tracking data
    frame_num : int or tuple
        Frame number to analyze. If tuple (start, end), returns list of headings for frames in range
    return_cricket_azimuth : bool
        If True, also return cricket azimuth
    object_azimuth : array-like or None
        Optional object position to calculate azimuth to
        
    Returns:
    --------
    float or list
        Mouse heading(s) in radians. If frame_num is tuple, returns list of headings.
        If return_cricket_azimuth is True, returns tuple of (heading, cricket_azimuth)
    """
    def _get_single_frame_heading(frame):
        # Check if any keypoints are nan
        try:
            if pd.isnull(dataframe.leftearx.iloc[frame]) or pd.isnull(dataframe.rightearx.iloc[frame]) or \
               pd.isnull(dataframe.lefteary.iloc[frame]) or pd.isnull(dataframe.righteary.iloc[frame]) or \
               pd.isnull(dataframe.nosex.iloc[frame]) or pd.isnull(dataframe.nosey.iloc[frame]):
                if return_cricket_azimuth:
                    return np.nan, np.nan
                return np.nan
        except:
            if pd.isnull(dataframe.leftearx.iloc[frame]) or pd.isnull(dataframe.rightearx.iloc[frame]) or \
               pd.isnull(dataframe.lefteary.iloc[frame]) or pd.isnull(dataframe.righteary.iloc[frame]) or \
               pd.isnull(dataframe.nosepointx.iloc[frame]) or pd.isnull(dataframe.nosepointy.iloc[frame]):
                if return_cricket_azimuth:
                    return np.nan, np.nan
                return np.nan

        #mean of left and right ear keypoints to get midpoint
        x_midear = np.mean([dataframe.leftearx.iloc[frame], dataframe.rightearx.iloc[frame]])
        y_midear = np.mean([dataframe.lefteary.iloc[frame], dataframe.righteary.iloc[frame]])

        #translating heading based on polar coordinates centered on midear
        try:
            dx, dy = dataframe.nosex.iloc[frame]- x_midear, dataframe.nosey.iloc[frame]-y_midear
        except:
            dx, dy = dataframe.nosepointx.iloc[frame]- x_midear, dataframe.nosepointy.iloc[frame]-y_midear

        _, heading = cart2pol(dx, dy)
        
        computed_object_azimuth = None
        if object_azimuth is not None:
            if isinstance(object_azimuth, list):
                computed_object_azimuth = []
                for obj in object_azimuth:
                    dx, dy = obj[0] - x_midear, obj[1] - y_midear
                    _, az = cart2pol(dx, dy)
                    computed_object_azimuth.append(az)
            elif isinstance(object_azimuth, np.ndarray):
                dx, dy = object_azimuth[0] - x_midear, object_azimuth[1] - y_midear
                _, computed_object_azimuth = cart2pol(dx, dy)

        if return_cricket_azimuth:
            try:
                if pd.isnull(dataframe.stimx.iloc[frame]) or pd.isnull(dataframe.stimy.iloc[frame]):
                    return heading, np.nan
                dx, dy = dataframe.stimx.iloc[frame]- x_midear, dataframe.stimy.iloc[frame]-y_midear
            except:
                if pd.isnull(dataframe.stimulusx.iloc[frame]) or pd.isnull(dataframe.stimulusy.iloc[frame]):
                    return heading, np.nan
                dx, dy = dataframe.stimulusx.iloc[frame]- x_midear, dataframe.stimulusy.iloc[frame]-y_midear
            _, cricket_azimuth = cart2pol(dx, dy)
            if computed_object_azimuth is not None:
                return heading, cricket_azimuth, computed_object_azimuth
            else:
                return heading, cricket_azimuth
        else:
            if computed_object_azimuth is not None:
                return heading, computed_object_azimuth
            else:
                return heading

    # Handle frame_num as tuple (range) or int
    if isinstance(frame_num, tuple):
        start_frame, end_frame = frame_num
        return [_get_single_frame_heading(frame) for frame in range(start_frame, end_frame)]
    else:
        return _get_single_frame_heading(frame_num)
    

def get_approach_events(distance_to_cricket,
                        mouse_speed,
                        heading_cricket_mouse,
                        processed_file,
                        distance_threshold=5,
                        speed_threshold=10,
                        heading_threshold=50,
                        plot=False):
    """
    Detect approach events by working backwards from cricket proximity events.
    
    Args:
        distance_to_cricket: Array of distances from mouse to cricket
        mouse_speed: Array of mouse speeds
        heading_cricket_mouse: Array of heading angles towards cricket
        processed_file: DataFrame with position data for distance calculations
        distance_threshold: Approach detection threshold for distance to cricket (default: 5 cm)
        speed_threshold: Approach detection threshold for mouse speed (default: 5 cm/s)
        heading_threshold: Approach detection threshold for heading angle (default: 50 degrees either side of 0 so 100 degrees total)
        plot: Boolean, whether to create plots (default: False)
        
    Returns:
        Tuple of (complete_approach_events, incomplete_approach_events)
        - complete_approach_events: List of tuples (start_frame, end_frame) for each complete approach event
        - incomplete_approach_events: List of tuples (start_frame, end_frame) for each incomplete approach event
    """
    # Define conditions
    cricket_inprox = distance_to_cricket < distance_threshold
    mouse_moving = mouse_speed > speed_threshold  # Changed threshold to 5 cm/s as per instructions
    mouse_heading = np.abs(heading_cricket_mouse) < heading_threshold
    max_gap = 35 # Maximum gap in frames to consider as single chase instance
    min_distance_approach = 10  # Minimum distance mouse must travel to consider as a valid approach
    distance_diff_threshold = 5  # Maximum distance difference between two chase instances to consider as a single chase

    # Find cricket proximity events
    cricket_prox_diff = np.diff(np.concatenate(([False], cricket_inprox, [False])).astype(int))
    cricket_prox_starts = np.where(cricket_prox_diff == 1)[0]
    cricket_prox_ends = np.where(cricket_prox_diff == -1)[0]

    # For each cricket proximity event, work backwards to find chase start
    chase_events = []

    for prox_start, prox_end in zip(cricket_prox_starts, cricket_prox_ends):
        # Work backwards from cricket proximity start
        chase_start = None
        
        # Look backwards for the start of movement and heading conditions
        for i in range(prox_start - 1, -1, -1):
            # Check if mouse is moving and heading towards cricket
            if mouse_moving[i] and mouse_heading[i]:
                chase_start = i
            else:
                # If we hit a gap, check if it's small enough to bridge
                gap_start = i
                gap_found = False
                
                # Look for the end of the gap (up to max_gap frames)
                for j in range(max(0, i - max_gap), i):
                    if mouse_moving[j] and mouse_heading[j]:
                        # Found movement/heading before the gap
                        if i - j <= max_gap:
                            # Gap is small enough, continue searching backwards
                            chase_start = j
                            gap_found = True
                            break
                
                if not gap_found:
                    # Gap too large or no movement found, stop searching
                    break
        
        if chase_start is not None:
            chase_events.append((chase_start, prox_end))

    # Merge nearby chase events and handle approaches with same start but different ends
    merged_chase_events = []
    if chase_events:
        # Sort chase events by start time, then by end time
        chase_events.sort(key=lambda x: (x[0], x[1]))
        
        current_start, current_end = chase_events[0]
        
        for i in range(1, len(chase_events)):
            next_start, next_end = chase_events[i]
            
            # Check if this is the same approach start (indicating multiple proximity events)
            if next_start == current_start:
                # Same start, extend to the latest end
                current_end = max(current_end, next_end)
            else:
                # Different start, check if chases should be merged based on proximity
                frame_gap = next_start - current_end
                distance_diff = abs(distance_to_cricket[next_start] - distance_to_cricket[current_end])
                
                if frame_gap < max_gap and distance_diff < distance_diff_threshold:
                    # Merge: extend current chase to include next chase
                    current_end = next_end
                else:
                    # Don't merge: save current chase and start new one
                    merged_chase_events.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
        
        # Add the last chase event
        merged_chase_events.append((current_start, current_end))

    # Filter chase events by total distance covered
    filtered_chase_events = []
    for start, end in merged_chase_events:
        # Calculate total distance covered during this chase
        total_distance = get_distance_travelled(processed_file, start, end)
        
        # Only keep chases where total distance covered is >= 10cm
        if total_distance >= min_distance_approach:
            filtered_chase_events.append((start, end))

    # Find incomplete approaches - sequences where mouse is moving and heading towards cricket
    # but doesn't end up in proximity
    incomplete_approaches = []
    
    # Find all sequences where both movement and heading conditions are true
    movement_and_heading = mouse_moving & mouse_heading
    
    # Find start and end of these sequences
    seq_diff = np.diff(np.concatenate(([False], movement_and_heading, [False])).astype(int))
    seq_starts = np.where(seq_diff == 1)[0]
    seq_ends = np.where(seq_diff == -1)[0]
    
    for seq_start, seq_end in zip(seq_starts, seq_ends):
        # Check if this sequence overlaps with any complete approach
        is_complete_approach = False
        for complete_start, complete_end in filtered_chase_events:
            # Check for overlap
            if not (seq_end <= complete_start or seq_start >= complete_end):
                is_complete_approach = True
                break
        
        # If not a complete approach, check if it meets our criteria
        if not is_complete_approach:
            # Calculate total distance covered during this sequence
            total_distance = get_distance_travelled(processed_file, seq_start, seq_end)
            
            # Check if mouse gets close to cricket during this sequence
            min_distance_during_seq = np.min(distance_to_cricket[seq_start:seq_end])
            
            # Only keep if distance covered >= 10cm and mouse doesn't get within proximity threshold
            if total_distance >= min_distance_approach and min_distance_during_seq >= distance_threshold:
                incomplete_approaches.append((seq_start, seq_end))
    
    # Merge nearby incomplete approaches using same logic as complete approaches
    merged_incomplete_approaches = []
    if incomplete_approaches:
        # Sort by start time, then by end time
        incomplete_approaches.sort(key=lambda x: (x[0], x[1]))
        
        current_start, current_end = incomplete_approaches[0]
        
        for i in range(1, len(incomplete_approaches)):
            next_start, next_end = incomplete_approaches[i]
            
            # Check if this is the same approach start
            if next_start == current_start:
                # Same start, extend to the latest end
                current_end = max(current_end, next_end)
            else:
                # Different start, check if approaches should be merged based on proximity
                frame_gap = next_start - current_end
                distance_diff = abs(distance_to_cricket[next_start] - distance_to_cricket[current_end])
                
                if frame_gap < max_gap and distance_diff < distance_diff_threshold:
                    # Merge: extend current approach to include next approach
                    current_end = next_end
                else:
                    # Don't merge: save current approach and start new one
                    merged_incomplete_approaches.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
        
        # Add the last approach
        merged_incomplete_approaches.append((current_start, current_end))
    
    # Adjust approach end times to end the moment before proximity is reached
    adjusted_chase_events = []
    for start, end in filtered_chase_events:
        # Find the first frame in proximity within this approach
        first_proximity_frame = None
        for frame in range(start, end + 1):
            if cricket_inprox[frame]:
                first_proximity_frame = frame
                break
        
        # If we found a proximity frame, end the approach the frame before
        if first_proximity_frame is not None and first_proximity_frame > start:
            adjusted_end = first_proximity_frame - 1
            adjusted_chase_events.append((start, adjusted_end))
        elif first_proximity_frame is None:
            # No proximity found, keep original end
            adjusted_chase_events.append((start, end))
        # If first_proximity_frame == start, this means proximity starts immediately, skip this approach
    
    # Create plots if requested
    if plot:
        # Create chase condition array using adjusted events
        chase_condition = np.zeros_like(distance_to_cricket < 5, dtype=bool)
        for start, end in adjusted_chase_events:
            chase_condition[start:end+1] = True
        
        # Create incomplete approach condition array
        incomplete_condition = np.zeros_like(distance_to_cricket < 5, dtype=bool)
        for start, end in merged_incomplete_approaches:
            incomplete_condition[start:end] = True

        # Convert frame indices to seconds (30 fps)
        time_seconds = np.arange(len(chase_condition)) / 30

        # Plot each condition separately
        fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True)

        # Add scatter plot as first subplot - color red for complete approaches, orange for incomplete
        scatter_colors = np.where(chase_condition, 'lightcoral', 
                                np.where(incomplete_condition, 'orange', 'gray'))
        axes[0].scatter(time_seconds, distance_to_cricket, c=scatter_colors, s=0.5)
        axes[0].axhline(5, color='black', linestyle='dashed')
        axes[0].set_ylabel('Distance to cricket')

        axes[1].plot(time_seconds, cricket_inprox.astype(int), 'r-', alpha=0.7)
        axes[1].set_ylabel('Cricket in proximity')

        axes[2].plot(time_seconds, mouse_moving.astype(int), 'g-', alpha=0.7)
        axes[2].set_ylabel('Mouse moving \n (>5 cm/s)')

        axes[3].plot(time_seconds, mouse_heading.astype(int), 'b-', alpha=0.7)
        axes[3].set_ylabel('Mouse heading \n to cricket')

        axes[4].plot(time_seconds, chase_condition.astype(int), 'k-', linewidth=2)
        axes[4].set_ylabel('Complete Approach')
        
        axes[5].plot(time_seconds, incomplete_condition.astype(int), 'orange', linewidth=2)
        axes[5].set_ylabel('Incomplete Approach')
        axes[5].set_xlabel('Time (seconds)')
    
    return adjusted_chase_events, merged_incomplete_approaches

def get_head_position(dataframe):
    head_x = (dataframe.leftearx + dataframe.rightearx) / 2
    head_y = (dataframe.lefteary + dataframe.righteary) / 2
    
    # If any ear position is NaN, return NaN for both head positions
    nan_mask = np.isnan(dataframe.leftearx) | np.isnan(dataframe.rightearx) | np.isnan(dataframe.lefteary) | np.isnan(dataframe.righteary)
    head_x = np.where(nan_mask, np.nan, head_x)
    head_y = np.where(nan_mask, np.nan, head_y)
    
    return head_x, head_y
    
#distance to target less than 5cm
def get_distance_to_cricket(dataframe):
    head_x, head_y = get_head_position(dataframe)
    return np.sqrt((head_x - dataframe.stimx)**2 + (head_y - dataframe.stimy)**2)

def create_combined_analysis_plot(processed_file,
                                  mouse_x,
                                  mouse_y,
                                  mouse_speed,
                                  cricket_speed,
                                  heading_cricket_mouse,
                                  distance_to_cricket,
                                  approach_events=None,
                                  file_key=None):
    """
    Create a 2x2 combined plot showing mouse-cricket interaction analysis.
    
    Args:
        processed_file: DataFrame with position data
        mouse_x, mouse_y: Mouse position arrays
        mouse_speed, cricket_speed: Speed arrays for mouse and cricket
        heading_cricket_mouse: Mouse heading angles towards cricket
        distance_to_cricket: Distance from mouse to cricket
        approach_events: List of approach event tuples (start_frame, end_frame)
    
    Returns:
        matplotlib figure with 2x2 subplots
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set larger font sizes
    plt.rcParams.update({'font.size': 12})
    
    # Create figure with 2x2 subplots - make it wider
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Mouse-Cricket Interaction Analysis\n{file_key}', fontsize=16, fontweight='bold')
    
    # Plot 1,1: Position scatter plot with speed coloring
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(mouse_x, mouse_y, c=mouse_speed, alpha=0.7, s=10, cmap='viridis')
    scatter2 = ax1.scatter(processed_file.stimx, processed_file.stimy, c=cricket_speed, 
                          alpha=0.7, s=10, cmap='Reds', marker='^')
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_title('Mouse and Cricket Positions\n(colored by speed)', fontsize=14)
    
    # Add colorbars
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8, aspect=20)
    cbar1.set_label('Mouse Speed (cm/s)', fontsize=12)
    cbar2 = plt.colorbar(scatter2, ax=ax1, shrink=0.8, aspect=20, location='right')
    cbar2.set_label('Cricket Speed (cm/s)', fontsize=12)
    
    # Plot 1,2: Approach events analysis (make it wider)
    ax2 = axes[0, 1]
    
    # Get approach events if not provided
    if approach_events is None:
        from livecricket_functions import get_approach_events
        approach_events, incomplete_approach_events = get_approach_events(
            distance_to_cricket, mouse_speed, heading_cricket_mouse, processed_file, plot=False)
    else:
        # If approach_events provided, we need to get incomplete_approach_events separately
        from livecricket_functions import get_approach_events
        _, incomplete_approach_events = get_approach_events(
            distance_to_cricket, mouse_speed, heading_cricket_mouse, processed_file, plot=False)
    
    # Create approach condition arrays
    chase_condition = np.zeros_like(distance_to_cricket < 5, dtype=bool)
    for start, end in approach_events:
        chase_condition[start:end+1] = True
    
    incomplete_condition = np.zeros_like(distance_to_cricket < 5, dtype=bool)
    for start, end in incomplete_approach_events:
        incomplete_condition[start:end] = True
    
    # Convert frame indices to seconds (30 fps)
    time_seconds = np.arange(len(distance_to_cricket)) / 30
    
    # Plot distance with approach events highlighted - bigger dots (s=2)
    # First plot all points in gray
    ax2.scatter(time_seconds, distance_to_cricket, c='gray', s=2, alpha=0.5, label='All data')
    
    # Then overlay complete approaches in red
    complete_mask = chase_condition
    if np.any(complete_mask):
        ax2.scatter(time_seconds[complete_mask], distance_to_cricket[complete_mask], 
                   c='red', s=2, alpha=0.8, label='Complete approaches')
    
    # Then overlay incomplete approaches in orange
    incomplete_mask = incomplete_condition
    if np.any(incomplete_mask):
        ax2.scatter(time_seconds[incomplete_mask], distance_to_cricket[incomplete_mask], 
                   c='orange', s=2, alpha=0.8, label='Incomplete approaches')
    
    ax2.axhline(5, color='black', linestyle='dashed', alpha=0.8, label='Proximity threshold')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Distance to cricket (cm)', fontsize=12)
    ax2.set_title('Distance to Cricket\n(approaches highlighted)', fontsize=14)
    
    # Plot 2,1: Heading histogram with matplotlib default colors
    ax3 = axes[1, 0]
    
    # Get approach heading data
    approach_heading = []
    if approach_events:
        for start, end in approach_events:
            heading_cricket_mouse_approach = heading_cricket_mouse[start:end]
            approach_heading.append(heading_cricket_mouse_approach)
    
    # Plot histograms with default colors
    ax3.hist(heading_cricket_mouse, bins=np.linspace(-180, 180, 36),
             histtype='step', density=True, label='All data', linewidth=2)
    
    if approach_heading:
        ax3.hist(np.concatenate(approach_heading), bins=np.linspace(-180, 180, 36),
                histtype='step', density=True, label='Approach events', linewidth=2)
    
    ax3.set_xlabel('Heading (degrees)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Mouse Heading to Cricket', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 2,2: Speed histogram with pastel colors
    ax4 = axes[1, 1]
    ax4.hist(mouse_speed, bins=np.linspace(0, 40, 50), histtype='step', 
             label='Mouse', density=True, linewidth=2, color='lightblue')
    ax4.hist(cricket_speed, bins=np.linspace(0, 40, 50), histtype='step', 
             label='Cricket', density=True, linewidth=2, color='lightcoral')
    ax4.set_xlim(-0.5, 40)
    ax4.set_xlabel('Speed (cm/s)', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('Speed Distribution', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig