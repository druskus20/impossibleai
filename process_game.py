from videoClipDeaths import videoClipDeaths, group_frames




videoClipDeaths("video_raw.mp4", 'keypresses_raw.npz', 100, 15)
group_frames("frames_nodeaths.npz", 'keypresses_nodeaths.npz')
