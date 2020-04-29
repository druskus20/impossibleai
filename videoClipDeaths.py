import cv2
import numpy as np

def get_attempt_area(frame):
    return frame[0:100, 390:480]       



def attempt_has_changed(frame1_attempt_area, frame2_attempt_area, tolerance=0): # !!!! TOL
    
    frame1_attempt_area = cv2.inRange(frame1_attempt_area, np.asarray([220, 220, 220]), np.asarray([255, 255, 255]))
    frame2_attempt_area = cv2.inRange(frame2_attempt_area, np.asarray([220, 220, 220]), np.asarray([255, 255, 255]))

    diff = frame1_attempt_area - frame2_attempt_area
    
    m_norm = np.sum(abs(diff))         # Manhattan norm
    #z_norm = norm(diff.ravel(), 0)    # Zero norm
   
    if m_norm > tolerance:
        return True
    return False

# Returns a list with the frames in which attempt # changes
def videoFindDeaths(vidPath):
    cap = cv2.VideoCapture(vidPath)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    dimmensions = size + (3, )

    frame =  np.zeros(dimmensions, dtype=np.int8)
    last_frame =  np.zeros(dimmensions, dtype=np.int8)

    deathFrames = []
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        attempt_area_1 = get_attempt_area(frame)
        attempt_area_2 = get_attempt_area(last_frame)

        cv2.waitKey(1) 
        if attempt_has_changed(attempt_area_1, attempt_area_2):
            print("Attempt has changed "  + str(frame_number))
            deathFrames.append(frame_number)

        last_frame = frame
    return deathFrames, total_frames


def get_video_segment(videoPath, skip_prev_len=20, skip_next_len=20):
    vidPath = videoPath
   

    # Build the segments
    segRange = []
    offset = 0   
    deathFrames, total_frames = videoFindDeaths(vidPath)    
    
    for death in deathFrames:
        if (death-skip_prev_len < offset):
            offset = death+skip_next_len
        else:
            segRange.append((offset, (death-skip_prev_len)))
            offset = death+skip_next_len
            # If deaths overlap (offset has to be higher than 0)
            if death-skip_prev_len < offset:
                offset = death+skip_next_len
    

    if offset < total_frames:
        segRange.append((offset, total_frames-1))

    if len(segRange) == 0:
        return None
    return segRange

def videoClipDeaths(videoPath, keypresses_path, skip_prev_len=20, skip_next_len=20, cut_attempt=True, savefiles=['keypresses_nodeaths.npz', 'frames_nodeaths.npz', 'video_nodeaths.mp4']):

    vidPath = videoPath
    shotsPath = savefiles[2]
    keypresses = np.load(keypresses_path,  allow_pickle=True)['arr_0']
    
    segRange = get_video_segment(vidPath, skip_prev_len, skip_next_len)
    


    if not segRange:
        print("Empty segment range")
        return None
    print (f"Segments: {segRange}")


    ## CUT KEYS ###
    new_keypresses = []
    for r in segRange:
        new_keypresses.extend(keypresses[int(r[0]):int(r[1])])

    np.savez_compressed(savefiles[0], np.asarray(new_keypresses))

    ### CUT VIDEO ###
    # Open the video
    cap = cv2.VideoCapture(vidPath)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    
    
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if cut_attempt == True:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))-200,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))-100)
       
    new_frames =  []
    dimmensions = size + (3, )
    frame =  np.zeros(dimmensions, dtype=np.int8)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(shotsPath, fourcc, fps, size)
    
    # Iterate through the frames of the video from segRange
    for idx, (begFidx, endFidx) in enumerate(segRange):
        cap.set(cv2.CAP_PROP_POS_FRAMES, begFidx)
        ret = True  # has frame returned
        while (cap.isOpened() and ret and writer.isOpened()):
            ret, frame = cap.read()
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1

                
            # Cut attempt area
            if cut_attempt == True and frame is not None:
                frame = frame[100:, 200:]   

            # End contition
            if frame_number < endFidx:
                writer.write(frame)
                new_frames.append(frame)
            else:
                break

    writer.release()
    np.savez_compressed(savefiles[1], np.asarray(new_frames), allow_pickle=True)

    # Check how many frame that new.avi has
    cap2 = cv2.VideoCapture(shotsPath)



def resize_image(image, res=(520//2, 476//2)):
    print(image.shape)
    pr_image = cv2.resize(image, res)
    print(pr_image.shape)
 
    return np.asarray(pr_image, dtype=np.uint8)

def group_frames(framesPath, keypressesPath,   savefile='processed_data.npz'):
    keypresses = np.load(keypressesPath,  allow_pickle=True)['arr_0']
    frames = np.load(framesPath,  allow_pickle=True)['arr_0']
    data = []
    # Groups 5 frames and a keypress
    i = 0
    while (i+3 < len(frames)): # Discards last elements if necessary
        f1 = resize_image(frames[i])
        f2 = resize_image(frames[i+1])
        f3 = resize_image(frames[i+2])
        if (any(keypresses[i:i+3])):
            key = 1
        else:
            key = 0

        framegroup = [f1, f2, f3, np.asarray([key])]
        data.append(framegroup)
        i+=3
    
   
    np.savez_compressed(savefile, np.asarray(data), allow_pickle=True)
    print(f"Frames agrupados, dataset guardado en {savefile}")