import streamlit as st
import cv2
import os
import imutils
import numpy as np
import time
from skimage import io, color
from moviepy import video

def colorize_video(path_to_video):
    """Colorize a video using the Caffe model.

    Args:
        path_to_video: The path to the video file.

    Returns:
        A NumPy array containing the colorized video.
    """

    path_to_model = r"./model"
    protoPath = os.path.sep.join([path_to_model, "colorization_deploy_v2.prototxt"])
    modelPath = os.path.sep.join([path_to_model, "colorization_release_v2.caffemodel"])
    pointPath = os.path.sep.join([path_to_model, "pts_in_hull.npy"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    cap = cv2.VideoCapture(path_to_video)
    frame_counter = 0
    writer = None

    class8_ab = net.getLayerId("class8_ab")
    conv8_313_rh = net.getLayerId("conv8_313_rh")
    points = np.load(pointPath).transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8_ab).blobs = [points.astype("float32")]
    net.getLayer(conv8_313_rh).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    prevFrametime = 0
    nextFrametime = 0

    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        frame = imutils.resize(frame, 400)
        Lab = cv2.cvtColor(frame.astype("float32") / 255.0, cv2.COLOR_BGR2LAB)
        Lab_resized = cv2.resize(Lab, (224, 224))
        L = cv2.split(Lab_resized)[0]-50
        net.setInput(cv2.dnn.blobFromImage(L))
        a_b = net.forward()[0, :, :, :].transpose((1, 2, 0))
        a_b = cv2.resize(a_b, (frame.shape[1], frame.shape[0]))
        L = cv2.split(Lab)[0]
        updated_Lab = np.concatenate((L[:, :, np.newaxis], a_b), axis=2)
        output = cv2.cvtColor(updated_Lab, cv2.COLOR_LAB2BGR)
        output = (255 * np.clip(output, 0, 1)).astype("uint8")

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("color.avi", fourcc, 20,
                                    (output.shape[1], output.shape[0]), True)

        if writer is not None:
            writer.write(output)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

    return output

def main():
    """The main function of the Streamlit app."""

    st.title("Video Colorization")

    # Upload video
    vid_data = st.file_uploader("Upload a video to colorize:")

    # Colorize video
    if vid_data is not None:
        path_to_video = os.path.join("./temp", "userinput.mp4")
        with open(path_to_video, "wb") as f:
            f.write(vid_data.read())

        output = colorize_video(path_to_video)

        # Display output
        st.video(output, caption="Colorized video")

if __name__ == "__main__":
    main()
