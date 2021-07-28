import copy
import threading

import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode, ClientSettings

import av
import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from main import draw_landmarks, draw_stick_figure


class Tokyo2020PictogramVideoProcessor(VideoProcessorBase):
    def __init__(self, static_image_mode,
                    model_complexity,
                    min_detection_confidence,
                    min_tracking_confidence,rev_color,display_mode) -> None:
        mp_pose = mp.solutions.pose
        self._pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._cvFpsCalc = CvFpsCalc(buffer_len=10)

        self.rev_color = rev_color
        self.display_mode = display_mode

        self._lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        display_fps = self._cvFpsCalc.get()

        # 色指定
        if self.rev_color:
            color = (255, 255, 255)
            bg_color = (100, 33, 3)
        else:
            color = (100, 33, 3)
            bg_color = (255, 255, 255)

        # カメラキャプチャ #####################################################
        image = frame.to_ndarray(format="bgr24")

        image = cv.flip(image, 1)  # ミラー表示
        debug_image01 = copy.deepcopy(image)
        debug_image02 = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        cv.rectangle(debug_image02, (0, 0), (image.shape[1], image.shape[0]),
                    bg_color,
                    thickness=-1)


        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        with self._lock:
            results = self._pose.process(image)

        # 描画 ################################################################
        if results.pose_landmarks is not None:
            # 描画
            debug_image01 = draw_landmarks(
                debug_image01,
                results.pose_landmarks,
            )
            debug_image02 = draw_stick_figure(
                debug_image02,
                results.pose_landmarks,
                color=color,
                bg_color=bg_color,
            )

        cv.putText(debug_image01, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(debug_image02, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv.LINE_AA)

        if self.display_mode == "Pose":
            return av.VideoFrame.from_ndarray(debug_image01, format="bgr24")
        elif self.display_mode == "Pictogram":
            return av.VideoFrame.from_ndarray(debug_image02, format="bgr24")
        elif self.display_mode == "Both":
            new_image = np.zeros(image.shape, dtype=np.uint8)
            h, w = image.shape[0:2]
            half_h = h // 2
            half_w = w // 2

            offset_y = h // 4
            new_image[offset_y: offset_y + half_h, 0: half_w, :] = cv.resize(debug_image02, (half_w, half_h))
            new_image[offset_y: offset_y + half_h, half_w:, :] = cv.resize(debug_image01, (half_w, half_h))
            return av.VideoFrame.from_ndarray(new_image, format="bgr24")


def main():
    with st.beta_expander("Model parameters (there parameters are effective only at initialization)"):
        static_image_mode = st.checkbox("Static image mode")
        model_complexity = st.radio("Model complexity", [0, 1, 2], index=1)
        min_detection_confidence = st.slider("Min detection confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        min_tracking_confidence = st.slider("Min tracking confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    rev_color = st.checkbox("Reverse color")
    display_mode = st.radio("Display mode", ["Pictogram", "Pose", "Both"], index=0)

    def processor_factory():
        return Tokyo2020PictogramVideoProcessor(static_image_mode=static_image_mode, model_complexity=model_complexity, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence, rev_color=rev_color, display_mode=display_mode)

    webrtc_ctx = webrtc_streamer(
        key="tokyo2020-Pictogram",
        mode=WebRtcMode.SENDRECV,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        ),
        video_processor_factory=processor_factory,
        async_transform=True,
    )
    st.session_state["started"] = webrtc_ctx.state.playing

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.rev_color = rev_color
        webrtc_ctx.video_processor.display_mode = display_mode

if __name__ == "__main__":
    main()
