import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="陸上動作解析アプリ", layout="wide")
st.title("🏃‍♂️ 陸上競技 動作解析＆グラフ可視化ツール")
st.caption("動画をアップロードすると関節角度・動きを解析し、解析動画の再生・2D/3D可視化・CSVダウンロードができます。")

# --- 部員限定アクセス制御 ---
EXPECTED_PASSCODE = os.environ.get("APP_PASSCODE", "4592026")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.subheader("🔒 部員限定ログイン")
    st.write("陸上競技部の部員に共有された合言葉を入力してください。")
    entered_code = st.text_input("合言葉", type="password")
    if st.button("入室する"):
        if entered_code == EXPECTED_PASSCODE:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("合言葉が違います。")
    st.stop()

if st.sidebar.button("ログアウト"):
    st.session_state.authenticated = False
    st.rerun()
# --- ここまで ---

# MediaPipe Pose 33点の主要インデックス
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]

JOINT_NAMES = {
    0: "頭（鼻）",
    11: "左肩",
    12: "右肩",
    13: "左肘",
    14: "右肘",
    15: "左手首",
    16: "右手首",
    23: "左腰",
    24: "右腰",
    25: "左膝",
    26: "右膝",
    27: "左足首",
    28: "右足首",
}

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}


@st.cache_resource(show_spinner="姿勢推定モデルを準備中...")
def get_model_path(model_type: str) -> str:
    """公式の .task モデルファイルをローカルに取得してパスを返す"""
    model_dir = Path(tempfile.gettempdir()) / "mediapipe_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / f"pose_landmarker_{model_type}.task"
    if not target.exists() or target.stat().st_size == 0:
        urllib.request.urlretrieve(MODEL_URLS[model_type], target)
    return str(target)


def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """3点(a, b, c)から b を頂点とする角度(度)を計算する"""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return np.nan
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def draw_pose_overlay(rgb_frame: np.ndarray, landmarks) -> np.ndarray:
    """検出されたランドマークと骨格線をRGB画像に描画する"""
    h, w, _ = rgb_frame.shape
    canvas = rgb_frame.copy()
    points = {}

    for idx, lm in enumerate(landmarks):
        px = int(np.clip(lm.x * w, 0, w - 1))
        py = int(np.clip(lm.y * h, 0, h - 1))
        points[idx] = (px, py)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in points and end_idx in points:
            cv2.line(canvas, points[start_idx], points[end_idx], (0, 220, 120), 2)

    for px, py in points.values():
        cv2.circle(canvas, (px, py), 4, (255, 80, 60), -1)

    return canvas


def transcode_to_h264(raw_mp4_path: str, fps: float) -> bytes:
    """ブラウザ再生しやすいH.264(yuv420p)のMP4バイト列に変換する"""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        out_path = raw_mp4_path + "_h264.mp4"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            raw_mp4_path,
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and Path(out_path).exists() and Path(out_path).stat().st_size > 0:
            data = Path(out_path).read_bytes()
            Path(out_path).unlink(missing_ok=True)
            Path(raw_mp4_path).unlink(missing_ok=True)
            return data

    data = Path(raw_mp4_path).read_bytes() if Path(raw_mp4_path).exists() else b""
    Path(raw_mp4_path).unlink(missing_ok=True)
    return data


@st.cache_data(show_spinner="動画を解析中...")
def analyze_video(video_bytes: bytes, model_type: str, min_det_conf: float):
    """動画バイト列を受け取り、関節角度データ・3Dワールド座標・骨格描画済み動画バイト列を返す"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_bytes)
        tmp_path = tmp_file.name

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_out_file = tempfile.NamedTemporaryFile(delete=False, suffix="_raw.mp4")
    raw_out_path = raw_out_file.name
    raw_out_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    model_path = get_model_path(model_type)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=min_det_conf,
        min_pose_presence_confidence=min_det_conf,
        min_tracking_confidence=0.5,
    )

    records = []
    world_frames = {}
    detected_frames = 0
    frame_idx = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h_curr, w_curr = frame.shape[:2]
            out_w = width if width > 0 else w_curr
            out_h = height if height > 0 else h_curr
            if writer is None:
                writer = cv2.VideoWriter(raw_out_path, fourcc, fps, (out_w, out_h))

            time_sec = frame_idx / fps
            timestamp_ms = int(round(time_sec * 1000))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            row = {
                "frame": frame_idx,
                "time_sec": round(time_sec, 4),
                "left_knee_deg": np.nan,
                "right_knee_deg": np.nan,
                "left_hip_deg": np.nan,
                "right_hip_deg": np.nan,
                "left_elbow_deg": np.nan,
                "right_elbow_deg": np.nan,
                "hip_center_y": np.nan,
                "head_y": np.nan,
            }

            draw_rgb = rgb
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                detected_frames += 1
                lm = result.pose_landmarks[0]

                def pt(idx: int) -> np.ndarray:
                    return np.array([lm[idx].x, lm[idx].y])

                nose = pt(NOSE)
                l_sh = pt(LEFT_SHOULDER)
                r_sh = pt(RIGHT_SHOULDER)
                l_el = pt(LEFT_ELBOW)
                r_el = pt(RIGHT_ELBOW)
                l_wr = pt(LEFT_WRIST)
                r_wr = pt(RIGHT_WRIST)
                l_hp = pt(LEFT_HIP)
                r_hp = pt(RIGHT_HIP)
                l_kn = pt(LEFT_KNEE)
                r_kn = pt(RIGHT_KNEE)
                l_an = pt(LEFT_ANKLE)
                r_an = pt(RIGHT_ANKLE)

                row["left_knee_deg"] = calc_angle(l_hp, l_kn, l_an)
                row["right_knee_deg"] = calc_angle(r_hp, r_kn, r_an)
                row["left_hip_deg"] = calc_angle(l_sh, l_hp, l_kn)
                row["right_hip_deg"] = calc_angle(r_sh, r_hp, r_kn)
                row["left_elbow_deg"] = calc_angle(l_sh, l_el, l_wr)
                row["right_elbow_deg"] = calc_angle(r_sh, r_el, r_wr)
                row["hip_center_y"] = float((l_hp[1] + r_hp[1]) / 2.0)
                row["head_y"] = float(nose[1])

                draw_rgb = draw_pose_overlay(rgb, lm)

                # 3Dワールド座標の保存（表示用にY軸を上向き＋に反転）
                if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                    wlm = result.pose_world_landmarks[0]
                    world_frames[frame_idx] = {
                        idx: (float(wlm[idx].x), float(-wlm[idx].y), float(wlm[idx].z))
                        for idx in JOINT_NAMES.keys()
                    }

            out_bgr = cv2.cvtColor(draw_rgb, cv2.COLOR_RGB2BGR)
            if (out_bgr.shape[1], out_bgr.shape[0]) != (out_w, out_h):
                out_bgr = cv2.resize(out_bgr, (out_w, out_h))
            writer.write(out_bgr)

            records.append(row)
            frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    Path(tmp_path).unlink(missing_ok=True)

    annotated_video_bytes = transcode_to_h264(raw_out_path, fps)
    df = pd.DataFrame(records)
    return df, world_frames, annotated_video_bytes, fps, detected_frames


# サイドバー：解析設定
st.sidebar.header("解析設定")
model_type = st.sidebar.selectbox(
    "推定モデル",
    options=["lite", "full"],
    index=0,
    format_func=lambda x: {"lite": "lite: 高速（推奨）", "full": "full: 高精度（処理時間長め）"}[x],
)
min_det_conf = st.sidebar.slider("人物検出のしきい値", 0.2, 0.9, 0.5, 0.05)

# メイン：動画アップロード
uploaded_file = st.file_uploader(
    "自分の走行・跳躍・投擲などの動画をアップロード（MP4 / MOV / AVI）",
    type=["mp4", "mov", "avi", "m4v"],
)

if uploaded_file is not None:
    df, world_frames, annotated_video_bytes, fps, detected_frames = analyze_video(
        uploaded_file.getvalue(),
        model_type=model_type,
        min_det_conf=min_det_conf,
    )

    st.info(f"FPS: {fps:.1f} ｜ 総フレーム数: {len(df)} ｜ 骨格検出できたフレーム: {detected_frames}")

    if detected_frames == 0:
        st.warning("骨格が検出できませんでした。全身が映っている動画か、検出のしきい値を下げてお試しください。")

    st.subheader("骨格描画付き 解析動画の再生")
    if annotated_video_bytes:
        st.video(annotated_video_bytes, format="video/mp4")
        base_name = uploaded_file.name.rsplit(".", 1)[0]
        st.download_button(
            label="🎬 解析済み動画をダウンロード (MP4)",
            data=annotated_video_bytes,
            file_name=f"{base_name}_annotated.mp4",
            mime="video/mp4",
        )
    else:
        st.error("解析済み動画の生成に失敗しました。")

    st.subheader("グラフ・可視化モードの選択")
    view_mode = st.radio(
        "表示形式を選んでください",
        options=["2D時系列グラフ（角度・上下動）", "3D骨格・軌跡ビュー（回転・立体確認）"],
        horizontal=True,
    )

    max_t = float(df["time_sec"].max()) if len(df) > 0 else 0.0
    if max_t > 0:
        t_min, t_max = st.slider(
            "表示する時間範囲（秒）",
            min_value=0.0,
            max_value=round(max_t, 2),
            value=(0.0, round(max_t, 2)),
            step=0.05,
        )
        plot_df = df[(df["time_sec"] >= t_min) & (df["time_sec"] <= t_max)]
    else:
        plot_df = df

    metric_labels = {
        "left_knee_deg": "左膝角度 (度)",
        "right_knee_deg": "右膝角度 (度)",
        "left_hip_deg": "左股関節角度 (度)",
        "right_hip_deg": "右股関節角度 (度)",
        "left_elbow_deg": "左肘角度 (度)",
        "right_elbow_deg": "右肘角度 (度)",
        "hip_center_y": "腰中心の高さ（正規化Y座標・小が上）",
        "head_y": "頭の高さ（鼻Y座標・小が上）",
    }

    if view_mode == "2D時系列グラフ（角度・上下動）":
        selected_metrics = st.multiselect(
            "グラフに重ねて表示する項目を選択してください（複数選択可）",
            options=list(metric_labels.keys()),
            default=["left_knee_deg", "right_knee_deg"],
            format_func=lambda k: metric_labels[k],
        )

        if selected_metrics and not plot_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            for col in selected_metrics:
                ax.plot(
                    plot_df["time_sec"],
                    plot_df[col],
                    marker="o",
                    markersize=2.5,
                    linewidth=1.5,
                    label=metric_labels[col],
                )
            ax.set_xlabel("Time (sec)")
            ax.set_ylabel("Value (deg / norm Y)")
            ax.set_title("Motion Analysis Time Series")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.write("表示する項目を1つ以上選択してください。")
    else:
        selected_metrics = []
        valid_frames = [int(f) for f in plot_df["frame"] if int(f) in world_frames]

        if not valid_frames:
            st.warning("選択した時間範囲に3D骨格データがありません。")
        else:
            st.caption("※ 腰中心を原点とした3D推定姿勢です。ドラッグで回転、スクロールで拡大縮小できます。")
            target_frame = st.select_slider(
                "3D骨格を表示するフレームを選択",
                options=valid_frames,
                value=valid_frames[len(valid_frames) // 2],
                format_func=lambda f: f"Frame {f} ({f / fps:.2f} 秒)",
            )

            trail_joints = st.multiselect(
                "時間範囲内の移動軌跡を線で表示する部位（円盤投げなら手首や頭がおすすめ）",
                options=list(JOINT_NAMES.keys()),
                default=[RIGHT_WRIST, LEFT_WRIST, NOSE],
                format_func=lambda idx: JOINT_NAMES[idx],
            )

            fig3d = go.Figure()

            # 指定フレームの3D骨格
            pts = world_frames[target_frame]
            bone_x, bone_y, bone_z = [], [], []
            for s_idx, e_idx in POSE_CONNECTIONS:
                if s_idx in pts and e_idx in pts:
                    bone_x.extend([pts[s_idx][0], pts[e_idx][0], None])
                    bone_y.extend([pts[s_idx][1], pts[e_idx][1], None])
                    bone_z.extend([pts[s_idx][2], pts[e_idx][2], None])

            fig3d.add_trace(
                go.Scatter3d(
                    x=bone_x,
                    y=bone_y,
                    z=bone_z,
                    mode="lines",
                    line=dict(color="#00CC66", width=6),
                    name="骨格ライン",
                )
            )

            joint_ids = list(pts.keys())
            fig3d.add_trace(
                go.Scatter3d(
                    x=[pts[i][0] for i in joint_ids],
                    y=[pts[i][1] for i in joint_ids],
                    z=[pts[i][2] for i in joint_ids],
                    mode="markers",
                    marker=dict(size=5, color="#FF4B4B"),
                    text=[JOINT_NAMES[i] for i in joint_ids],
                    hoverinfo="text",
                    name="関節ポイント",
                )
            )

            # 選択部位の時間軌跡
            for j_idx in trail_joints:
                tx = [world_frames[f][j_idx][0] for f in valid_frames if j_idx in world_frames[f]]
                ty = [world_frames[f][j_idx][1] for f in valid_frames if j_idx in world_frames[f]]
                tz = [world_frames[f][j_idx][2] for f in valid_frames if j_idx in world_frames[f]]
                fig3d.add_trace(
                    go.Scatter3d(
                        x=tx,
                        y=ty,
                        z=tz,
                        mode="lines+markers",
                        marker=dict(size=2),
                        line=dict(width=3),
                        name=f"{JOINT_NAMES[j_idx]} の軌跡",
                    )
                )

            fig3d.update_layout(
                height=620,
                scene=dict(
                    xaxis_title="左右 X (m)",
                    yaxis_title="上下 Y (m)",
                    zaxis_title="前後奥行き Z (m)",
                    aspectmode="data",
                ),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig3d, use_container_width=True)

    st.subheader("解析データのダウンロード")
    export_cols = ["frame", "time_sec"] + (selected_metrics if selected_metrics else list(metric_labels.keys()))
    export_df = df[export_cols]
    csv_data = export_df.to_csv(index=False).encode("utf-8-sig")

    base_name = uploaded_file.name.rsplit(".", 1)[0]
    st.download_button(
        label="📥 解析データをCSVでダウンロード",
        data=csv_data,
        file_name=f"{base_name}_motion_analysis.csv",
        mime="text/csv",
    )

    with st.expander("数値データテーブルを確認する"):
        st.dataframe(export_df, use_container_width=True)
