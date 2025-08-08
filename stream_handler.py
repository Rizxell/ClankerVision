import cv2
import subprocess
import time
import numpy as np
from yt_dlp import YoutubeDL

def youtube_handler(source, width=640, height=360):
    """Handle YouTube stream using yt_dlp + ffmpeg"""
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'noplaylist': True,
        'simulate': True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(source, download=False)
            if 'url' in info_dict:
                stream_url = info_dict['url']
            else:
                raise Exception("Tidak Bisa Ambil Stream URL YouTube")
    except Exception as e:
        raise Exception(f"Error yt_dlp: {e}")

    ffmpeg_cmd = [
        'ffmpeg',
        '-i', stream_url,
        '-loglevel', 'quiet',
        '-an',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-vf', f'scale={width}:{height}',
        '-'
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1)
    if process.poll() is not None and process.returncode != 0:
        stderr_output = process.stderr.read().decode('utf-8')
        raise Exception(f"FFmpeg gagal memulai. Error: {stderr_output}")
    return process, width, height

def m3u8_handler(source, width=640, height=360):
    """Handle HTTP Live Streaming (.m3u8) via ffmpeg"""
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', source,
        '-loglevel', 'quiet',
        '-an',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-vf', f'scale={width}:{height}',
        '-'
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1)
    if process.poll() is not None and process.returncode != 0:
        stderr_output = process.stderr.read().decode('utf-8')
        raise Exception(f"FFmpeg gagal memulai. Error: {stderr_output}")
    return process, width, height

def rtsp_handler(source, width=640, height=360):
    """Handle RTSP stream via ffmpeg"""
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', source,
        '-loglevel', 'quiet',
        '-an',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-vf', f'scale={width}:{height}',
        '-'
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1)
    if process.poll() is not None and process.returncode != 0:
        stderr_output = process.stderr.read().decode('utf-8')
        raise Exception(f"FFmpeg gagal memulai. Error: {stderr_output}")
    return process, width, height

def flv_handler(source, width=640, height=360):
    """Handle FLV stream via ffmpeg"""
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', source,
        '-loglevel', 'quiet',
        '-an',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-vf', f'scale={width}:{height}',
        '-'
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1)
    if process.poll() is not None and process.returncode != 0:
        stderr_output = process.stderr.read().decode('utf-8')
        raise Exception(f"FFmpeg gagal memulai. Error: {stderr_output}")
    return process, width, height

def file_handler(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise Exception(f"Gagal membuka video/file: {source}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, w, h

def open_stream(source, width=640, height=360):
    """
    Pilih handler berdasarkan jenis sumber stream.
    """
    src_l = source.lower()
    if "youtube.com" in src_l or "youtu.be" in src_l:
        return youtube_handler(source, width, height)
    elif src_l.endswith(".m3u8") or "m3u8" in src_l:
        return m3u8_handler(source, width, height)
    elif src_l.startswith("rtsp://"):
        return rtsp_handler(source, width, height)
    elif src_l.endswith(".flv") or "flv" in src_l:
        return flv_handler(source, width, height)
    elif src_l.startswith("http"):
        # Bisa file mp4/http file
        if "m3u8" in src_l:
            return m3u8_handler(source, width, height)
        elif "flv" in src_l:
            return flv_handler(source, width, height)
        else:
            return file_handler(source)
    else:
        # File lokal
        return file_handler(source)

def read_frame(stream, width, height):
    """
    Baca frame dari handler (ffmpeg = raw, cv2 = .read())
    """
    if isinstance(stream, subprocess.Popen):
        raw_frame_size = width * height * 3
        raw_frame = stream.stdout.read(raw_frame_size)
        if not raw_frame or len(raw_frame) != raw_frame_size:
            return False, None
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3)).copy()
        return True, frame
    else:
        return stream.read()