from pathlib import Path
import exifread
from pymediainfo import MediaInfo

def extract_image_metadata(file_path: Path) -> dict:
    metadata = {}
    try:
        with open(file_path, "rb") as f:
            tags = exifread.process_file(f, details=True)

        for tag, value in tags.items():
            metadata[tag] = str(value)

    except Exception as e:
        metadata["error"] = str(e)

    return metadata


def extract_video_metadata(file_path: Path) -> dict:
    metadata = {}
    try:
        media_info = MediaInfo.parse(file_path)

        for track in media_info.tracks:
            track_data = track.to_data()
            metadata[track.track_type] = track_data

    except Exception as e:
        metadata["error"] = str(e)

    return metadata
