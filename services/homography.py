import cv2
import numpy as np
from typing import Tuple, List


def compute_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    """Bounding box의 중심점(하단 중심) 반환.
    
    사람 탐지에서는 발 위치가 좌석 점유와 더 연관성이 높으므로
    상자의 하단 중심을 대표 좌표로 사용합니다.
    """
    cx = (x1 + x2) / 2
    cy = y2  # 하단 중심 (발 위치)
    return cx, cy


def compute_homography(point_pairs: List[dict]) -> List[float]:
    """
    대응점 4쌍으로 Homography 행렬을 계산합니다.

    Args:
        point_pairs: [{"src": [x, y], "dst": [x, y]}, ...] 4개

    Returns:
        9개 값(행 우선 flatten)으로 표현된 3x3 행렬
    """
    if len(point_pairs) < 4:
        raise ValueError("Homography 계산에는 최소 4쌍의 대응점이 필요합니다.")

    src_pts = np.array([[p["src"][0], p["src"][1]] for p in point_pairs], dtype=np.float32)
    dst_pts = np.array([[p["dst"][0], p["dst"][1]] for p in point_pairs], dtype=np.float32)

    H, mask = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        raise ValueError("Homography 행렬 계산 실패 — 대응점이 일직선이거나 중복됩니다.")

    return H.flatten().tolist()  # 9개 값으로 반환


def apply_homography(
    homography_matrix: List[float],
    cx: float,
    cy: float
) -> Tuple[float, float]:
    """
    3x3 Homography 행렬을 적용해 카메라 좌표 → 좌석 좌표계로 변환.

    Args:
        homography_matrix: 9개 값(행 우선 flatten)으로 표현된 3x3 행렬
        cx, cy: 카메라 이미지 상의 좌표

    Returns:
        (mapped_x, mapped_y): 좌석 좌표계에서의 위치
    """
    H = np.array(homography_matrix, dtype=np.float64).reshape(3, 3)
    point = np.array([cx, cy, 1.0], dtype=np.float64)

    transformed = H @ point          # 동차 좌표 변환
    w = transformed[2]
    if abs(w) < 1e-8:
        raise ValueError(f"Homography 변환 결과 w=0 (특이점): point=({cx}, {cy})")

    mapped_x = transformed[0] / w
    mapped_y = transformed[1] / w
    return float(mapped_x), float(mapped_y)