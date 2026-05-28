const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const request = async (endpoint, options = {}) => {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `API 요청 실패: ${response.status}`);
  }

  return response.json();
};

// 전체 좌석 조회
export const getAllSeats = () => {
  return request("/seats");
};

// 특정 카메라 좌석 조회
export const getSeatsByCamera = (cameraId) => {
  return request(`/seats/${cameraId}`);
};

// 특정 좌석 상세 조회
export const getSeatDetail = (cameraId, seatId) => {
  return request(`/seats/${cameraId}/${seatId}`);
};