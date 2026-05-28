const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

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

  if (response.status === 204) return null;

  return response.json();
};

// 카메라 목록 조회
export const getAdminCameras = () => {
  return request("/admin/cameras");
};

// 카메라 등록
export const createAdminCamera = ({ id, name }) => {
  return request("/admin/cameras", {
    method: "POST",
    body: JSON.stringify({
      id,
      name,
    }),
  });
};

// 좌석 등록
export const createAdminSeat = ({ name, cameraId, x, y }) => {
  return request("/admin/seats", {
    method: "POST",
    body: JSON.stringify({
      name,
      camera_id: cameraId,
      coord_x: x,
      coord_y: y,
    }),
  });
};

// 좌석 삭제
export const deleteAdminSeat = (seatId) => {
  return request(`/admin/seats/${seatId}`, {
    method: "DELETE",
  });
};

// 전체 좌석 조회
export const getAllSeats = () => {
  return request("/seats");
};

// 특정 카메라 좌석 조회
export const getSeatsByCamera = (cameraId) => {
  return request(`/seats/${cameraId}`);
};