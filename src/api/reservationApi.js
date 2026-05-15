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

// 좌석 예약
export const createReservation = ({ seatId }) => {
  return request("/reservations", {
    method: "POST",
    body: JSON.stringify({
      seat_id: seatId,
    }),
  });
};
// 내 좌석 현황 추가 예정