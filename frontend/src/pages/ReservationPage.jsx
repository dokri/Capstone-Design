import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./ReservationPage.css";
import chairIcon from "../assets/chair.svg";
import { createReservation } from "../api/reservationApi";

const SEAT_TYPES = {
  using: {
    label: "사용중",
    color: "#00BC3F",
    bgColor: "#C3FFD7",
    icon: chairIcon,
    filter:
      "invert(42%) sepia(80%) saturate(2850%) hue-rotate(125deg) brightness(105%) contrast(120%)",
  },
  vacant: {
    label: "미사용",
    color: "#454545",
    bgColor: "#EFEFEF",
    icon: chairIcon,
    filter: "none",
  },
  auto: {
    label: "자동반납",
    color: "#D25354",
    bgColor: "#FDE9E8",
    icon: chairIcon,
    filter:
      "invert(38%) sepia(74%) saturate(2476%) hue-rotate(340deg) brightness(95%) contrast(105%)",
  },
  temp: {
    label: "일시비움",
    color: "#FFC200",
    bgColor: "#FEF2CC",
    icon: chairIcon,
    filter:
      "invert(85%) sepia(85%) saturate(1500%) hue-rotate(10deg) brightness(105%) contrast(105%)",
  },
};

const ReservationPage = ({
  seats,
  setSeats,
  reservedSeatId,
  setReservedSeatId,
}) => {
  const navigate = useNavigate();

  const [selectedSeat, setSelectedSeat] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSeatClick = (seat) => {
    setSelectedSeat(seat);
  };

  const closeModal = () => {
    if (isLoading) return;
    setSelectedSeat(null);
  };

  const getSeatId = (seat) => seat.id;

  const getSeatNumber = (seat) => {
    return seat.id;
  };

  const getSeatStatus = (seat) => {
    return seat.status || "vacant";
  };

  const getSeatStateText = (status) => {
    if (status === "vacant") return "예약 가능";
    if (status === "using") return "사용중";
    if (status === "auto") return "자동반납";
    if (status === "temp") return "일시비움";
    return "미확인";
  };

  const isReservable = (status) => status === "vacant";

  const handleConfirmReservation = async () => {
    if (!selectedSeat || getSeatStatus(selectedSeat) !== "vacant") return;

    try {
      setIsLoading(true);

      let response = null;

      try {
        response = await createReservation({
          seatId: getSeatId(selectedSeat),
        });
      } catch (error) {
        console.warn("백엔드 예약 실패, 프론트 상태로 대체:", error);
      }

      setSeats((prevSeats) =>
        prevSeats.map((seat) =>
          getSeatId(seat) === getSeatId(selectedSeat)
            ? {
                ...seat,
                status: "using",
                is_booked: true,
                isBooked: true,
              }
            : seat
        )
      );

      setReservedSeatId(response?.seat_id || getSeatId(selectedSeat));
      alert("좌석 예약이 완료되었습니다.");
      setSelectedSeat(null);
      navigate("/");
    } catch (error) {
      console.error("예약 처리 중 오류:", error);
      alert("예약 처리 중 오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <div className="content-header"></div>
      <h1 className="header-title">도서관 좌석 관리 대시보드</h1>

      <div className="dashboard-card reservation-dashboard-card">
        <h2 className="subtitle">제4 열람실 (4F)</h2>

        <div className="seat-grid reservation-seat-grid">
          {seats.map((seat, idx) => {
            const status = getSeatStatus(seat);
            const config = SEAT_TYPES[status] || SEAT_TYPES.vacant;
            const seatNum = idx + 1;

            return (
              <React.Fragment key={getSeatId(seat)}>
                <div
                  className="seat-card reservation-seat-card"
                  style={{ backgroundColor: config.bgColor }}
                  onClick={() => handleSeatClick(seat)}
                >
                  <div
                    className="status-bar"
                    style={{ backgroundColor: config.color }}
                  ></div>

                  <div className="card-content">
                    <img
                      src={config.icon}
                      alt={config.label}
                      className="seat-img"
                      style={{ filter: config.filter }}
                    />
                    <div className="status-label">{config.label}</div>
                  </div>

                  <div className="seat-number">{getSeatNumber(seat)}</div>
                </div>

                {seatNum % 5 === 0 && seatNum % 10 !== 0 && (
                  <div className="aisle-v"></div>
                )}
                {seatNum === 30 && <div className="aisle-h"></div>}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {selectedSeat && (
        <div className="reservation-modal-overlay" onClick={closeModal}>
          <div
            className="reservation-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <button className="reservation-modal-close" onClick={closeModal}>
              ×
            </button>

            <div className="reservation-modal-title">
              <span className="reservation-modal-icon">ⓘ</span>
              좌석 상세 정보
            </div>

            <div className="reservation-info-box">
              <div className="reservation-info-row">
                <span className="reservation-info-label">좌석 번호</span>
                <span className="reservation-info-value">
                  No. {getSeatNumber(selectedSeat)}
                </span>
              </div>

              <div className="reservation-info-row">
                <span className="reservation-info-label">현재 상태</span>
                <span
                  className={`reservation-info-value ${
                    isReservable(getSeatStatus(selectedSeat))
                      ? "reservation-info-available"
                      : "reservation-info-unavailable"
                  }`}
                >
                  {getSeatStateText(getSeatStatus(selectedSeat))}
                </span>
              </div>

              <div className="reservation-info-row">
                <span className="reservation-info-label">위치</span>
                <span className="reservation-info-value">제4 열람실 (4F)</span>
              </div>

              <div className="reservation-info-row">
                <span className="reservation-info-label">이용 가능 시간</span>
                <span className="reservation-info-value">09:00 ~ 22:00</span>
              </div>
            </div>

            <button
              className={`reservation-confirm-button ${
                !isReservable(getSeatStatus(selectedSeat)) ||
                isLoading ||
                reservedSeatId
                  ? "reservation-confirm-button-disabled"
                  : ""
              }`}
              disabled={
                !isReservable(getSeatStatus(selectedSeat)) ||
                isLoading ||
                !!reservedSeatId
              }
              onClick={handleConfirmReservation}
            >
              {reservedSeatId
                ? "이미 좌석을 예약했습니다"
                : isLoading
                ? "예약 처리중..."
                : "예약 확정"}
            </button>

            <p className="reservation-modal-note">
              * 예약 후 10분 내 미입실 시 자동 취소됩니다.
            </p>
          </div>
        </div>
      )}
    </>
  );
};

export default ReservationPage;