import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./MySeatPage.css";
import chairIcon from "../assets/chair.svg";

import {
  getMyReservation,
  returnReservation,
  restoreReservation,
} from "../api/reservationApi";

const SEAT_TYPES = {
  using: {
    label: "사용중",
    color: "#00BC3F",
    bgColor: "#C3FFD7",
    icon: chairIcon,
    filter:
      "invert(42%) sepia(80%) saturate(2850%) hue-rotate(125deg) brightness(105%) contrast(120%)",
  },
  waiting: {
    label: "착석대기",
    color: "#3B82F6",
    bgColor: "#DBEAFE",
    icon: chairIcon,
    filter:
      "invert(40%) sepia(95%) saturate(1800%) hue-rotate(200deg) brightness(100%) contrast(100%)",
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

const TEMP_SECONDS = 11 * 60;

const formatRemainingTime = (seconds) => {
  const safeSeconds = Math.max(0, seconds);
  const hours = String(Math.floor(safeSeconds / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((safeSeconds % 3600) / 60)).padStart(2, "0");
  const secs = String(safeSeconds % 60).padStart(2, "0");

  return `${hours}:${minutes}:${secs}`;
};

const getDisplayStatus = (seat) => {
  if (seat?.is_booked && seat?.status === "vacant") {
    return "waiting";
  }

  return seat?.status;
};

const MySeatPage = ({
  seats,
  setSeats,
  reservedSeatId,
  setReservedSeatId,
}) => {
  const navigate = useNavigate();

  const [remainingSeconds, setRemainingSeconds] = useState(TEMP_SECONDS);
  const [alerts, setAlerts] = useState([]);
  const [mySeatId, setMySeatId] = useState(reservedSeatId);
  const [isActionLoading, setIsActionLoading] = useState(false);

  const mySeat = seats.find((seat) => seat.id === mySeatId);
  const mySeatDisplayStatus = getDisplayStatus(mySeat);

  const isTemp = mySeatDisplayStatus === "temp";
  const isAuto = mySeatDisplayStatus === "auto";
  const isWaiting = mySeatDisplayStatus === "waiting";

  useEffect(() => {
    loadMyReservation();
  }, []);

  const loadMyReservation = async () => {
    try {
      const data = await getMyReservation();

      const reservationData = Array.isArray(data) ? data[0] : data;
      const seatId = reservationData?.seat_id || reservationData?.id;

      if (seatId) {
        setMySeatId(seatId);
        setReservedSeatId(seatId);

        setSeats((prevSeats) =>
          prevSeats.map((seat) =>
            seat.id === seatId
              ? {
                  ...seat,
                  status: reservationData.status || seat.status,
                  is_booked: reservationData.is_booked ?? seat.is_booked,
                  isBooked: reservationData.is_booked ?? seat.isBooked,
                  vacant_seconds:
                    reservationData.vacant_seconds ?? seat.vacant_seconds,
                }
              : seat
          )
        );
      }
    } catch (error) {
      console.warn("내 좌석 조회 실패, 프론트 상태 유지:", error);
    }
  };

  const addAlert = (type, title, description, key) => {
    setAlerts((prev) => {
      if (prev.some((alert) => alert.key === key)) return prev;

      return [
        {
          key,
          type,
          title,
          description,
        },
        ...prev,
      ];
    });
  };

  const removeAlert = (key) => {
    setAlerts((prev) => prev.filter((alert) => alert.key !== key));
  };

  useEffect(() => {
    if (!isTemp) {
      setRemainingSeconds(TEMP_SECONDS);
      return;
    }

    const serverVacantSeconds = Number(mySeat?.vacant_seconds ?? 0);
    const baseRemaining = Math.max(0, TEMP_SECONDS - serverVacantSeconds);

    setRemainingSeconds(baseRemaining);

    const timer = setInterval(() => {
      setRemainingSeconds((prev) => {
        const next = Math.max(0, prev - 1);

        if (next <= 600 && next > 590) {
          addAlert(
            "warning",
            "남은 시간 10분 남았습니다.",
            "* 자리 비움 시간 경과 시, 자동 반납 처리 됩니다.",
            "warning-10"
          );
        }

        if (next <= 300 && next > 290) {
          addAlert(
            "warning",
            "남은 시간 5분 남았습니다.",
            "* 자리 비움 시간 경과 시, 자동 반납 처리 됩니다.",
            "warning-5"
          );
        }

        return next;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [isTemp, mySeat?.vacant_seconds]);

  useEffect(() => {
    if (isAuto) {
      addAlert(
        "danger",
        "자동 반납 처리되었습니다.",
        "* 자리 비움 시간 초과로 좌석이 자동 반납 됩니다.",
        "auto-return"
      );
    }
  }, [isAuto]);

  const handleRestoreSeat = async () => {
    if (!mySeatId || isActionLoading) return;

    try {
      setIsActionLoading(true);

      await restoreReservation(mySeatId);

      setRemainingSeconds(TEMP_SECONDS);
      setAlerts([]);
      alert("좌석 사용 상태로 복귀했습니다.");

      await loadMyReservation();
    } catch (error) {
      console.warn("복귀 API 실패:", error);
      alert("복귀 처리에 실패했습니다.");
    } finally {
      setIsActionLoading(false);
    }
  };

  const handleReturnSeat = async () => {
    if (!mySeatId || isActionLoading) return;

    try {
      setIsActionLoading(true);

      await returnReservation(mySeatId);

      setReservedSeatId(null);
      localStorage.removeItem("reservedSeatId");

      setMySeatId(null);
      setRemainingSeconds(TEMP_SECONDS);
      setAlerts([]);
      alert("좌석이 반납되었습니다.");
      navigate("/");
    } catch (error) {
      console.warn("좌석 반납 API 실패:", error);
      alert("좌석 반납에 실패했습니다.");
    } finally {
      setIsActionLoading(false);
    }
  };

  if (!mySeat) {
    return (
      <>
        <div className="content-header"></div>
        <h1 className="header-title">도서관 좌석 관리 대시보드</h1>

        <div className="my-seat-empty-card">
          <h2>예약된 좌석이 없습니다.</h2>
        </div>
      </>
    );
  }

  return (
    <>
      <div className="content-header"></div>
      <h1 className="header-title">도서관 좌석 관리 대시보드</h1>

      <div className="dashboard-card my-seat-dashboard-card">
        <h2 className="subtitle">제4 열람실 (4F)</h2>

        <div className="seat-grid">
          {seats.map((seat, idx) => {
            const displayStatus = getDisplayStatus(seat);
            const config = SEAT_TYPES[displayStatus] || SEAT_TYPES.vacant;
            const seatNum = idx + 1;

            return (
              <React.Fragment key={seat.id}>
                <div
                  className={`seat-card ${
                    seat.id === mySeatId ? "my-seat-highlight-card" : ""
                  }`}
                  style={{ backgroundColor: config.bgColor }}
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

                  <div className="seat-number">{seat.id}</div>
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

      {alerts.length > 0 && (
        <div className="my-seat-alert-panel">
          <div className="my-seat-alert-header">
            <span>알림센터</span>
            <button className="my-seat-alert-close">×</button>
          </div>

          <div className="my-seat-alert-list">
            {alerts.map((alert) => (
              <div
                key={alert.key}
                className={`my-seat-alert-card ${
                  alert.type === "danger"
                    ? "my-seat-alert-card-danger"
                    : "my-seat-alert-card-warning"
                }`}
              >
                <div className="my-seat-alert-main">
                  <div
                    className={`my-seat-alert-icon ${
                      alert.type === "danger"
                        ? "my-seat-alert-icon-danger"
                        : "my-seat-alert-icon-warning"
                    }`}
                  >
                    {alert.type === "danger" ? "✓" : "⚠"}
                  </div>

                  <div className="my-seat-alert-texts">
                    <div
                      className={`my-seat-alert-title ${
                        alert.type === "danger"
                          ? "my-seat-alert-title-danger"
                          : "my-seat-alert-title-warning"
                      }`}
                    >
                      {alert.title}
                    </div>
                    <div className="my-seat-alert-desc">
                      {alert.description}
                    </div>
                  </div>
                </div>

                <button
                  className="my-seat-alert-item-close"
                  onClick={() => removeAlert(alert.key)}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="my-seat-overlay">
        <div className="my-seat-modal">
          <button className="my-seat-close" onClick={() => navigate("/")}>
            ×
          </button>

          <div className="my-seat-title">
            <span className="my-seat-title-icon">ⓘ</span>
            좌석 상세 정보
          </div>

          <div className="my-seat-info-box">
            <div className="my-seat-row">
              <span className="my-seat-label">좌석 번호</span>
              <span className="my-seat-value">No.{mySeat.id}</span>
            </div>

            <div className="my-seat-row">
              <span className="my-seat-label">현재 상태</span>
              <span
                className={`my-seat-value ${
                  isTemp
                    ? "my-seat-temp"
                    : isAuto
                    ? "my-seat-auto"
                    : isWaiting
                    ? "my-seat-waiting"
                    : "my-seat-using"
                }`}
              >
                {isTemp
                  ? "자리 비움"
                  : isAuto
                  ? "자동 반납"
                  : isWaiting
                  ? "착석대기"
                  : "사용중"}
              </span>
            </div>

            <div className="my-seat-row">
              <span className="my-seat-label">위치</span>
              <span className="my-seat-value">제4 열람실 (4F)</span>
            </div>

            {isTemp && (
              <div className="my-seat-row">
                <span className="my-seat-label">남은 복귀 시간</span>
                <span className="my-seat-value my-seat-timer">
                  {formatRemainingTime(remainingSeconds)}
                </span>
              </div>
            )}
          </div>

          {!isAuto && (
            <>
              {isTemp && (
                <button
                  className="my-seat-return-button"
                  onClick={handleRestoreSeat}
                  disabled={isActionLoading}
                >
                  복귀
                </button>
              )}

              <button
                className={`my-seat-return-button ${
                  isTemp ? "my-seat-return-button-danger" : ""
                }`}
                onClick={handleReturnSeat}
                disabled={isActionLoading}
              >
                ✓ 좌석 반납
              </button>
            </>
          )}

          {isAuto && (
            <button
              className="my-seat-return-button my-seat-return-button-disabled"
              disabled
            >
              자동 반납 완료
            </button>
          )}

          {isTemp && (
            <p className="my-seat-note">
              * 자리 비움 시간이 초과하면 자동 반납 됩니다.
            </p>
          )}

          {isAuto && (
            <p className="my-seat-note">
              * 자리 비움 시간이 초과되어 자동 반납 처리되었습니다.
            </p>
          )}
        </div>
      </div>
    </>
  );
};

export default MySeatPage;