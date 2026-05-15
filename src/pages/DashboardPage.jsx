import React from 'react';
import chairIcon from '../assets/chair.svg';

const SEAT_TYPES = {
  using: {
    label: '사용중',
    color: '#00BC3F',
    bgColor: '#C3FFD7',
    icon: chairIcon,
    filter:
      'invert(42%) sepia(80%) saturate(2850%) hue-rotate(125deg) brightness(105%) contrast(120%)',
  },
  vacant: {
    label: '미사용',
    color: '#454545',
    bgColor: '#EFEFEF',
    icon: chairIcon,
    filter: 'none',
  },
  auto: {
    label: '자동반납',
    color: '#D25354',
    bgColor: '#FDE9E8',
    icon: chairIcon,
    filter:
      'invert(38%) sepia(74%) saturate(2476%) hue-rotate(340deg) brightness(95%) contrast(105%)',
  },
  temp: {
    label: '일시비움',
    color: '#FFC200',
    bgColor: '#FEF2CC',
    icon: chairIcon,
    filter:
      'invert(85%) sepia(85%) saturate(1500%) hue-rotate(10deg) brightness(105%) contrast(105%)',
  },
};

const DashboardPage = ({ seats }) => {
  return (
    <>
      <div className="content-header"></div>
      <h1 className="header-title">도서관 좌석 관리 대시보드</h1>

      <div className="dashboard-card">
        <h2 className="subtitle">제4 열람실 (4F)</h2>

        <div className="seat-grid">
          {seats.map((seat, idx) => {
            const config = SEAT_TYPES[seat.status] || SEAT_TYPES.vacant;
            const seatNum = idx + 1;

            return (
              <React.Fragment key={seat.id}>
                <div
                  className="seat-card"
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
    </>
  );
};

export default DashboardPage;