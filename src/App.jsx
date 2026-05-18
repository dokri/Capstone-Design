import React, { useEffect, useState } from 'react';
import { Routes, Route, NavLink, useLocation } from 'react-router-dom';
import './App.css';
import myongjiLogo from './assets/myongji.png';
import DashboardPage from './pages/DashboardPage.jsx';
import ReservationPage from './pages/ReservationPage.jsx';
import AdminPage from './pages/AdminPage.jsx';
import MySeatPage from './pages/MySeatPage.jsx';
import { getAllSeats } from './api/seatApi';

const createInitialSeats = () => {
  const initialSeats = Array.from({ length: 70 }, (_, i) => ({
    id: i + 1,
    name: `Seat-${String(i + 1).padStart(2, '0')}`,
    status: 'vacant',
    is_booked: false,
    camera_id: null,
  }));

  const usingIdx = [
    1, 3, 5, 7, 10, 12, 14, 15, 18, 21, 25, 28, 31, 33, 34,
    36, 37, 39, 40, 42, 48, 51, 53, 55, 57, 60, 63, 66, 68, 70,
  ];

  const autoIdx = [
    2, 9, 16, 19, 22, 26, 32, 44, 46, 49, 52, 61, 65, 69,
  ];

  const tempIdx = [
    6, 13, 20, 24, 27, 29, 38, 41, 43, 45, 47, 50, 58, 64, 67,
  ];

  const vacantIdx = [4, 8, 11, 17, 23, 30, 35, 54, 56, 59, 62];

  usingIdx.forEach((num) => {
    initialSeats[num - 1].status = 'using';
  });

  autoIdx.forEach((num) => {
    initialSeats[num - 1].status = 'auto';
  });

  tempIdx.forEach((num) => {
    initialSeats[num - 1].status = 'temp';
  });

  vacantIdx.forEach((num) => {
    initialSeats[num - 1].status = 'vacant';
  });

  return initialSeats;
};

const convertSeatGroupsToSeats = (seatGroups) => {
  if (!Array.isArray(seatGroups)) return [];

  return seatGroups.flatMap((group) =>
    (group.seats || []).map((seat) => ({
      ...seat,
      id: seat.id,
      name: seat.name,
      status: seat.status || 'vacant',
      is_booked: seat.is_booked ?? false,
      camera_id: group.camera_id,
    }))
  );
};

const App = () => {
  const [seats, setSeats] = useState(() => {
    const savedSeats = localStorage.getItem('seats');

    if (savedSeats) {
      return JSON.parse(savedSeats);
    }

    return createInitialSeats();
  });

  const [reservedSeatId, setReservedSeatId] = useState(() => {
    const savedSeatId = localStorage.getItem('reservedSeatId');
    return savedSeatId ? Number(savedSeatId) : null;
  });

  const [isSeatApiConnected, setIsSeatApiConnected] = useState(false);

  const location = useLocation();

  const isAdminPage = location.pathname === '/admin';

  const fetchSeats = async () => {
    try {
      const data = await getAllSeats();
      const apiSeats = convertSeatGroupsToSeats(data);

      if (apiSeats.length >= 70) {
        setSeats(apiSeats);
        setIsSeatApiConnected(true);
      } else {
        setIsSeatApiConnected(false);
      }
    } catch (error) {
      console.error('좌석 상태 조회 실패:', error);
      setIsSeatApiConnected(false);
    }
  };

  useEffect(() => {
    fetchSeats();

    const intervalId = setInterval(() => {
      fetchSeats();
    }, 3000);

    return () => clearInterval(intervalId);
  }, []);

  // 예약 좌석 저장
  useEffect(() => {
    if (reservedSeatId) {
      localStorage.setItem('reservedSeatId', String(reservedSeatId));
    } else {
      localStorage.removeItem('reservedSeatId');
    }
  }, [reservedSeatId]);

  // 전체 좌석 상태 저장
  useEffect(() => {
    localStorage.setItem('seats', JSON.stringify(seats));
  }, [seats]);

  return (
    <div className="container">
      {!isAdminPage && (
        <div className="sidebar">
          <NavLink to="/" className="logo-section">
            <img
              src={myongjiLogo}
              alt="Myongji University Logo"
              className="logo-img"
            />
          </NavLink>

          {reservedSeatId ? (
            <NavLink
              to="/my-seat"
              className={({ isActive }) =>
                isActive ? 'nav-button active' : 'nav-button'
              }
            >
              내 좌석 현황
            </NavLink>
          ) : (
            <NavLink
              to="/reservation"
              className={({ isActive }) =>
                isActive ? 'nav-button active' : 'nav-button'
              }
            >
              좌석 예약
            </NavLink>
          )}

          <div className="sidebar-footer">
            <NavLink
              to="/admin"
              className={({ isActive }) =>
                isActive ? 'admin-button active-admin' : 'admin-button'
              }
            >
              관리자 페이지
            </NavLink>
          </div>
        </div>
      )}

      <div className={`main-content ${isAdminPage ? 'admin-main-content' : ''}`}>
        <Routes>
          <Route path="/" element={<DashboardPage seats={seats} />} />

          <Route
            path="/reservation"
            element={
              <ReservationPage
                seats={seats}
                setSeats={setSeats}
                reservedSeatId={reservedSeatId}
                setReservedSeatId={setReservedSeatId}
              />
            }
          />

          <Route
            path="/my-seat"
            element={
              <MySeatPage
                seats={seats}
                setSeats={setSeats}
                reservedSeatId={reservedSeatId}
                setReservedSeatId={setReservedSeatId}
              />
            }
          />

          <Route path="/admin" element={<AdminPage />} />
        </Routes>
      </div>
    </div>
  );
};

export default App;