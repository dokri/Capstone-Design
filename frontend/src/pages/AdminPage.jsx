import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  getAdminCameras,
  createAdminCamera,
  createAdminSeat,
  deleteAdminSeat,
  getAllSeats,
} from "../api/adminApi";

import "./AdminPage.css";

const YOLO_STREAM_BASE_URL = "https://guided-contractor-chubby-fascinating.trycloudflare.com";

function AdminPage() {
  const navigate = useNavigate();

  const [cameras, setCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState("");

  const [cameraId, setCameraId] = useState("");
  const [cameraName, setCameraName] = useState("");

  const [seats, setSeats] = useState([]);
  const [seatName, setSeatName] = useState("");
  const [mappedPoint, setMappedPoint] = useState(null);

  const [loading, setLoading] = useState(true);

  const selectedCamera = cameras.find(
    (camera) => camera.id === selectedCameraId
  );

  const filteredSeats = seats.filter(
    (seat) => seat.cameraId === selectedCameraId
  );

  useEffect(() => {
    loadInitialData();
  }, []);

  const getSeatX = (seat) => {
    return seat.coord_x ?? seat.x ?? seat.coordinate_x ?? "-";
  };

  const getSeatY = (seat) => {
    return seat.coord_y ?? seat.y ?? seat.coordinate_y ?? "-";
  };

  const getSeatCameraId = (group, seat) => {
    return group.camera_id ?? seat.camera_id ?? seat.cameraId ?? "";
  };

  const loadInitialData = async () => {
    try {
      setLoading(true);

      const cameraData = await getAdminCameras();
      setCameras(cameraData || []);

      if (cameraData?.length > 0) {
        setSelectedCameraId(cameraData[0].id);
      }

      await loadSeats();
    } catch (error) {
      console.error(error);
      alert("초기 데이터를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  };

  const loadSeats = async () => {
    try {
      const seatGroups = await getAllSeats();

      const convertedSeats = seatGroups.flatMap((group) =>
        (group.seats || []).map((seat) => ({
          id: seat.id,
          name: seat.name,
          cameraId: getSeatCameraId(group, seat),
          x: getSeatX(seat),
          y: getSeatY(seat),
        }))
      );

      setSeats(convertedSeats);
    } catch (error) {
      console.error(error);
    }
  };

  const handleAddCamera = async () => {
    if (!cameraId.trim() || !cameraName.trim()) {
      alert("카메라 ID와 명칭을 입력해주세요.");
      return;
    }

    try {
      const created = await createAdminCamera({
        id: cameraId.trim(),
        name: cameraName.trim(),
      });

      setCameras((prev) => [...prev, created]);
      setSelectedCameraId(created.id);

      setCameraId("");
      setCameraName("");
      setMappedPoint(null);
    } catch (error) {
      console.error(error);
      alert("카메라 등록에 실패했습니다.");
    }
  };

  const handleVideoClick = (e) => {
    if (!selectedCameraId) return;

    const rect = e.currentTarget.getBoundingClientRect();

    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;

    setMappedPoint({
      x: Number(x.toFixed(1)),
      y: Number(y.toFixed(1)),
    });
  };

  const handleAddSeat = async () => {
    if (!selectedCameraId) {
      alert("카메라를 먼저 등록해주세요.");
      return;
    }

    if (!seatName.trim()) {
      alert("좌석명을 입력해주세요.");
      return;
    }

    if (!mappedPoint) {
      alert("영상에서 좌석 위치를 클릭해주세요.");
      return;
    }

    try {
      const createdSeat = await createAdminSeat({
        name: seatName.trim(),
        cameraId: selectedCameraId,
        x: mappedPoint.x,
        y: mappedPoint.y,
      });

      setSeats((prev) => [
        ...prev,
        {
          id: createdSeat.id,
          name: createdSeat.name,
          cameraId: createdSeat.camera_id ?? selectedCameraId,
          x: createdSeat.coord_x ?? createdSeat.x ?? mappedPoint.x,
          y: createdSeat.coord_y ?? createdSeat.y ?? mappedPoint.y,
        },
      ]);

      setSeatName("");
      setMappedPoint(null);
    } catch (error) {
      console.error(error);
      alert("좌석 등록에 실패했습니다.");
    }
  };

  const handleDeleteSeat = async (seatId) => {
    try {
      await deleteAdminSeat(seatId);
      setSeats((prev) => prev.filter((seat) => seat.id !== seatId));
    } catch (error) {
      console.error(error);
      alert("좌석 삭제에 실패했습니다.");
    }
  };

  if (loading) {
    return (
      <main className="admin-loading">
        관리자 데이터를 불러오는 중입니다...
      </main>
    );
  }

  return (
    <main className="admin-page">
      <button
        className="admin-close-button"
        type="button"
        onClick={() => navigate("/")}
      >
        ×
      </button>

      <header className="admin-header">
        <p className="admin-eyebrow">Admin</p>
        <h1>관리자 페이지</h1>
        <p>카메라 등록 후 영상 화면에서 좌석 위치를 직접 매핑합니다.</p>
      </header>

      <section className="admin-layout">
        <aside className="admin-sidebar">
          <div className="admin-card">
            <h2>카메라 등록</h2>

            <div className="form-group">
              <label>카메라 ID</label>
              <input
                value={cameraId}
                onChange={(e) => setCameraId(e.target.value)}
                placeholder="예: CAM001"
              />
            </div>

            <div className="form-group">
              <label>카메라 명칭</label>
              <input
                value={cameraName}
                onChange={(e) => setCameraName(e.target.value)}
                placeholder="예: 4층 열람실 카메라"
              />
            </div>

            <button
              className="primary-button"
              type="button"
              onClick={handleAddCamera}
            >
              + 카메라 등록
            </button>
          </div>

          <div className="admin-card">
            <h2>좌석 매핑</h2>

            <div className="form-group">
              <label>카메라 선택</label>
              <select
                value={selectedCameraId}
                onChange={(e) => {
                  setSelectedCameraId(e.target.value);
                  setMappedPoint(null);
                }}
              >
                <option value="">카메라를 등록해주세요</option>

                {cameras.map((camera) => (
                  <option key={camera.id} value={camera.id}>
                    {camera.id} · {camera.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>좌석명</label>
              <input
                value={seatName}
                onChange={(e) => setSeatName(e.target.value)}
                placeholder="예: Seat-01"
              />
            </div>

            <div className="coordinate-row">
              <div>
                <span>X 좌표(%)</span>
                <strong>{mappedPoint ? mappedPoint.x : "-"}</strong>
              </div>

              <div>
                <span>Y 좌표(%)</span>
                <strong>{mappedPoint ? mappedPoint.y : "-"}</strong>
              </div>
            </div>

            <button
              className="primary-button"
              type="button"
              onClick={handleAddSeat}
            >
              좌석 등록
            </button>
          </div>
        </aside>

        <section className="admin-main">
          <div className="admin-card video-card">
            <div className="section-title-row">
              <div>
                <h2>카메라 영상</h2>
                <p>
                  {selectedCamera
                    ? `${selectedCamera.name} 화면에서 좌석 위치를 클릭하세요.`
                    : "카메라를 등록하면 영상이 표시됩니다."}
                </p>
              </div>

              <span className="camera-badge">
                {selectedCamera ? selectedCamera.id : "카메라 미등록"}
              </span>
            </div>

            <div
              className={`video-map-box ${
                !selectedCameraId ? "disabled" : ""
              }`}
              onClick={handleVideoClick}
            >
              {!selectedCameraId ? (
                <div className="empty-video-box">
                  카메라를 등록하면 영상이 표시됩니다.
                </div>
              ) : (
                <>
                  <img
                    className="admin-video"
                    src={`${YOLO_STREAM_BASE_URL}/video/${selectedCameraId}`}
                    alt={`${selectedCameraId} camera stream`}
                  />

                  {mappedPoint && (
                    <div
                      className="mapping-point"
                      style={{
                        left: `${mappedPoint.x}%`,
                        top: `${mappedPoint.y}%`,
                      }}
                    />
                  )}
                </>
              )}
            </div>
          </div>

          <div className="admin-card">
            <h2>등록 현황</h2>

            <div className="summary-grid">
              <div>
                <span>총 카메라</span>
                <strong>{cameras.length}대</strong>
              </div>

              <div>
                <span>선택 카메라</span>
                <strong>{selectedCameraId || "-"}</strong>
              </div>

              <div>
                <span>등록 좌석</span>
                <strong>{filteredSeats.length}석</strong>
              </div>
            </div>

            <div className="seat-table">
              <div className="seat-table-head">
                <span>좌석</span>
                <span>카메라</span>
                <span>X</span>
                <span>Y</span>
                <span>관리</span>
              </div>

              {filteredSeats.length === 0 ? (
                <p className="empty-text">등록된 좌석이 없습니다.</p>
              ) : (
                filteredSeats.map((seat) => (
                  <div key={seat.id} className="seat-table-row">
                    <span>{seat.name}</span>
                    <span>{seat.cameraId}</span>
                    <span>{seat.x}</span>
                    <span>{seat.y}</span>

                    <span>
                      <button
                        className="delete-seat-button"
                        type="button"
                        onClick={() => handleDeleteSeat(seat.id)}
                      >
                        삭제
                      </button>
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>
        </section>
      </section>
    </main>
  );
}

export default AdminPage;