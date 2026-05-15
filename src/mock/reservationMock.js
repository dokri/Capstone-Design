export const mockReserveSeat = async (seat_id) => {
  const requestBody = {
    seat_id,
  };

  console.log('백엔드로 보낸다고 가정한 요청 JSON:', requestBody);

  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        success: true,
        message: '예약이 완료되었습니다.',
        seat_id,
        status: 'using',
      });
    }, 500);
  });
};