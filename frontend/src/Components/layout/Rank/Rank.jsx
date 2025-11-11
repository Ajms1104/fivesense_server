import React, { useState, useEffect, useRef } from 'react';
import styles from './rank.module.css';

const Rank = ({ stocks = [], onStockSelect }) => {
  const [currentTime, setCurrentTime] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;
  const [activeTab, setActiveTab] = useState('realtime');
  
  const [selectedStockCode, setSelectedStockCode] = useState(null);
  const tabsRef = useRef(null);
  const [indicatorStyle, setIndicatorStyle] = useState({});

  useEffect(() => {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    setCurrentTime(`${hours}:${minutes}`);
  }, []);

  // activeTab이 변경될 때마다 인디케이터 스타일을 업데이트하는 로직
  useEffect(() => {
    if (tabsRef.current) {
      const activeButton = tabsRef.current.querySelector(`.${styles.active}`);
      if (activeButton) {
        setIndicatorStyle({
          left: activeButton.offsetLeft,
          width: activeButton.offsetWidth,
        });
      }
    }
  }, [activeTab]);

  const totalPages = Math.ceil(stocks.length / itemsPerPage);
  const currentStocks = stocks.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);
  const goToNextPage = () => setCurrentPage((prev) => Math.min(prev + 1, totalPages));
  const goToPrevPage = () => setCurrentPage((prev) => Math.max(prev - 1, 1));
  
  // 주식 선택 핸들러: 외부로 선택을 알리는 동시에 내부 상태도 업데이트
  const handleStockClick = (stockCode) => {
    setSelectedStockCode(stockCode); // 내부 상태 업데이트
    onStockSelect(stockCode);      // 부모 컴포넌트로 선택된 코드 전달
  };

  // 즐겨찾기 버튼 클릭 핸들러
  const handleFavoriteClick = (e) => {
    e.stopPropagation(); // 이벤트 버블링을 막아 행 전체가 클릭되는 것을 방지
    // TODO: 즐겨찾기 추가/삭제 로직 구현
    alert('즐겨찾기 기능 구현 예정');
  };

  return (
    <div className={styles.rank_container}>
      <div className={styles.tab_container} ref={tabsRef}>
        <button
          className={`${styles.tab_button} ${activeTab === 'realtime' ? styles.active : ''}`}
          onClick={() => setActiveTab('realtime')}
        >실시간 주식 차트</button>
        <button
          className={`${styles.tab_button} ${activeTab === 'watchlist' ? styles.active : ''}`}
          onClick={() => setActiveTab('watchlist')}
        >관심 종목 주식 차트</button>
        <div className={styles.active_indicator} style={indicatorStyle} />
      </div>

      <div className={styles.content_container}>
        {activeTab === 'realtime' && (
          <>
            <div className={styles.header}>
              <h2 className={styles.title}>실시간 주식 차트</h2>
              <span className={styles.update_time}>현재 {currentTime} 기준</span>
            </div>
            
            <table className={styles.stocks_table}>
              <thead>
                <tr>
                  <th>종목</th>
                  <th>현재가</th>
                  <th>등락률</th>
                </tr>
              </thead>
              <tbody>
                {currentStocks.length === 0 ? (
                  <tr><td colSpan="3" className={styles.no_data}>데이터가 없습니다</td></tr>
                ) : (
                  currentStocks.map((stock, index) => {
                    const actualRank = (currentPage - 1) * itemsPerPage + index + 1;
                    return (
                      <tr key={stock.code} onClick={() => handleStockClick(stock.code)} className={`${styles.stock_row} ${selectedStockCode === stock.code ? styles.selected : ''}`}>
                        <td>
                          <div className={styles.name_cell}>
                            <span className={styles.rank_number}>{actualRank}</span>
                            <button className={styles.favorite_btn} onClick={handleFavoriteClick}>♥</button>
                            <span className={styles.stock_name}>{stock.name}</span>
                          </div>
                        </td>
                        <td className={styles.price_cell}>{Math.abs(stock.price)?.toLocaleString()}원</td>
                        <td className={`${styles.change_cell} ${stock.change >= 0 ? styles.up : styles.down}`}>
                          {stock.change >= 0 ? "▲" : "▼"} {Math.abs(stock.change)}%
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>

            <div className={styles.pagination}>
              <button onClick={goToPrevPage} disabled={currentPage === 1}>이전</button>
              <span>{currentPage} / {totalPages || 1}</span>
              <button onClick={goToNextPage} disabled={currentPage === totalPages || totalPages === 0}>다음</button>
            </div>
          </>
        )}
        
        {activeTab === 'watchlist' && (
          <div className={styles.no_data}>관심 종목 기능은 준비 중입니다.</div>
        )}
      </div>
    </div>
  );
};

export default Rank;
