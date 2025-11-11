// pages/Home/Home.jsx
import React, { useState } from 'react';

// 로직 분리
import { useTopStocks } from '../hooks/useTopStocks';

// 레이아웃 컴포넌트들
import StockChart from '../Components/layout/StockChart/StockChart.jsx'
import Rank from '../Components/layout/Rank/Rank.jsx';
import Sidebar from '../Components/layout/Sidebar/Sidebar.jsx';
import Topbar from '../Components/layout/Topbar/Topbar.jsx';
import AiResult from '../Components/layout/Airesult/airesult.jsx';

// 페이지 전용 스타일
import styles from '../styles/main.module.css';

const Home = () => {
  //주식 데이터 로직 
  const { topStocks, loading, error } = useTopStocks();
  const [selectedStock, setSelectedStock] = useState(null);

  const handleStockSelect = (stockCode) => {
    setSelectedStock(stockCode);
  };

  const renderChartSection = () => {
    if (selectedStock) {
      return (
        <>
          {/* [추가 예정] 애니메이션 효과*/}
          <button className={`${styles.backButton} ${styles.fade_in}`} onClick={() => setSelectedStock(null)}>
            ← 랭킹으로 돌아가기
          </button>
          <StockChart stockCode={selectedStock} />
        </>
      );
    }
    if (loading) return <div>거래량 상위 종목을 불러오는 중...</div>;
    if (error) return <div>{error}</div>;
    return <Rank stocks={topStocks} onStockSelect={handleStockSelect} />;
  };

  return (
    <div className={styles.homeContainer}> {/*각 요소들 위치를 위한 그리드 설정 */}
      <div className={styles.sidebar_wrapper}>
        <Sidebar />
      </div>
      <div className={styles.topbar_wrapper}>
        <Topbar />
      </div>
      
      {/* 메인 콘텐츠 */}
      <main className={styles.main_content}>
        <section className={styles.chart_section}>
          {renderChartSection()}
        </section>
        
        {/* 오른쪽 내용 : AI 분석결과 */}
        <div className={styles.right_panel_wrapper}>
          <section className={styles.ai_result_section}>
            <AiResult />
          </section>
        </div>
      </main>
    </div>
  );
};

export default Home;
