// src/Components/layout/StockChart/StockChart.jsx
import React, { useState, useEffect, useRef } from 'react';
import { createChart, CrosshairMode } from 'lightweight-charts';
import styles from './stockChart.module.css'; 

// 디바운스 헬퍼 함수
function debounce(func, timeout = 200) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => { func.apply(this, args); }, timeout);
  };
}

// 하위 컴포넌트 1: 차트 헤더
const ChartHeader = ({ stockInfo, isLoading }) => {
  if (isLoading || !stockInfo?.name) {
    return <div className={styles.header_skeleton} />;
  }
  return (
    <div className={styles.chart_header}>
      <div className={styles.stock_identity}>
        <div className={styles.stock_logo}>{stockInfo.name.charAt(0)}</div>
        <h2>{stockInfo.name}</h2>
      </div>
      <div className={styles.stock_price_info}>
        <span className={`${styles.current_price} ${styles[stockInfo.changeType]}`}>
          {stockInfo.price.toLocaleString()}원
        </span>
        <span className={`${styles.change_amount} ${styles[stockInfo.changeType]}`}>
          {stockInfo.changeAmount >= 0 ? '▲' : '▼'} {Math.abs(stockInfo.changeAmount).toLocaleString()} ({stockInfo.changeRate}%)
        </span>
      </div>
    </div>
  );
};

// 하위 컴포넌트 2: 메인 탭
const MainTabs = () => (
  <div className={styles.main_tabs}>
    <button className={`${styles.tab_button} ${styles.active}`}>차트</button>
    <button className={styles.tab_button}>종목정보</button>
    <button className={styles.tab_button}>뉴스·공시</button>
  </div>
);

// 하위 컴포넌트 3: 차트 컨트롤
const ChartControls = ({ chartType, onChartTypeChange }) => (
  <div className={styles.chart_controls}>
    <div className={styles.timeframe_selector}>
      {['1분', '일', '주', '월', '년'].map(type => (
        <button key={type} className={chartType === type ? styles.active : ''} onClick={() => onChartTypeChange(type)}>
          {type}
        </button>
      ))}
    </div>
  </div>
);

// 하위 컴포넌트 4: 로딩 스켈레톤 UI
const ChartSkeleton = () => (
  <div className={styles.skeleton_container}>
    <div className={`${styles.skeleton} ${styles.skeleton_price}`} />
    <div className={`${styles.skeleton} ${styles.skeleton_volume}`} />
  </div>
);

//메인 차트 컴포넌트
const StockChart = ({ stockCode = '005930' }) => {
  const priceChartContainerRef = useRef(null);
  const volumeChartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candlestickSeriesRef = useRef(null);
  const volumeSeriesRef = useRef(null);

  const [stockInfo, setStockInfo] = useState(null);
  const [chartType, setChartType] = useState('일');
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!priceChartContainerRef.current || !volumeChartContainerRef.current) return;

    const chartOptions = {
      layout: { background: { color: 'transparent' }, textColor: '#333333', fontFamily: 'Pretendard' },
      grid: { vertLines: { color: '#f0f0f0' }, horzLines: { color: '#f0f0f0' } },
      crosshair: { mode: CrosshairMode.Normal },
      timeScale: { borderColor: '#dddddd', borderVisible: true, timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderColor: '#dddddd', borderVisible: true, scaleMargins: { top: 0.1, bottom: 0.2 } },
      watermark: { visible: false },
    };

    const priceChart = createChart(priceChartContainerRef.current, chartOptions);
    const volumeChart = createChart(volumeChartContainerRef.current, {
      ...chartOptions,
      rightPriceScale: { ...chartOptions.rightPriceScale, scaleMargins: { top: 0.2, bottom: 0 } },
    });

    candlestickSeriesRef.current = priceChart.addCandlestickSeries({
      upColor: '#d84040', downColor: '#3b64d8', borderVisible: false, wickUpColor: '#d84040', wickDownColor: '#3b64d8'
    });
    volumeSeriesRef.current = volumeChart.addHistogramSeries({
      priceFormat: { type: 'volume' },
    });

    chartRef.current = { priceChart, volumeChart };

    const handleResize = () => {
      if (chartRef.current && priceChartContainerRef.current && volumeChartContainerRef.current) {
        priceChart.resize(priceChartContainerRef.current.clientWidth, priceChartContainerRef.current.clientHeight);
        volumeChart.resize(volumeChartContainerRef.current.clientWidth, volumeChartContainerRef.current.clientHeight);
      }
    };
    const debouncedResize = debounce(handleResize, 200);

    window.addEventListener('resize', debouncedResize);
    setTimeout(handleResize, 50);

    return () => {
      window.removeEventListener('resize', debouncedResize);
      if (chartRef.current) {
        chartRef.current.priceChart.remove();
        chartRef.current.volumeChart.remove();
      }
    };
  }, []);

  // [Phase 2: 데이터 로딩]
  useEffect(() => {
    if (!candlestickSeriesRef.current || !volumeSeriesRef.current) return;

    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        let apiId;
        let requestData = { stk_cd: stockCode, upd_stkpc_tp: "1" };
        
        switch (chartType) {
          case '1분': apiId = 'KA10080'; requestData.tic_scope = '1'; break;
          case '일': apiId = 'KA10081'; break;
          case '주': apiId = 'KA10082'; break;
          case '월': apiId = 'KA10083'; break;
          case '년': apiId = 'KA10094'; break;
          default: apiId = 'KA10081';
        }

        const response = await fetch(`/api/stock/daily-chart/${stockCode}?apiId=${apiId}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(requestData)
        });
        if (!response.ok) throw new Error('API 응답에 실패했습니다.');
        const data = await response.json();
        
        let chartData;
        switch (chartType) {
            case '월': chartData = data.stk_mth_pole_chart_qry; break;
            case '일': chartData = data.stk_dt_pole_chart_qry; break;
            case '주': chartData = data.stk_stk_pole_chart_qry; break;
            case '년': chartData = data.stk_yr_pole_chart_qry; break;
            case '1분': chartData = data.stk_min_pole_chart_qry || data.stk_stk_pole_chart_qry; break;
            default: chartData = data.stk_dt_pole_chart_qry;
        }

        if (chartData && chartData.length > 0) {
            const processedData = chartData.map(item => {
                let dateStr = (chartType === '1분') ? item.cntr_tm : (item.dt || item.trd_dt);
                if (!dateStr) return null;
                let timestamp;
                try {
                    if (chartType === '년' && dateStr.length === 4) {
                        timestamp = new Date(parseInt(dateStr), 0, 1).getTime() / 1000;
                    } else if (chartType === '1분' && dateStr.length === 14) {
                        timestamp = new Date(parseInt(dateStr.slice(0, 4)), parseInt(dateStr.slice(4, 6)) - 1, parseInt(dateStr.slice(6, 8)), parseInt(dateStr.slice(8, 10)), parseInt(dateStr.slice(10, 12))).getTime() / 1000;
                    } else if (dateStr.length === 8) {
                        timestamp = new Date(parseInt(dateStr.slice(0, 4)), parseInt(dateStr.slice(4, 6)) - 1, parseInt(dateStr.slice(6, 8))).getTime() / 1000;
                    } else { return null; }
                } catch (e) { return null; }
                if (isNaN(timestamp)) return null;

                let close = parseFloat(item.cur_prc || item.clos_prc);
                if (isNaN(close)) return null;
                
                let open = parseFloat(item.open_pric || item.open_prc) || close;
                let high = parseFloat(item.high_pric || item.high_prc) || Math.max(close, open);
                let low = parseFloat(item.low_pric || item.low_prc) || Math.min(close, open);
                let volume = parseFloat(item.trde_qty || item.trd_qty) || 0;

                return { time: timestamp, open, high, low, close, volume };
            }).filter(Boolean);

            if(processedData.length === 0) throw new Error('유효한 차트 데이터가 없습니다.');

            processedData.sort((a, b) => a.time - b.time);
            
            const candlestickData = processedData.map(({ time, open, high, low, close }) => ({ time, open, high, low, close }));
            const volumeData = processedData.map(({ time, volume, open, close }) => ({ time, value: volume, color: close >= open ? 'rgba(216, 64, 64, 0.7)' : 'rgba(59, 100, 216, 0.7)' }));
            
            candlestickSeriesRef.current.setData(candlestickData);
            volumeSeriesRef.current.setData(volumeData);
            
            chartRef.current.priceChart.timeScale().fitContent();
            
            const latestData = processedData[processedData.length - 1];
            setStockInfo({
                name: stockCode, price: latestData.close, changeAmount: latestData.close - latestData.open,
                changeRate: ((latestData.close - latestData.open) / latestData.open * 100).toFixed(2),
                changeType: latestData.close >= latestData.open ? 'up' : 'down'
            });
        } else { throw new Error('차트 데이터가 없습니다.'); }
      } catch (err) {
        setError(err.message);
        setStockInfo(null);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [stockCode, chartType]);

  return (
    <div className={`${styles.stock_chart_layout} ${styles.enter}`}>
      <ChartHeader stockInfo={stockInfo} isLoading={isLoading && !error} />
      <MainTabs />
      <ChartControls chartType={chartType} onChartTypeChange={setChartType} />
      <div className={styles.chart_area_container}>
        {isLoading && <div className={styles.loader_wrapper}><ChartSkeleton /></div>}
        {error && !isLoading && <div className={styles.error}>{error}</div>}
        
        <div className={`${styles.chart_content_wrapper} ${!isLoading && !error ? styles.visible : ''}`}>
          <div ref={priceChartContainerRef} className={styles.price_chart_container} />
          <div ref={volumeChartContainerRef} className={styles.volume_chart_container} />
        </div>
      </div>
    </div>
  );
};

export default StockChart;
