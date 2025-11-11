// src/Components/layout/Airesult/AiResult.jsx
import React, { useState, useEffect, useRef } from 'react';
import style from './airesult.module.css';
import { getFormattedDate, stockNameData, lstmAccuracyData, predictedPriceData, changeRateData } from './mockData';

// 유틸리티 함수
const formatNumber = (num) => new Intl.NumberFormat('ko-KR').format(num);
const formatChangeRate = (rate) => `${rate > 0 ? '+' : ''}${rate.toFixed(2)}%`;

// 헤더 컴포넌트
const Header = ({ selectedStock, handleStockChange }) => (
  <header className={style.summaryHeader}>
    <p>3가지의 AI가 종합 예측한</p>
    <select className={style.stockSelect} value={selectedStock} onChange={handleStockChange}>
      {Object.keys(stockNameData).map(stock => (
        <option key={stock} value={stock}>{stock}</option>
      ))}
    </select>
    <p>의 오늘 주가 변화 예측이에요</p>
  </header>
);

// 종합 예측 섹션 컴포넌트
const TotalPrediction = ({ price, changeRate }) => (
  <section className={style.totalPredictionSection}>
    <div className={style.totalPriceInfo}>
      <h3 className={style.totalTitle}>주가 변화 예측</h3>
      <p className={`${style.totalPrice} ${changeRate > 0 ? style.up : style.down}`}>
        {formatNumber(price)} 원
      </p>
      <p className={`${style.totalChangeRate} ${changeRate > 0 ? style.up : style.down}`}>
        {formatChangeRate(changeRate)}
      </p>
    </div>
    <div className={style.priceRangeBar}></div>
  </section>
);

// 개별 AI 모델 카드 컴포넌트
const AiResultCard = ({ modelName, dataType, accuracy, predictedPrice, changeRate, cardColor }) => (
    <div className={style.modelCard}>
      <div className={style.modelHeader}>
        <h3 className={style.modelTitle}>{modelName}</h3>
        <span className={style.dataType}>{dataType}</span>
      </div>
      <div className={style.accuracySection}>
        <span>{modelName === 'FinBERT' ? '확신도' : '정확도'}</span>
        <span>{modelName === 'TFT' ? '-' : `${accuracy.toFixed(2)}%`}</span>
      </div>
      <div className={style.progressBarContainer}>
        <div 
            className={style.progressBar} 
            style={{ 
                width: modelName === 'TFT' ? '0%' : `${accuracy}%`, 
                backgroundColor: cardColor 
            }}
        ></div>
      </div>
      <div className={style.predictionResult} style={{ backgroundColor: `${cardColor}20` }}>
        <p>{modelName === 'FinBERT' ? '뉴스 감성 지수' : `${getFormattedDate()} 주가 예측`}</p>
        {modelName === 'TFT' ? (
          <p className={style.predictedPrice}>준비중</p>
        ) : modelName === 'FinBERT' ? (
          <p className={`${style.predictedPrice} ${predictedPrice > 0 ? style.up : style.down}`}>
            {predictedPrice > 0 ? `+${predictedPrice}` : predictedPrice}
          </p>
        ) : (
          <>
            <p className={style.predictedPrice}>{formatNumber(predictedPrice)} 원</p>
            <p className={`${style.changeRate} ${changeRate > 0 ? style.up : style.down}`}>
              {formatChangeRate(changeRate)}
            </p>
          </>
        )}
      </div>
    </div>
  );

// 메인 컴포넌트 
const AiResult = () => {
  const [selectedStock, setSelectedStock] = useState('기아');
  const [predictionData, setPredictionData] = useState({
    price: predictedPriceData['기아'],
    changeRate: changeRateData['기아'],
    lstmAccuracy: lstmAccuracyData['기아'],
  });
  const [finbertData, setFinbertData] = useState({ score: 0, confidence: 0 });
  
  const activeTabRef = useRef(null);
  const [indicatorStyle, setIndicatorStyle] = useState({});

  useEffect(() => {
    if (activeTabRef.current) {
      const { offsetLeft, clientWidth } = activeTabRef.current;
      setIndicatorStyle({ left: offsetLeft, width: clientWidth });
    }
  }, []);

  useEffect(() => {
    setFinbertData({
      score: Number((Math.random() * 2 - 1).toFixed(2)),
      confidence: Number((Math.random() * 3.4 + 80).toFixed(2)),
    });
  }, [selectedStock]);
  
  const handleStockChange = (event) => {
    const stockName = event.target.value;
    setSelectedStock(stockName);

    setPredictionData({
        price: predictedPriceData[stockName] || 0,
        changeRate: changeRateData[stockName] || 0,
        lstmAccuracy: lstmAccuracyData[stockName] || 0,
    });
  };

  return (
    <div className={style.container}>
      <div className={style.tab_container}>
        <button ref={activeTabRef} className={`${style.tab_button} ${style.active}`}>
            AI 주가 예측
        </button>
        {/* 나중에 다른 탭 추가용 | 예: <button className={style.tab_button}>다른 예측</button> */}
        <div className={style.active_indicator} style={indicatorStyle} />
      </div>

      {/* 각 AI 분석결과 */}
      <div className={style.content_container}>
        <Header selectedStock={selectedStock} handleStockChange={handleStockChange} />
        <TotalPrediction price={predictionData.price} changeRate={predictionData.changeRate} />
        <div className={style.modelCardsContainer}>
          <AiResultCard modelName="LSTM" dataType="주가 Data 학습" accuracy={predictionData.lstmAccuracy} predictedPrice={predictionData.price} changeRate={predictionData.changeRate} cardColor="#00c49f" />
          <AiResultCard modelName="FinBERT" dataType="뉴스 Data 학습" accuracy={finbertData.confidence} predictedPrice={finbertData.score} cardColor="#8c54ff" />
          <AiResultCard modelName="LSTM 앙상블 모델" dataType="LSTM , BiLSTM, GRU, Transformer" accuracy={0} predictedPrice={0} changeRate={0} cardColor="#f0c419" />
        </div>
      </div>
    </div>
  );
};

export default AiResult;
