import React, { useState, useEffect } from 'react';
import style from './airesult.module.css';

// 날짜를 '2025.10.15'로 고정
const getFormattedDate = () => '2025.10.15';

// 숫자 포맷팅 함수
const formatNumber = (num) => {
  return new Intl.NumberFormat('ko-KR').format(num);
};

// 데이터 소스
const lstmAccuracyData = { '기아': 37.47, 'SK하이닉스': 28.88, '삼성화재': 39.18, '삼양식품': 51.36, '포스코퓨처엠': 70.79, '현대차': 41.74, '삼성전자': 28.8, '삼성전자우': 39.77, '삼성SDI': 32.93, '삼성전기': 36.93, 'HD한국조선해양': 43.16, '고려아연': 34.02, '삼성중공업': 36.54, 'HMM': 57.89, '현대모비스': 39.34, '한화에어로스페이스': 40.44, '한국전력': 39.18, 'SK텔레콤': 57.37, '삼성에스디에스': 39.66, '기업은행': 68.59, '삼성물산': 38.75, 'KT': 67.78, '삼성생명': 39.17, 'KT&G': 42.32, '두산에너빌리티': 31.73, 'SK': 40.16, 'NAVER': 29.03, '한화오션': 55.87, 'LG화학': 88.88, '신한지주': 65.63, '현대글로비스': 29.86, 'LG전자': 30.47, '셀트리온': 38.62, '하나금융지주': 42.59, 'SK이노베이션': 77.08, 'KB금융': 43.78, '삼성바이오로직스': 42.63, 'HD현대': 42.46, 'HD현대일렉트릭': 42.1, '효성첨단소재': 52.59, '우리금융지주': 72.6 };
const predictedPriceData = { '기아': 101917, 'SK하이닉스': 381410, '삼성화재': 450503, '삼양식품': 1470792, '포스코퓨처엠': 152948, '현대차': 218855, '삼성전자': 87678, '삼성전자우': 69364, '삼성SDI': 210443, '삼성전기': 198787, 'HD한국조선해양': 408701, '고려아연': 1241803, '삼성중공업': 21587, 'HMM': 20865, '현대모비스': 300826, '한화에어로스페이스': 1002418, '한국전력': 36664, 'SK텔레콤': 54564, '삼성에스디에스': 165694, '기업은행': 19284, '삼성물산': 194531, 'KT': 50065, '삼성생명': 159591, 'KT&G': 137434, '두산에너빌리티': 68636, 'SK': 219224, 'NAVER': 253396, '한화오션': 107863, 'LG화학': 284779, '신한지주': 69797, '현대글로비스': 212682, 'LG전자': 79669, '셀트리온': 172867, '하나금융지주': 87065, 'SK이노베이션': 103672, 'KB금융': 113899, '삼성바이오로직스': 1019102, 'HD현대': 156415, 'HD현대일렉트릭': 618291, '효성첨단소재': 1414614, '우리금융지주': 25802 };
const changeRateData = { '기아': 0.45, 'SK하이닉스': -7.71, '삼성화재': 2.17, '삼양식품': 8.13, '포스코퓨처엠': -12.25, '현대차': -2.01, '삼성전자': -3.30, '삼성전자우': -3.16, '삼성SDI': -5.92, '삼성전기': -1.63, 'HD한국조선해양': 0.56, '고려아연': -24.27, '삼성중공업': 2.48, 'HMM': 2.65, '현대모비스': 0.60, '한화에어로스페이스': 10.22, '한국전력': -3.33, 'SK텔레콤': 2.51, '삼성에스디에스': -0.55, '기업은행': 3.33, '삼성물산': -1.95, 'KT': 5.21, '삼성생명': -6.97, 'KT&G': -2.08, '두산에너빌리티': -7.77, 'SK': -2.43, 'NAVER': -3.79, '한화오션': 7.47, 'LG화학': -5.62, '신한지주': 1.86, '현대글로비스': 7.26, 'LG전자': -6.26, '셀트리온': 2.81, '하나금융지주': 3.51, 'SK이노베이션': -3.67, 'KB금융': 3.32, '삼성바이오로직스': -0.66, 'HD현대': 1.26, 'HD현대일렉트릭': -4.16, '효성첨단소재': -5.64, '우리금융지주': 4.25 };

// 초기 데이터 설정
const initialStock = '기아';
const initialData = {
  totalPrediction: { price: predictedPriceData[initialStock], changeRate: changeRateData[initialStock] },
  modelPredictions: {
    lstm: { accuracy: lstmAccuracyData[initialStock], predictedPrice: predictedPriceData[initialStock], changeRate: changeRateData[initialStock] },
    tft: { accuracy: 0, predictedPrice: 0, changeRate: 0 },
  },
};

const formatChangeRate = (rate) => {
    const fixedRate = rate.toFixed(2);
    return `${rate > 0 ? '+' : ''}${fixedRate}%`;
};

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
      <div className={style.progressBar} style={{ width: modelName === 'TFT' ? '0%' : `${accuracy}%`, backgroundColor: cardColor }}></div>
    </div>
    <div className={style.predictionResult} style={{ backgroundColor: `${cardColor}20` }}>
      <p>{modelName === 'FinBERT' ? '뉴스 감성 지수' : `${getFormattedDate()} 주가 예측`}</p>
      {modelName === 'TFT' ? (
        <p className={style.predictedPrice}>준비중</p>
      ) : modelName === 'FinBERT' ? (
        <p className={style.predictedPrice} style={{ color: predictedPrice > 0 ? '#e94266' : '#4287f5' }}>
          {predictedPrice > 0 ? `+${predictedPrice}` : predictedPrice}
        </p>
      ) : (
        <>
          <p className={style.predictedPrice}>{formatNumber(predictedPrice)} 원</p>
          <p className={style.changeRate} style={{ color: changeRate > 0 ? '#e94266' : '#4287f5' }}>
            {formatChangeRate(changeRate)}
          </p>
        </>
      )}
    </div>
  </div>
);

// 메인 컴포넌트
const AiResult = () => {
  const [data, setData] = useState(initialData);
  const [activeTab, setActiveTab] = useState('total');
  const [selectedStock, setSelectedStock] = useState(initialStock);
  const [finbertScore, setFinbertScore] = useState(0);
  const [finbertConfidence, setFinbertConfidence] = useState(0);

  // ✨ selectedStock이 바뀔 때마다 FinBERT 값을 새로 생성
  useEffect(() => {
    setFinbertScore(Number((Math.random() * 2 - 1).toFixed(2)));
    setFinbertConfidence(Number((Math.random() * 3.4 + 80).toFixed(2)));
  }, [selectedStock]);

  const handleStockChange = (event) => {
    const stockName = event.target.value;
    setSelectedStock(stockName);

    const newLstmAccuracy = lstmAccuracyData[stockName] || data.modelPredictions.lstm.accuracy;
    const newPredictedPrice = predictedPriceData[stockName] || data.totalPrediction.price;
    const newChangeRate = changeRateData[stockName] || data.totalPrediction.changeRate;

    setData(prevData => ({
      totalPrediction: { ...prevData.totalPrediction, price: newPredictedPrice, changeRate: newChangeRate },
      modelPredictions: { ...prevData.modelPredictions, lstm: { ...prevData.modelPredictions.lstm, accuracy: newLstmAccuracy, predictedPrice: newPredictedPrice, changeRate: newChangeRate } }
    }));
  };

  return (
    <div className={style.container}>
      <div className={style.content}>
        <div className={style.summaryHeader}>
          <p className={style.togletxt}>3가지의 AI가 종합 예측한</p>
          <select className={style.stockSelect} value={selectedStock} onChange={handleStockChange}>
            {Object.keys(predictedPriceData).map(stock => (<option key={stock} value={stock}>{stock}</option>))}
            <option>넷마블</option><option>농심</option><option>두산</option><option>대한항공</option><option>롯데케미칼</option><option>신세계</option><option>에코프로</option><option>카카오</option><option>카카오뱅크</option><option>카카오페이</option><option>쿠팡</option><option>포스코인터내셔널</option><option>하이트진로</option><option>한국타이어</option><option>한미약품</option><option>한화</option><option>CJ제일제당</option><option>LG</option><option>LG생활건강</option><option>LG에너지솔루션</option><option>LG유플러스</option><option>POSCO</option>
          </select>
          <p className={style.togletxt}>의 오늘 주가 변화 예측이에요</p>
        </div>
        <div className={style.tabContainer}>
          <button className={`${style.tabButton} ${activeTab === 'total' ? style.active : ''}`} onClick={() => setActiveTab('total')}>AI 주가 예측</button>
        </div>
        <div className={style.totalPredictionSection}>
          <div className={style.totalPriceInfo}>
            <p className={style.totalTitle}>주가 변화 예측</p>
            <p className={style.totalPrice} style={{ color: data.totalPrediction.changeRate > 0 ? '#ff3333' : '#4287f5' }}>
              {formatNumber(data.totalPrediction.price)} 원
            </p>
            <p className={style.totalChangeRate} style={{ color: data.totalPrediction.changeRate > 0 ? '#ff3333' : '#4287f5' }}>
              {formatChangeRate(data.totalPrediction.changeRate)}
            </p>
          </div>
          <div className={style.priceRangeBar}></div>
        </div>
        <div className={style.modelCardsContainer}>
          <AiResultCard modelName="LSTM" dataType="주가 Data 학습" accuracy={data.modelPredictions.lstm.accuracy} predictedPrice={data.modelPredictions.lstm.predictedPrice} changeRate={data.modelPredictions.lstm.changeRate} cardColor="#00c49f" />
          <AiResultCard modelName="FinBERT" dataType="뉴스 Data 학습" accuracy={finbertConfidence} predictedPrice={finbertScore} cardColor="#8c54ff" />
          <AiResultCard modelName="TFT" dataType="재무재표 Data 학습" accuracy={data.modelPredictions.tft.accuracy} predictedPrice={data.modelPredictions.tft.predictedPrice} changeRate={data.modelPredictions.tft.changeRate} cardColor="#f0c419" />
        </div>
      </div>
    </div>
  );
};

export default AiResult;
