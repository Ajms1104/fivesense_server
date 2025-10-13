import React, { useState, useEffect } from 'react';
import style from './airesult.module.css';

// 오늘의 날짜를 YYYY.MM.DD 형식으로 가져오는 함수
const getFormattedDate = () => {
  const today = new Date();
  const year = today.getFullYear();
  const month = String(today.getMonth() + 1).padStart(2, '0');
  const day = String(today.getDate()).padStart(2, '0');
  return `${year}.${month}.${day}`;
};

// 숫자 포맷팅 함수 (예: 60600 -> 60,600)
const formatNumber = (num) => {
  return new Intl.NumberFormat('ko-KR').format(num);
};

// 백엔드에서 받아올 데이터에 대한 더미(dummy) 데이터
const dummyData = {
  totalPrediction: {
    price: 95000,
    changeRate: 1.17,
  },
  modelPredictions: {
    lstm: {
      accuracy: 63,
      predictedPrice: 95800,
      changeRate: 0.84,
    },
    finbert: {
      accuracy: 67,
      predictedPrice: 96300,
      changeRate: 1.37,
    },
    tft: {
      accuracy: 70,
      predictedPrice: 60000,
      changeRate: 1.35,
    },
  },
};

const AiResultCard = ({ modelName, dataType, accuracy, predictedPrice, changeRate, cardColor }) => (
  // AI 개별 결과
  <div className={style.modelCard}>
    <div className={style.modelHeader}>
      <h3 className={style.modelTitle}>{modelName}</h3>
      <span className={style.dataType}>{dataType}</span>
    </div>

    <div className={style.accuracySection}>
      <span>정확도</span>
      <span>{accuracy}%</span>
    </div>

    <div className={style.progressBarContainer}>
      <div className={style.progressBar}></div>
    </div>

    <div className={style.predictionResult} style={{ backgroundColor: `${cardColor}20` /* 20은 투명도 */ }}>
      <p>{getFormattedDate()} '주식이름' 주가 예측</p>
      <p className={style.predictedPrice}>{formatNumber(predictedPrice)} 원</p>
      <p className={style.changeRate}>+{changeRate.toFixed(2)}%</p>
    </div>
  </div>
);

// 메인 AI 예측 결과 컴포넌트
const AiResult = () => {
  const [data, setData] = useState(dummyData); // 실제로는 API 호출로 데이터 설정
  const [activeTab, setActiveTab] = useState('total');

  // useEffect(() => {
  //   // 백엔드에서 데이터를 받아오는 로직
  //   const fetchData = async () => {
  //     // const response = await fetch('your-api-endpoint');
  //     // const result = await response.json();
  //     // setData(result);
  //   };
  //   fetchData();
  // }, []);

  return (
    <div className={style.container}>
        {/*메인 주식 BOX*/}
      <div className={style.content}>
        <div className={style.summaryHeader}>
          <p className={style.togletxt}>3가지의 AI가 종합 예측한</p>
          <select className={style.stockSelect}>
            <option>기아</option> 
            <option>넷마블</option> <option>농심</option> 
            <option>두산</option> <option>대한항공</option> 
            <option>롯데케미칼</option> 
            <option>삼성SDI</option> <option>삼성물산</option> <option>삼성바이오로직스</option> <option>삼성생명</option> <option>삼성전자</option> 
            <option>셀트리온</option> <option>신세계</option> <option>신한지주</option> 
            <option>에코프로</option> <option>우리금융지주</option> 
            <option>카카오</option> <option>카카오뱅크</option> <option>카카오페이</option> 
            <option>쿠팡</option> <option>포스코퓨처엠</option> <option>포스코인터내셔널</option> 
            <option>하나금융지주</option> <option>하이트진로</option> <option>한국전력</option> <option>한국조선해양</option> <option>한국타이어</option> <option>한미약품</option> 
            <option>한화</option> <option>현대글로비스</option> <option>현대모비스</option> <option>현대자동차</option> <option>현대제철</option> 
            <option>CJ제일제당</option> <option>HMM</option> <option>KB금융</option> <option>KB국민은행</option> 
            <option>KT</option> 
            <option>LG</option> <option>LG생활건강</option> <option>LG에너지솔루션</option> <option>LG유플러스</option> <option>LG전자</option> <option>LG화학</option> 
            <option>NAVER</option> <option>POSCO</option> 
            <option>SK</option> <option>SK바이오사이언스</option> <option>SK이노베이션</option> <option>SK하이닉스</option>
          </select>
          <p className={style.togletxt}>의 오늘 주가 변화 예측이에요</p>
        </div>

        {/* 설정 button*/}
        <div className={style.tabContainer}>
        <button
          className={`${style.tabButton} ${activeTab === 'total' ? style.active : ''}`}
          onClick={() => setActiveTab('total')}
        > AI 주가 예측
        </button>
      </div>

        {/* AI total 평가예측 */}
        <div className={style.totalPredictionSection}>
          <div className={style.totalPriceInfo}>
            <p className={style.totalTitle}>주가 변화 예측</p>
            <p className={style.totalPrice}>{formatNumber(data.totalPrediction.price)} 원</p>
            <p className={style.totalChangeRate}>+{data.totalPrediction.changeRate.toFixed(2)}%</p>
          </div>
          <div className={style.priceRangeBar}>
            {/* 이 부분은 각 모델의 예측 가격 위치를 계산하여 동적으로 렌더링해야 합니다. */}
          </div>
        </div>

        <div className={style.modelCardsContainer}>
          <AiResultCard
            modelName="LSTM"
            dataType="주가 Data 학습"
            accuracy={data.modelPredictions.lstm.accuracy}
            predictedPrice={data.modelPredictions.lstm.predictedPrice}
            changeRate={data.modelPredictions.lstm.changeRate}
            cardColor="#00c49f"
          />
          <AiResultCard
            modelName="FinBERT"
            dataType="뉴스 Data 학습"
            accuracy={data.modelPredictions.finbert.accuracy}
            predictedPrice={data.modelPredictions.finbert.predictedPrice}
            changeRate={data.modelPredictions.finbert.changeRate}
            cardColor="#8c54ff"
          />
            
          <AiResultCard
            modelName="TFT"
            dataType="재무재표 Data 학습"
            accuracy={data.modelPredictions.tft.accuracy}
            predictedPrice={data.modelPredictions.tft.predictedPrice}
            changeRate={data.modelPredictions.tft.changeRate}
            cardColor="#f0c419"
          />
          
        </div>
      </div>
    </div>
  );
};

export default AiResult;
