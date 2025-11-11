import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';

import style from './sidebar.module.css';

//이미지 모음
import teamlogo from '../../../assets/teamlogo.png';
import side_btn from '../../../assets/Vector_3.svg';
import aireslut from '../../../assets/solar_chart-linear.svg';
import star from '../../../assets/star.svg';
import down_btn from '../../../assets/down_btn.svg';
import gpt from '../../../assets/Group.svg';

function Sidebar() {
  const navigate = useNavigate();

  //사이드바 접힘/펼침 상태 추가예정
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isTodayOpen, setIsTodayOpen] = useState(false);
  const [is7DaysOpen, setIs7DaysOpen] = useState(false);
  const [news, setNews] = useState([]);
  const [newsError, setNewsError] = useState(null);

  const toggleToday = () => setIsTodayOpen(prev => !prev);
  const toggle7Days = () => setIs7DaysOpen(prev => !prev);
  // 사이드바 접힘 상태를 변경하는 함수
  const toggleSidebar = () => setIsCollapsed(prev => !prev);

  useEffect(() => {
    const fetchNews = async () => {
      try {
        const response = await fetch('/api/stock/news');
        if (!response.ok) {
          throw new Error('뉴스를 가져오는데 실패했습니다');
        }
        const data = await response.json();
        // 데이터가 배열이 아니거나 비어있을 경우 뜨는 코드
        setNews(Array.isArray(data) ? data : []);
      } catch (error) {
        console.error('뉴스 로딩 에러:', error);
        setNewsError(error.message);
        setNews([]);
      }
    };
    fetchNews();
  }, []);


  return (
    <aside className={`${style.sidebar} ${isCollapsed ? style.collapsed : ''}`}>
      <div className={style.sidebar_top}>
        <Link to="/" className={style.logo_link}>
          <img src={teamlogo} alt="Five Sense Logo" className={style.logo_img} />
          <h1 className={style.logo_text}>FIVE_SENSE</h1>
        </Link>
        <button type="button" className={style.toggle_button} onClick={toggleSidebar} aria-label="사이드바 토글">
          <img src={side_btn} alt="Toggle Sidebar" />
        </button>
      </div>

      <div className={style.sidebar_main}>
        <div className={style.menu_section}>
          <h2 className={style.section_title}>메뉴</h2>
          <nav className={style.nav_menu}>
            <Link to="/" className={style.nav_item}>
              <img src={aireslut} alt=""/>
              <span>AI 예측결과</span>
            </Link>
            <Link to ="/aichat" className={style.nav_item}>
              <img src={gpt} alt="" />
              <span>Chat-GPT</span>
            </Link>
            <Link to="/bookmark" className={style.nav_item}>
              <img src={star} alt="" />
              <span>즐겨찾기</span>
            </Link>
          </nav>
        </div>

        <div className={style.menu_section}>
          <h2 className={style.section_title}>검색 기록</h2>
          <div className={style.history_menu}>
            <button type="button" className={style.dropdown_toggle} onClick={toggleToday}>
              <span>오늘</span>
              <img src={down_btn} alt="펼치기" className={`${style.down_icon} ${isTodayOpen ? style.rotated : ''}`} />
            </button>
            {/* 검색기록 애니메이션 */}
            {isTodayOpen && (
              <div className={`${style.dropdown_menu} ${style.fade_in}`}>
                <a href="#" className={style.dropdown_item}>검색 기록이 저장될 예정입니다</a> {/* 추가 될 예정 */}
              </div>
            )}
            <button type="button" className={style.dropdown_toggle} onClick={toggle7Days}>
              <span>7일 전</span>
              <img src={down_btn} alt="펼치기" className={`${style.down_icon} ${is7DaysOpen ? style.rotated : ''}`} />
            </button>
            {is7DaysOpen && (
              <div className={`${style.dropdown_menu} ${style.fade_in}`}>
                <a href="#" className={style.dropdown_item}>검색 기록이 저장될 예정입니다</a>{/* 추가 될 예정 */}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className={style.sidebar_bottom}>
        <h2 className={style.section_title}>최신 뉴스</h2>
        <div className={style.news_list}>
          {newsError && <div className={style.message_text}>{newsError}</div>}
          {!newsError && news.length === 0 && <div className={style.message_text}>뉴스를 불러오는 중...</div>}
          {news.map((item, index) => (
            <a key={index} href={item.link} target="_blank" rel="noopener noreferrer" className={style.news_item}>
              {item.title}
            </a>
          ))}
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;
