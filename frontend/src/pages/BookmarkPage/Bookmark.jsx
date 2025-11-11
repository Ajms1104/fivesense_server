// pages/BookmarkPage/Bookmark.jsx 
import React, { useState } from 'react';
import style from './bookmark.module.css';

import Sidebar from '../../Components/layout/Sidebar/Sidebar.jsx';
import Topbar from '../../Components/layout/Topbar/Topbar.jsx';

// 탭 메뉴 컴포넌트
const TabMenu = ({ activeTab, setActiveTab }) => (
    <aside className={style.tab_menu}>
        <button 
            className={activeTab === 'stock' ? style.active : ''}
            onClick={() => setActiveTab('stock')}
        >
            📈 주식
        </button>
        <button 
            className={activeTab === 'search' ? style.active : ''}
            onClick={() => setActiveTab('search')}
        >
            🕜 검색 기록
        </button>
    </aside>
);

// 북마크 리스트 컴포넌트
const BookmarkList = ({ items, onDelete, type }) => (
    <ul className={style.bookmark_list}>
        {items.length > 0 ? (
            items.map(item => (
                <li key={item.id} className={style.bookmark_item}>
                    <span className={style.favorite_icon}>⭐</span>
                    <span className={style.item_name}>
                        {type === 'stock' ? item.name : item.query}
                    </span>
                    <button className={style.delete_btn} onClick={() => onDelete(item.id)}>
                        ✕
                    </button>
                </li>
            ))
        ) : (
            <div className={style.empty_message}>
                {type === 'stock' ? '즐겨찾기한 주식이 없습니다.' : '검색 기록이 없습니다.'}
            </div>
        )}
    </ul>
);

// 메인 북마크 페이지 컴포넌트
const BookmarkPage = () => {
    const [activeTab, setActiveTab] = useState('stock');

    const [stockBookmarks, setStockBookmarks] = useState([
        { id: 1, name: '삼성전자' },
        { id: 2, name: 'SK하이닉스' },
        { id: 3, name: 'LG에너지솔루션' },
    ]);
    
    const [searchHistory, setSearchHistory] = useState([
        { id: 1, query: '배당주 순위' },
        { id: 2, query: 'PER이 낮은 반도체 주식' },
        { id: 3, query: '2차전지 관련주 전망' },
    ]);

    const handleDeleteStockBookmark = (stockId) => {
        setStockBookmarks(prev => prev.filter(stock => stock.id !== stockId));
    };

    const handleDeleteSearchHistory = (historyId) => {
        setSearchHistory(prev => prev.filter(item => item.id !== historyId));
    };
    
    return (
        // 
        <div className={style.page_grid_container}>
            <div className={style.sidebar_wrapper}>
                <Sidebar />
            </div>
            <div className={style.topbar_wrapper}>
                <Topbar />
            </div>
            
            <main className={style.main_content}>
                <header className={style.header}>
                    <h1>즐겨찾기</h1>
                </header>
                <div className={style.content_wrapper}>
                    <TabMenu activeTab={activeTab} setActiveTab={setActiveTab} />
                    <section className={style.content_area}>
                        {activeTab === 'stock' ? (
                            <BookmarkList 
                                items={stockBookmarks} 
                                onDelete={handleDeleteStockBookmark} 
                                type="stock" 
                            />
                        ) : (
                            <BookmarkList 
                                items={searchHistory} 
                                onDelete={handleDeleteSearchHistory} 
                                type="search" 
                            />
                        )}
                    </section>
                </div>
            </main>
        </div>
    );
};

export default BookmarkPage;
