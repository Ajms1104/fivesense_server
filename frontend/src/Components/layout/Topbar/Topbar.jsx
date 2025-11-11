// src/Components/layout/Topbar/Topbar.jsx 

import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom'; 
import { useAuth } from '../../../contexts/AuthContext'; 
import style from './topbar.module.css';
import userProfileIcon from '../../../assets/user.svg';

// 모바일 전용 내비게이션 컴포넌트
const MobileNav = () => (
    <nav className={style.mobile_nav}>
        <NavLink to="/" end className={({ isActive }) => isActive ? `${style.mobile_nav_link} ${style.active}` : style.mobile_nav_link}>
            실시간 랭킹
        </NavLink>
        <NavLink to="/aichat" className={({ isActive }) => isActive ? `${style.mobile_nav_link} ${style.active}` : style.mobile_nav_link}>
            AI 채팅
        </NavLink>
    </nav>
);


//  Topbar 컴포넌트 
const Topbar = () => {
    const [showUserPopup, setShowUserPopup] = useState(false);
    const { logout } = useAuth();
    const navigate = useNavigate(); 

    const toggleUserPopup = () => setShowUserPopup(prev => !prev);

    const handleLogout = () => {
        logout();
        navigate('/login'); // 로그아웃 후 로그인 페이지로 이동
    };

    return (
        <header className={style.topbar_container}>
            <div className={style.topbar_left}></div>
            <MobileNav />
            <div className={style.topbar_right}>
                <button type="button" onClick={toggleUserPopup} className={style.user_profile_button}>
                    <img src={userProfileIcon} alt="User Profile" />
                </button>
                {showUserPopup && (
                    <div className={style.user_popup}>
                        <button onClick={handleLogout}>로그아웃</button>
                    </div>
                )}
            </div>
        </header>
    );
};

export default Topbar;
