// pages/LoginPage/Login.jsx (수정)
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext.jsx';
import style from './login.module.css';
import teamlogo from '../../assets/teamlogo.png';

// [시니어 멘토링] 헤더 컴포넌트로 분리하여 구조를 명확하게 합니다.
const LoginHeader = () => (
    <header className={style.header}>
        <img src={teamlogo} alt="팀 로고" className={style.logo_image} />
        <h1 className={style.logo_text}>FIVE_SENSE</h1>
    </header>
);

// --- 👑 메인 로그인 페이지 컴포넌트 ---
const Login = () => {
    const navigate = useNavigate();
    const { login } = useAuth(); // AuthContext에서 login 함수 가져오기
    const [accountid, setAccountid] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState(''); // [시니어 멘토링] 에러 상태 추가
    const [isLoading, setIsLoading] = useState(false); // [시니어 멘토링] 로딩 상태 추가

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);

        try {
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ accountid, password })
            });
            const data = await response.json();

            if (data.success) {
                // [시니어 멘토링] 로그인 성공 시 AuthContext를 통해 상태 업데이트 및 페이지 이동
                login(data.user); // user 정보를 context에 저장
                navigate('/');
            } else {
                setError(data.message || '아이디 또는 비밀번호가 올바르지 않습니다.');
            }
        } catch (err) {
            setError('서버와 통신 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className={style.page_container}>
            <div className={style.login_box}>
                <LoginHeader />
                <form className={style.login_form} onSubmit={handleSubmit}>
                    <div className={style.form_group}>
                        <label htmlFor="accountid">아이디</label>
                        <input
                            type="text"
                            id="accountid"
                            value={accountid}
                            onChange={e => setAccountid(e.target.value)}
                            placeholder="아이디를 입력하세요"
                            required
                        />
                    </div>
                    <div className={style.form_group}>
                        <label htmlFor="password">비밀번호</label>
                        <input
                            type="password"
                            id="password"
                            value={password}
                            onChange={e => setPassword(e.target.value)}
                            placeholder="비밀번호를 입력하세요"
                            required
                        />
                    </div>
                    
                    {/* [시니어 멘토링] 에러 메시지를 UI에 직접 표시 */}
                    {error && <p className={style.error_message}>{error}</p>}

                    <div className={style.button_group}>
                        <button className={style.login_btn} type="submit" disabled={isLoading}>
                            {isLoading ? '로그인 중...' : '로그인'}
                        </button>
                        <button 
                            className={style.join_btn} 
                            type="button" 
                            onClick={() => navigate('/join')}
                            disabled={isLoading}
                        >
                            회원가입
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default Login;
