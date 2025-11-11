// pages/JoinPage/Join.jsx (수정)
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import style from './join.module.css';
import teamlogo from '../../assets/teamlogo.png';

// 헤더 컴포넌트
const JoinHeader = () => (
    <header className={style.header}>
        <img src={teamlogo} alt="팀 로고" className={style.logo_image} />
        <h1 className={style.logo_text}>FIVE_SENSE</h1>
    </header>
);

// --- 👑 메인 회원가입 페이지 컴포넌트 ---
const Join = () => {
    const navigate = useNavigate();
    const [formData, setFormData] = useState({
        accountid: '',
        username: '',
        email: '',
        password: '',
        confirmPw: ''
    });
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleChange = (e) => {
        const { id, value } = e.target;
        setFormData(prev => ({ ...prev, [id]: value }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        
        if (formData.password !== formData.confirmPw) {
            setError('비밀번호가 일치하지 않습니다.');
            return;
        }

        setIsLoading(true);
        try {
            const response = await fetch('/api/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    accountid: formData.accountid,
                    username: formData.username,
                    password: formData.password,
                    email: formData.email
                })
            });
            const data = await response.json();

            if (data.success) {
                alert('회원가입이 완료되었습니다. 로그인 페이지로 이동합니다.');
                navigate('/login');
            } else {
                setError(data.message || '회원가입에 실패했습니다. 다시 시도해주세요.');
            }
        } catch (err) {
            setError('서버와 통신 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className={style.page_container}>
            <div className={style.join_box}>
                <JoinHeader />
                <form className={style.join_form} onSubmit={handleSubmit}>
                    {/* [시니어 멘토링] 입력 필드를 배열로 만들어 map으로 렌더링하여 코드 중복을 줄입니다. */}
                    {[
                        { id: 'accountid', label: '아이디', type: 'text' },
                        { id: 'username', label: '닉네임', type: 'text' },
                        { id: 'email', label: '이메일', type: 'email' },
                        { id: 'password', label: '비밀번호', type: 'password' },
                        { id: 'confirmPw', label: '비밀번호 확인', type: 'password' },
                    ].map(field => (
                        <div className={style.form_group} key={field.id}>
                            <label htmlFor={field.id}>{field.label}</label>
                            <input
                                type={field.type}
                                id={field.id}
                                value={formData[field.id]}
                                onChange={handleChange}
                                required
                            />
                        </div>
                    ))}

                    {error && <p className={style.error_message}>{error}</p>}

                    <div className={style.button_group}>
                        <button className={style.submit_join_btn} type="submit" disabled={isLoading}>
                            {isLoading ? '가입 진행 중...' : '회원가입'}
                        </button>
                        <button 
                            type="button" 
                            className={style.to_login_btn} 
                            onClick={() => navigate('/login')}
                            disabled={isLoading}
                        >
                            이미 계정이 있으신가요? 로그인
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default Join;
