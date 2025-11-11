// src/pages/ChatPage/ChatPage.jsx (새 파일)
import React, { useState, useEffect, useRef } from "react";
import style from './chat.module.css'; 

import Sidebar from '../../Components/layout/Sidebar/Sidebar.jsx';
import Topbar from '../../Components/layout/Topbar/Topbar.jsx';

const SendIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2 .01 7z" fill="currentColor"/>
    </svg>
);
const ChatHeader = () => (
    <div className={style.header}>
        <h2 className={style.title}>오늘은 어떤 주식이 궁금하신가요?</h2>
        <p className={style.welcome_message}>
            안녕하세요! 주식 투자에 대해 궁금한 점이 있으시면 언제든 물어보세요.
        </p>
    </div>
);
const MessageList = ({ messages, isLoading, error, messagesEndRef }) => (
    <div className={style.message_container}>
        {messages.map((msg, index) => (
            <div key={index} className={`${style.chat_message} ${style[msg.type]}`}>
                <div className={style.message_content}>{msg.content}</div>
                <div className={style.message_timestamp}>
                    {new Date(msg.timestamp).toLocaleTimeString('ko-KR', {
                        hour: '2-digit', minute: '2-digit'
                    })}
                </div>
            </div>
        ))}
        {isLoading && (
            <div className={`${style.chat_message} ${style.ai} ${style.loading}`}>
                <div className={style.loading_dots}><span></span><span></span><span></span></div>
            </div>
        )}
        {error && (
            <div className={style.error_message}>
                <span className={style.error_icon}>⚠️</span> {error}
            </div>
        )}
        <div ref={messagesEndRef} />
    </div>
);
const ChatInputForm = ({ input, setInput, handleSubmit, isLoading }) => (
    <form className={style.chat_input_form} onSubmit={handleSubmit}>
        <input
            type="text"
            placeholder="메시지를 입력하세요..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                }
            }}
            disabled={isLoading}
            autoFocus
        />
        <button type="submit" className={style.send_button} disabled={isLoading || input.trim() === ""}>
            <SendIcon />
        </button>
    </form>
);

// 메인 채팅 페이지 컴포넌트 
const ChatPage = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [currentRoomId, setCurrentRoomId] = useState(null);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        fetchChatHistory();
    }, []);

    useEffect(() => {
        scrollToBottom();
    }, [messages, isLoading]);

    const fetchChatHistory = async () => {
        try {
            const response = await fetch('/api/chat/history');
            if (!response.ok) {
                console.error('채팅 기록을 불러오는데 실패했습니다.');
                return;
            }
            const data = await response.json();
            const sortedMessages = data
                .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
                .map(msg => ({
                    type: msg.type.toLowerCase(),
                    content: msg.content,
                    timestamp: msg.timestamp
                }));
            setMessages(sortedMessages);
        } catch (error) {
            console.error('채팅 기록 로딩 오류:', error);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (input.trim() === "") return;

        const userMessageContent = input.trim();
        const userMessage = {
            type: 'user', content: userMessageContent, timestamp: new Date().toISOString()
        };
        
        setMessages(prev => [...prev, userMessage]);
        setInput("");
        setIsLoading(true);
        setError(null);

        try {
            const url = currentRoomId ? `/api/chat/${currentRoomId}` : '/api/chat';
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMessageContent }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || '메시지 전송에 실패했습니다.');

            const aiMessage = {
                type: 'ai', content: data.aiResponse, timestamp: new Date().toISOString()
            };
            setMessages(prev => [...prev, aiMessage]);

            if (data.roomId && !currentRoomId) setCurrentRoomId(data.roomId);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className={style.page_grid_container}>
            <div className={style.sidebar_wrapper}>
                <Sidebar />
            </div>
            <div className={style.topbar_wrapper}>
                <Topbar />
            </div>
            
            <main className={style.main_content}>
                <div className={style.chat_window}>
                    {messages.length === 0 && !isLoading && <ChatHeader />}
                    <MessageList messages={messages} isLoading={isLoading} error={error} messagesEndRef={messagesEndRef} />
                </div>
                <div className={style.input_section}>
                    <ChatInputForm input={input} setInput={setInput} handleSubmit={handleSubmit} isLoading={isLoading} />
                    <p className={style.disclaimer}>
                        투자에 대한 모든 결과는 전적으로 개인에게 있으며 손해에 대해 FIVESENSE 에선 책임지지 않습니다
                    </p>
                </div>
            </main>
        </div>
    );
};

export default ChatPage;
