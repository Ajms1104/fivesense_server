package com.example.fivesense.config; 

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class CorsConfig implements WebMvcConfigurer {

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**") // 모든 경로(/**)에 CORS 설정 적용
                .allowedOrigins("http://localhost:5173", "http://116.124.191.174:15019")
                .allowedMethods("GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS") // 허용할 HTTP 메서드 종류
                .allowedHeaders("*") // 모든 종류의 HTTP 헤더 허용
                .allowCredentials(true) // 자격 증명(쿠키, 토큰 등) 정보 전송 허용
                .maxAge(3600); // Pre-flight 요청 결과의 캐싱 시간(초) 설정
    }
}
