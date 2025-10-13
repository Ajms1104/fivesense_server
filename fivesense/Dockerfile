# JDK와 Gradle이 모두 설치된 이미지를 사용하여, 소스코드를 .jar 파일로 빌드
FROM gradle:jdk21-alpine AS builder

# 작업 디렉토리 설정
WORKDIR /home/gradle/src

# 전체 소스 코드를 빌더 컨테이너 안으로 복사
COPY . .

# Gradle을 사용하여 애플리케이션을 빌드
RUN gradle build -x test --stacktrace

# 실제 애플리케이션 실행에 필요한 최소한의 JRE 환경만 사용
FROM eclipse-temurin:21-jre-alpine

WORKDIR /app

# '빌더' 스테이지에서 생성된 .jar 파일만 복사
COPY --from=builder /home/gradle/src/build/libs/*.jar app.jar

# 8080 포트
EXPOSE 8080

# 최종적으로 app.jar 파일을 실행
ENTRYPOINT ["java","-jar","/app.jar"]
