# fivesense/Dockerfile (최종 완성본)
FROM openjdk:21-slim

# 작업 디렉토리를 /app으로 설정
WORKDIR /app

# Gradle 빌드로 생성된 JAR 파일을 컨테이너 안으로 복사
# context가 ./fivesense 이므로, build 폴더를 바로 찾을 수 있음
COPY build/libs/*.jar app.jar

# 8080 포트를 외부에 노출
EXPOSE 8080

# 컨테이너가 시작될 때 실행할 명령어
ENTRYPOINT ["java","-jar","/app.jar"]
