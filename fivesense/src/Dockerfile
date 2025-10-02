# 베이스 이미지 설정: 가볍고 안정적인 Java 21 실행 환경을 사용합니다.
FROM openjdk:21-slim

# 빌드된 JAR 파일의 경로를 변수로 지정합니다.
ARG JAR_FILE=build/libs/*.jar

# 빌드된 JAR 파일을 컨테이너 안으로 복사하고, 이름을 app.jar로 통일합니다.
COPY ${JAR_FILE} app.jar

# 이 컨테이너가 시작될 때 실행할 명령어를 지정합니다.
ENTRYPOINT ["java","-jar","/app.jar"]
