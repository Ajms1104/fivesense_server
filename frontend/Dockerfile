# --- 1단계: 빌드(Build) 스테이지 ---
# React 코드를 빌드하기 위한 Node.js 18 환경을 준비합니다.
FROM node:18-alpine AS builder

# 작업 디렉토리를 /app으로 설정합니다.
WORKDIR /app

# package.json과 package-lock.json 파일을 먼저 복사합니다.
COPY package*.json ./

# 의존성을 설치합니다.
RUN npm install

# 소스 코드를 전부 복사합니다.
COPY . .

# 프로덕션용으로 빌드합니다. (결과물은 /app/dist 폴더에 생성됩니다.)
RUN npm run build

# --- 2단계: 서빙(Serve) 스테이지 ---
# 빌드된 결과물(정적 파일)을 사용자에게 보여줄 웹 서버를 준비합니다.
FROM nginx:stable-alpine

# 빌드 스테이지(/app/dist)의 결과물을 Nginx의 기본 HTML 폴더로 복사합니다.
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx가 80번 포트를 외부에 노출시킵니다.
EXPOSE 80

# 컨테이너가 시작될 때 Nginx를 실행합니다.
CMD ["nginx", "-g", "daemon off;"]
