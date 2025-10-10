# frontend/Dockerfile 
# --- 1단계: 빌드 스테이지 ---
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install

# node_modules에 설치된 vite 스크립트에 실행 권한을 부여합니다.
RUN chmod +x ./node_modules/.bin/vite

COPY . .
RUN npm run build

# --- 2단계: 서빙 스테이지 ---
FROM nginx:stable-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
