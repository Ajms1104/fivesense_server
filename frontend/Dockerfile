# --- 1단계: 빌드 스테이지 ---
# 표준 node 이미지를 사용하여 OS 호환성 문제를 해결합니다.
FROM node:18 AS builder

WORKDIR /app

# package.json과 lock 파일을 먼저 복사하여 의존성 캐싱을 활용합니다.
COPY package*.json ./

# 의존성을 설치합니다.
RUN npm install

# 나머지 소스 코드를 복사합니다.
COPY . .

# 프로덕션용으로 빌드합니다.
RUN npm run build

FROM nginx:stable-alpine


COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
