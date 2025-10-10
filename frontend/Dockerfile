
# 표준 node 이미지를 사용하여 OS 호환성 문제를 최소화합니다.
FROM node:18 AS builder

WORKDIR /app
COPY package*.json ./
RUN npm install

RUN chmod +x ./node_modules/.bin/vite

COPY . .
RUN npm run build

FROM nginx:stable-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
