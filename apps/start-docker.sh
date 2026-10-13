#!/bin/bash

# 启动 frontend 和 backend 项目的 Docker 脚本

set -e

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    启动 Frontend 和 Backend 项目    ${NC}"
echo -e "${BLUE}========================================${NC}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 函数：构建并启动 frontend
start_frontend() {
    echo -e "\n${GREEN}[1/4] 构建 Frontend Docker 镜像...${NC}"
    cd "$SCRIPT_DIR/frontend"
    docker build -t glm-ocr-frontend:latest .

    echo -e "${GREEN}[2/4] 启动 Frontend 容器...${NC}"
    # 停止并删除旧容器（如果存在）
    docker stop glm-ocr-frontend 2>/dev/null || true
    docker rm glm-ocr-frontend 2>/dev/null || true

    # 启动新容器
    docker run -d \
        --name glm-ocr-frontend \
        -p 127.0.0.1:3000:80 \
        --restart unless-stopped \
        glm-ocr-frontend:latest

    echo -e "${GREEN}✓ Frontend 已启动在 http://localhost:3000${NC}"
}

# 函数：构建并启动 backend
start_backend() {
    echo -e "\n${GREEN}[3/4] 构建 Backend Docker 镜像...${NC}"
    cd "$SCRIPT_DIR/backend"
    docker build -t glm-ocr-backend:latest .

    echo -e "${GREEN}[4/4] 启动 Backend 容器...${NC}"
    # 停止并删除旧容器（如果存在）
    docker stop glm-ocr-backend 2>/dev/null || true
    docker rm glm-ocr-backend 2>/dev/null || true

    # 启动新容器
    docker run -d \
        --name glm-ocr-backend \
        -p 127.0.0.1:8000:8000 \
        --restart unless-stopped \
        glm-ocr-backend:latest

    echo -e "${GREEN}✓ Backend 已启动在 http://localhost:8000${NC}"
}

# 主执行流程
start_frontend
start_backend

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✓ 所有服务已成功启动！${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Frontend:${NC} http://localhost:3000"
echo -e "${YELLOW}Backend:${NC}  http://localhost:8000"
echo -e "${YELLOW}API Docs:${NC} http://localhost:8000/docs"
echo -e "\n${YELLOW}查看日志命令：${NC}"
echo -e "  Frontend: docker logs -f glm-ocr-frontend"
echo -e "  Backend:  docker logs -f glm-ocr-backend"
echo -e "\n${YELLOW}停止服务命令：${NC}"
echo -e "  docker stop glm-ocr-frontend glm-ocr-backend"
echo -e "${BLUE}========================================${NC}"
