# Historical FAQ Bot 服务启动指南

## 一、快速启动（开发调试）

```bash
cd /root/autodl-tmp/HistoricalFAQ-Bot
export API_KEY="sk-d722d9cb7edf49b5b9fe88bb37908162"
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**参数说明：**
| 参数 | 作用 |
|------|------|
| `--host 0.0.0.0` | 监听所有网卡（局域网/公网可访问） |
| `--port 8000` | 服务端口 |
| `--reload` | 代码修改后自动重启（开发专用） |
| `--workers 1` | 工作进程数（默认1，开发时别改） |

---

## 二、生产环境启动（后台常驻）

### 方式 1：screen 会话（推荐）

```bash
cd /root/autodl-tmp/HistoricalFAQ-Bot
export API_KEY="sk-d722d9cb7edf49b5b9fe88bb37908162"

# 创建 screen 会话
screen -dmS faq-bot bash -c 'export API_KEY="sk-d722d9cb7edf49b5b9fe88bb37908162"; export LANG=C.UTF-8; export LC_ALL=C.UTF-8; uvicorn src.api.main:app --host 0.0.0.0 --port 8000'

# 查看实时日志
screen -r faq-bot

# 退出查看（Ctrl+A 然后 D）
```

### 方式 2：nohup 后台运行

```bash
cd /root/autodl-tmp/HistoricalFAQ-Bot
export API_KEY="sk-d722d9cb7edf49b5b9fe88bb37908162"

nohup uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &

# 查看日志
tail -f logs/api.log
```

### 方式 3：systemd 服务（服务器开机自启）

创建 `/etc/systemd/system/faq-bot.service`：

```ini
[Unit]
Description=Historical FAQ Bot API
After=network.target postgresql.service

[Service]
Type=simple
User=root
WorkingDirectory=/root/autodl-tmp/HistoricalFAQ-Bot
Environment="API_KEY=sk-d722d9cb7edf49b5b9fe88bb37908162"
Environment="LANG=C.UTF-8"
Environment="LC_ALL=C.UTF-8"
ExecStart=/usr/bin/python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

启用并启动：
```bash
systemctl daemon-reload
systemctl enable faq-bot
systemctl start faq-bot
systemctl status faq-bot
```

---

## 三、环境变量清单

| 变量 | 必填 | 说明 |
|------|------|------|
| `API_KEY` | ✅ | DeepSeek API Key |
| `LANG` | ❌ | 系统编码（设为 `C.UTF-8` 防止中文乱码） |
| `LC_ALL` | ❌ | 同上 |
| `LLM_MODE` | ❌ | `api` 或 `local`，默认 `api` |
| `LLM_CONCURRENCY` | ❌ | LLM 并发限制，默认 `10` |
| `DATABASE_URL` | ❌ | PostgreSQL 连接串（默认读取 pg_config） |

---

## 四、Uvicorn 配置对照表

| 场景 | 命令 |
|------|------|
| **开发调试** | `uvicorn src.api.main:app --reload` |
| **本地测试** | `uvicorn src.api.main:app --host 127.0.0.1 --port 8000` |
| **局域网访问** | `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` |
| **多 worker 生产** | `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4` |

> ⚠️ `--workers` 多进程模式下不能使用 `--reload`

---

## 五、验证启动

```bash
# 健康检查
curl -s http://localhost:8000/api/health | python -m json.tool

# 查看 API 文档（浏览器打开）
http://localhost:8000/docs

# 前端页面
http://localhost:8000/frontend.html
```

---

## 六、常见问题

### Q1: 中文日志乱码？
```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
```

### Q2: 端口被占用？
```bash
# 查看占用 8000 端口的进程
lsof -i :8000

# 杀掉进程
kill -9 <PID>
```

### Q3: screen 会话 attached 无法进入？
```bash
# 强制 detach 旧连接，重新进入
screen -dr faq-bot
```

### Q4: 如何停止服务？
```bash
# screen 方式
screen -r faq-bot
# 按 Ctrl+C

# nohup 方式
pkill -f "uvicorn src.api.main:app"

# systemd 方式
systemctl stop faq-bot
```

---

## 七、生产环境推荐架构

```
用户浏览器
    ↓ HTTPS (443)
Nginx（SSL 证书 + 反向代理）
    ↓ HTTP (8000)
Uvicorn（FastAPI 应用）
    ↓
PostgreSQL (5432)
```

Nginx 配置示例：
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
