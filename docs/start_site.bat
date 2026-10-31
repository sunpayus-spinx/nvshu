@echo off
rem 启动「女书拼音 -> 汉字释文」本地网页（浏览器直连识别服务需要 http 源，故用本地静态服务打开）
cd /d "%~dp0"
echo Starting local preview at http://127.0.0.1:8080/index.html ...
start "" "http://127.0.0.1:8080/index.html"
python -m http.server 8080
