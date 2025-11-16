#!/usr/bin/env python3
import time
import logging
import requests

# ========= 配置区 =========

# 基础 URL（不含路径），按你的配置来的：
BASE_URL = "http://della9.princeton.edu:6666"

# 如果你的服务路径是 /api/openai/v1/chat/completions，
# 把下面这一行改成 "/api/openai/v1/chat/completions"
CHAT_PATH = "/v1/chat/completions"

PROVER_MODEL = "Goedel-Prover-V2-8B-previous_ckpt"
INFORMAL_MODEL = "gpt-oss-120b"

# 间隔时间（秒），比如 300 秒 = 5 分钟 ping 一次
KEEPALIVE_INTERVAL = 300

# 如果你的本地服务不需要 API key 就留空
API_KEY = None  # 比如 vLLM 常用 "EMPTY"，如果需要你就填上

# ========= 日志设置 =========

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ========= 函数 =========
def ping_model(model_name: str, role: str):
    """
    给指定 model 发一个很短的聊天请求，丢掉内容，只看是否成功。
    role 用来标记是 prover 还是 informal，方便日志阅读。
    """
    url = BASE_URL + CHAT_PATH

    headers = {
        "Content-Type": "application/json",
    }
    if API_KEY is not None:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": f"You are a keep-alive {role} model. Reply with a short ok."
            },
            {
                "role": "user",
                "content": "ping"
            }
        ],
        "max_tokens": 8,
        "temperature": 0.0,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # 尽量从 OpenAI 风格的返回里取 content
        reply = ""
        try:
            msg = data["choices"][0]["message"]
            # 有些实现里 content 可能是 None，我们统一转成字符串
            reply = msg.get("content", "")
        except Exception:
            reply = str(data)

        # 一定先转成字符串再 replace，避免 NoneType 报错
        reply_str = str(reply).replace("\n", " ")[:80]
        logging.info("[{}] ping success, reply: {}".format(role, reply_str))
        return True

    except Exception as e:
        # 打印一些原始响应，方便以后调试
        raw = ""
        try:
            raw = resp.text[:200]
        except Exception:
            pass
        logging.error(f"[{role}] ping FAILED: {e}. Raw response: {raw}")
        return False



def main():
    logging.info("Starting keep-alive loop for prover_llm and informal_llm...")
    try:
        while True:
            # 先 ping prover_llm
            ping_model(PROVER_MODEL, role="prover_llm")

            # 再 ping informal_llm
            ping_model(INFORMAL_MODEL, role="informal_llm")

            # 休息一段时间
            logging.info(f"Sleeping {KEEPALIVE_INTERVAL} seconds before next ping...")
            time.sleep(KEEPALIVE_INTERVAL)
    except KeyboardInterrupt:
        logging.info("Received KeyboardInterrupt, exiting keep-alive loop.")


if __name__ == "__main__":
    main()
