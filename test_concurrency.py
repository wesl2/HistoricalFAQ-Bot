import asyncio
import aiohttp
import time

API_URL = "http://localhost:8000"
QUESTION = "唐太宗的用人之道"
CONCURRENT_REQUESTS = 12  # 并发数

async def single_request(session, idx):
    t0 = time.time()
    try:
        async with session.post(
            f"{API_URL}/api/query",
            json={"question": f"{QUESTION} ({idx})"}
        ) as resp:
            data = await resp.json()
            latency = (time.time() - t0) * 1000
            error_code = data.get("error_code")
            status = "OK" if error_code is None else error_code
            return idx, latency, status, data.get("answer", "")[:30]
    except Exception as e:
        return idx, (time.time() - t0) * 1000, f"ERROR: {e}", ""

async def main():
    print(f"并发测试: {CONCURRENT_REQUESTS} 个请求同时发送")
    print(f"LLM 并发限制: 10（Semaphore）")
    t0 = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [single_request(session, i) for i in range(CONCURRENT_REQUESTS)]
        results = await asyncio.gather(*tasks)
    
    total_time = (time.time() - t0) * 1000
    
    print(f"\n总耗时: {total_time:.0f}ms")
    print(f"平均耗时: {sum(r[1] for r in results)/len(results):.0f}ms")
    print(f"最快: {min(r[1] for r in results):.0f}ms")
    print(f"最慢: {max(r[1] for r in results):.0f}ms")
    
    ok_count = sum(1 for r in results if r[2] == "OK")
    error_count = CONCURRENT_REQUESTS - ok_count
    print(f"\n成功: {ok_count} | 失败: {error_count}")
    
    for idx, latency, status, preview in results:
        marker = "✓" if status == "OK" else "✗"
        print(f"  {marker} [{idx:2d}] {latency:6.0f}ms | {status}")

if __name__ == "__main__":
    asyncio.run(main())
