import os
os.environ["LLM_MODE"] = "api"
os.environ["API_PROVIDER"] = "deepseek"
os.environ["API_KEY"] = "sk-d722d9cb7edf49b5b9fe88bb37908162"

import asyncio
from src.llm.standard_llm_practice import StandardLLM

async def main():
    # 非流式
    # resp = await StandardLLM.ainvoke("李世民的丰功伟绩", mode="api")
    # print("非流式回答:", resp.content)
    # print("-" * 40)

    # 流式
    print("流式回答: ", end="", flush=True)
    async for chunk in StandardLLM.astream("大明王朝1566不是真实的历史 为何却感觉浓缩了千年封建王朝的一切故事", mode="api"):
        print(chunk.content, end="", flush=True)
    print()

asyncio.run(main())