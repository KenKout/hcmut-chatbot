from app.envs import settings
from openai import OpenAI, AsyncOpenAI

class LLM:
    def __init__(self):
        self.openai_client = OpenAI(
            api_key=settings.LLM_OPENAI_API_KEY,
            base_url=settings.LLM_OPENAI_BASE_URL,
        )

        self.async_openai_client = AsyncOpenAI(
            api_key=settings.LLM_OPENAI_API_KEY,
            base_url=settings.LLM_OPENAI_BASE_URL,
        )
        
    def build_prompt(self, query: str, context: str) -> str:
        prompt = f"""You are given a user query, some textual context and rules, all inside xml tags. You have to answer the query based on the context while respecting the rules.

<context>
[context]
</context>

<rules>
- If you don''t know, just say so.
- If you are not sure, ask for clarification.
- Answer in the same language as the user query.
- If the context appears unreadable or of poor quality, tell the user then answer as best as you can.
- If the answer is not in the context but you think you know the answer, explain that to the user then answer with your own knowledge.
- Answer directly and without using xml tags.
</rules>

<user_query>
[query]
</user_query>
"""
        prompt = prompt.replace("[context]", context).replace("[query]", query)
        return prompt

    def get_answer(self, query: str, context: str) -> str:
        system_prompt = self.build_prompt(query, context)
        response = self.openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            model=settings.LLM_OPENAI_MODEL
        )
        answer = response.choices[0].message.content.strip()
        return answer

    async def get_answer_async(self, query: str, context: str) -> str:
        system_prompt = self.build_prompt(query, context)
        response = await self.async_openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            model=settings.LLM_OPENAI_MODEL,
        )

        answer = response.choices[0].message.content.strip()
        return answer
    
    async def get_answer_async_stream(self, query: str, context: str):
        system_prompt = self.build_prompt(query, context)
        response = await self.async_openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            model=settings.LLM_OPENAI_MODEL,
            stream=True,
        )
        return response
