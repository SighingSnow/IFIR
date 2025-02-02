import asyncio
import logging
import os
from typing import Any, List
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
import random
from time import sleep

import aiolimiter

import openai
from openai import AsyncOpenAI, OpenAIError, OpenAI
    
def aopenai_client():
    client = AsyncOpenAI()
    AsyncOpenAI.api_key = os.environ["OPENAI_API_KEY"]
    return client

def openai_client():
    client = OpenAI()
    OpenAI.api_key = os.environ["OPENAI_API_KEY"]
    return client

async def _throttled_openai_embeddings_acreate(
    client: AsyncOpenAI,
    model: str,
    messages,
    limiter: aiolimiter.AsyncLimiter,
):
    async with limiter:
        for _ in range(20):
            try:
                return await client.embeddings.create(
                    model=model,
                    input=messages,
                )
            except openai.BadRequestError as e:
                print(e)
                return None
            except OpenAIError as e:
                print(e)
                sleep(random.randint(5, 10))
        return None

async def generate_from_openai_embeddings_completion(
    client,
    messages,
    engine_name: str,
    requests_per_minute: int = 100,
):
    """Generate from OpenAI Embedding API.

    Args:
        messages: List of messages to proceed.
        engine_name: Engine name to use, see https://platform.openai.com/docs/models
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """    
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_embeddings_acreate(
            client,
            model=engine_name,
            messages=message,
            limiter=limiter,
        )
        for message in messages
    ]
    
    responses = await asyncio.gather(*async_responses)
    
    outputs = []
    for response in responses:
        if response:
            outputs.append(response.data[0].embedding)
        else:
            outputs.append("Invalid Message")
    return outputs


async def _throttled_openai_chat_completion_acreate(
    client: AsyncOpenAI,
    model: str,
    messages,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
    logprobs: bool = False,
    top_logprobs: int = 20,
):
    async with limiter:
        for _ in range(10):
            try:
                return await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs
                    # logprobs=5
                )
            except openai.BadRequestError as e:
                print(e)
                return None
            except OpenAIError as e:
                print(e)
                sleep(random.randint(5, 10))
        return None


async def generate_from_openai_chat_completion(
    client,
    messages,
    engine_name: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    requests_per_minute: int = 100,
):
    """Generate from OpenAI Chat Completion API.

    Args:
        messages: List of messages to proceed.
        engine_name: Engine name to use, see https://platform.openai.com/docs/models
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """    
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            client,
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses)
    
    outputs = []
    for response in responses:
        if response:
            outputs.append(response.choices[0].message.content)
        else:
            outputs.append("Invalid Message")
    return outputs


async def generate_from_openai_chat_completion_prob(
    client,
    messages,
    engine_name: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    logprobs: bool = False,
    top_logprobs: int = 20,
    requests_per_minute: int = 100,
):
    """Generate from OpenAI Chat Completion API.

    Args:
        messages: List of messages to proceed.
        engine_name: Engine name to use, see https://platform.openai.com/docs/models
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.
        

    Returns:
        List of generated responses.
    """    
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            client,
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            top_p=top_p,
            top_logprobs=top_logprobs,
            limiter=limiter,
        )
        for message in messages
    ]
    
    responses = await asyncio.gather(*async_responses)
    
    outputs = []
    for response in responses:
        if response:
            outputs.append(response)
        else:
            outputs.append("Invalid Message")
    return outputs
