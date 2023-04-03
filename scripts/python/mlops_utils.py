import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_gpt_code_from_prompt(prompt, wrapper, model="gpt-3.5-turbo"):
  completion = openai.ChatCompletion.create(
    model=model,
    messages=[
      {"role": "user", "content": wrapper + prompt}
    ]
  )
  return completion.choices[0].message.content

  #"OPENAI_API_KEY": "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"